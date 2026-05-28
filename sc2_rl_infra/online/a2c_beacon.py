"""A2C que APRENDE MoveToBeacon — dos modos: VISOR (1 env) o PARALELO headless (N envs).

Una red conv pequeña (FullyConv, PyTorch) mira `feature_screen` y aprende DÓNDE mover
el marine (cabeza espacial screen×screen) más un crítico. A2C n-step.

Dos modos según `--num_envs`:

* **`--num_envs 1`** (default): un solo env, render en una ventana pygame de software
  (Brais VNC) para ver al agente pasar de torpe a resolver el mapa. Modo "spike" / debug
  visual.
* **`--num_envs N`** (N>1): N envs SC2 en paralelo (multiprocessing spawn), **headless**:
  sin visor, forward batched, gradientes más estables. Modo "entrenamiento serio". El
  sweet spot en Brais es N=8 (~2766 steps/s al 65% CPU; RESULTS §6); N=12 maximiza
  throughput pero deja CPU al 93%.

Reward shaping ON por defecto (`--shaped`): potential-based por distancia marine→beacon.
Da señal densa desde el primer step y rompe el arranque frío. El reward que se MUESTRA
y se compara con los baselines es SIEMPRE el nativo, no el del shaping.

Checkpoints (`--save_checkpoint_every N` + `--checkpoint_dir`): guarda
{model, optimizer, update, total_steps, best, recent} en .pt cada N updates y al terminar
(o al Ctrl+C). Reanuda con `--load_checkpoint <ruta>`. El .pt es portable: cárgalo en un
agente custom para verlo actuar con visor (Brais) o en Windows con render 3D (NOTES §7.4).

Replays (`--save_replay_every N` + `--replay_dir`): pasa `save_replay_episodes=N` al
SC2Env; cada env guarda un .SC2Replay cada N episodios suyos (con N envs se acumulan
rápido — sube el valor o ponlo a 0).

Es ANTICIPO de Fase 3; no sustituye al plan. Fase 1 (behaviour cloning) sigue siendo
el siguiente paso oficial.

Requisito: PyTorch en el env → `pip install torch` (CPU vale: la red es minúscula).

Uso (Brais, env `sc2-rl-infra` activo):

    # Visor en vivo (1 env, requiere VNC en :1, tools/vnc.sh):
    DISPLAY=:1 python -m sc2_rl_infra.online.a2c_beacon --fps 30

    # Entrenamiento paralelo (8 envs, headless, sin VNC):
    python -m sc2_rl_infra.online.a2c_beacon --num_envs 8 \\
        --save_checkpoint_every 100 --save_replay_every 200

    # Reanudar de un checkpoint:
    python -m sc2_rl_infra.online.a2c_beacon --num_envs 8 \\
        --load_checkpoint checkpoints/a2c_beacon/checkpoint_001000.pt

Cierra la ventana (o Esc/Q) para detener en modo visor; Ctrl+C en modo paralelo (guarda
el último checkpoint al salir).
"""

import collections
import math
import multiprocessing
import os
import sys
import time

import numpy as np
from absl import app, flags
from pysc2.env import sc2_env
from pysc2.lib import actions, features

os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
import pygame  # noqa: E402  (tras fijar SDL_AUDIODRIVER)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

FLAGS = flags.FLAGS
flags.DEFINE_integer("updates", 4000, "Número de updates A2C.")
flags.DEFINE_integer("nsteps", 16, "Decisiones por update (rollout n-step).")
flags.DEFINE_integer("num_envs", 1, "Envs SC2 en paralelo (1 = visor; >1 = headless paralelo).")
flags.DEFINE_float("lr", 7e-4, "Learning rate (RMSprop).")
flags.DEFINE_float("gamma", 0.99, "Factor de descuento.")
flags.DEFINE_float("entropy", 1e-3, "Coeficiente de entropía (exploración).")
flags.DEFINE_float("value_coef", 0.5, "Peso de la pérdida del crítico.")
flags.DEFINE_float("max_grad_norm", 0.5, "Recorte de norma de gradiente.")
flags.DEFINE_integer("screen", 84, "Resolución (px) de feature screen.")
flags.DEFINE_integer("minimap", 64, "Resolución (px) de feature minimap.")
flags.DEFINE_integer("step_mul", 8, "Game steps por agent step.")
flags.DEFINE_integer("fps", 0, "Cap de FPS del visor (0 = sin cap). Solo modo 1 env.")
flags.DEFINE_integer("cell", 380, "Lado (px) de cada panel del visor. Solo modo 1 env.")
flags.DEFINE_integer("log_every", 20, "Imprimir métricas cada N updates.")
flags.DEFINE_string("device", "cpu", "torch device: 'cpu' o 'cuda'.")
flags.DEFINE_bool("shaped", True, "Reward shaping potential-based por distancia al beacon "
                  "(señal densa que rompe el arranque frío). --noshaped usa solo el reward nativo.")
flags.DEFINE_float("shape_coef", 1.0, "Peso del shaping por distancia (súbelo si sigue plano).")
flags.DEFINE_integer("render_every", 1, "Renderiza 1 de cada N steps del visor. Solo modo 1 env.")
flags.DEFINE_integer("save_checkpoint_every", 0,
                     "Guarda checkpoint .pt cada N updates (0 = off). También al salir.")
flags.DEFINE_string("checkpoint_dir", "checkpoints/a2c_beacon",
                    "Carpeta para los .pt (relativa al cwd o absoluta).")
flags.DEFINE_string("load_checkpoint", "",
                    "Ruta a un .pt para reanudar entrenamiento (vacío = empezar de cero).")
flags.DEFINE_integer("save_replay_every", 0,
                     "save_replay_episodes del SC2Env: un .SC2Replay cada N episodios POR env "
                     "(con N envs se acumulan; 0 = off).")
flags.DEFINE_string("replay_dir", "sc2-rl-infra/a2c",
                    "Carpeta de replays bajo ~/StarCraftII/Replays/ (o absoluta).")

_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_SELECTED = features.SCREEN_FEATURES.selected.index
_PR_SELF = int(features.PlayerRelative.SELF)        # 1  (el marine)
_PR_NEUTRAL = int(features.PlayerRelative.NEUTRAL)  # 3  (el beacon)


# --- modelo: FullyConv (encoder espacial -> política espacial + valor) ---
class FullyConv(nn.Module):
    def __init__(self, in_ch, size):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 16, 5, padding=2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.spatial = nn.Conv2d(32, 1, 1)      # logits espaciales -> dónde moverse
        self.value = nn.Linear(32, 1)           # crítico

    def forward(self, x):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        spatial_logits = self.spatial(h).flatten(1)         # (B, size*size)
        value = self.value(h.mean(dim=(2, 3))).squeeze(-1)  # (B,)
        return spatial_logits, value


def obs_to_tensor(fs, device):
    """Polimórfico: fs ndarray (L, H, W) -> (1, 2, H, W); o (N, L, H, W) -> (N, 2, H, W)."""
    fs = np.asarray(fs)
    if fs.ndim == 3:
        fs = fs[None]
    pr = fs[:, _PLAYER_RELATIVE].astype(np.float32) / 4.0   # marine=0.25, beacon=0.75
    sel = (fs[:, _SELECTED] > 0).astype(np.float32)
    x = np.stack([pr, sel], axis=1)
    return torch.as_tensor(x, device=device)


def beacon_distance(fs, size):
    """Distancia euclídea normalizada (0..1) entre marine (SELF) y beacon (NEUTRAL).
    fs: ndarray (L, H, W). Devuelve None si alguno no está en pantalla.
    """
    pr = np.asarray(fs)[_PLAYER_RELATIVE]
    sy, sx = np.nonzero(pr == _PR_SELF)
    by, bx = np.nonzero(pr == _PR_NEUTRAL)
    if sx.size == 0 or bx.size == 0:
        return None
    d = math.hypot(sy.mean() - by.mean(), sx.mean() - bx.mean())
    return d / (math.sqrt(2.0) * size)


# --- checkpoint helpers ---
def save_checkpoint(path, model, opt, update, total_steps, best, recent):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "update": int(update),
        "total_steps": int(total_steps),
        "best": float(best),
        "recent": list(recent),
    }, path)
    print(f"[a2c_beacon] checkpoint -> {path}", flush=True)


def load_checkpoint(path, model, opt):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    if opt is not None and "optimizer" in ckpt:
        opt.load_state_dict(ckpt["optimizer"])
    update = int(ckpt.get("update", 0))
    total_steps = int(ckpt.get("total_steps", 0))
    best = float(ckpt.get("best", 0.0))
    recent = collections.deque(ckpt.get("recent", []), maxlen=20)
    print(f"[a2c_beacon] checkpoint cargado <- {path} "
          f"(update {update}, best {best:.2f})", flush=True)
    return update, total_steps, best, recent


# --- visor (modo 1 env, software, sin OpenGL; mismo enfoque que live_view) ---
MARGIN, LABEL_H, HEADER_H = 8, 22, 30
BG, FG = (24, 24, 28), (230, 230, 230)


def colorize(plane, feature):
    """(H, W) de enteros -> (H, W, 3) uint8 con la paleta oficial de PySC2."""
    try:
        palette = np.asarray(feature.palette)
        idx = np.clip(plane, 0, palette.shape[0] - 1).astype(np.int64)
        return palette[idx].astype(np.uint8)
    except Exception:
        scale = max(int(getattr(feature, "scale", 256)) - 1, 1)
        g = (np.clip(plane, 0, scale) * (255.0 / scale)).astype(np.uint8)
        return np.dstack([g, g, g])


def plane_surface(rgb, cell):
    surf = pygame.surfarray.make_surface(np.ascontiguousarray(rgb.transpose(1, 0, 2)))
    return pygame.transform.scale(surf, (cell, cell))


def init_window(cell):
    os.environ["SDL_VIDEODRIVER"] = "x11"  # pysc2 deja SDL en 'dummy' (invisible) en headless
    pygame.display.quit()
    pygame.display.init()
    pygame.font.init()
    print(f"[a2c_beacon] SDL video driver = {pygame.display.get_driver()}", flush=True)
    win = pygame.display.set_mode((2 * cell + 3 * MARGIN, HEADER_H + cell + LABEL_H + 2 * MARGIN))
    pygame.display.set_caption("sc2-rl-infra · A2C MoveToBeacon (entrenando, software)")
    return win


def render(win, font, fs, header, cell):
    """fs: ndarray (L, H, W) — feature_screen completo."""
    win.fill(BG)
    win.blit(font.render(header, True, FG), (MARGIN, (HEADER_H - LABEL_H) // 2 + 2))
    for i, feat in enumerate((features.SCREEN_FEATURES.player_relative,
                              features.SCREEN_FEATURES.unit_type)):
        x = MARGIN + i * (cell + MARGIN)
        win.blit(font.render(f"screen:{feat.name}", True, FG), (x, HEADER_H))
        win.blit(plane_surface(colorize(fs[feat.index], feat), cell), (x, HEADER_H + LABEL_H))
    pygame.display.flip()


def keep_running():
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            return False
        if e.type == pygame.KEYDOWN and e.key in (pygame.K_ESCAPE, pygame.K_q):
            return False
    return True


def select_if_needed(env, obs, win, font, cell):
    """Selecciona el marine al inicio de cada episodio (modo 1 env, con visor)."""
    while _MOVE_SCREEN not in obs.observation["available_actions"]:
        obs = env.step([actions.FUNCTIONS.select_army("select")])[0]
        render(win, font, np.asarray(obs.observation["feature_screen"]),
               "seleccionando marine...", cell)
        if obs.last():
            break
    return obs


# --- worker para envs en paralelo (multiprocessing spawn) ---
def _ensure_marine_selected(env, obs):
    """Igual que select_if_needed pero sin visor (para el worker headless)."""
    while _MOVE_SCREEN not in obs.observation["available_actions"]:
        obs = env.step([actions.FUNCTIONS.select_army("select")])[0]
        if obs.last():
            break
    return obs


def _worker(child_remote, parent_remote, env_cfg, worker_id):
    """Subproceso que corre un SC2Env y dialoga con el trainer por pipe.

    Protocolo (recibido del padre):
        ("step", x, y)  -> Move_screen([x, y]); auto-reset al terminar episodio.
                           Envía (feature_screen_array, reward_native, done).
        ("close",)      -> cierra el env y termina.

    Al arrancar envía el feature_screen inicial (tras el select inicial del marine).
    """
    parent_remote.close()  # solo lo usa el padre
    env = None
    try:
        # multiprocessing spawn no hereda parsing de flags absl; sin esto PySC2 lanza
        # UnparsedFlagAccessError. Mismo workaround que benchmark_parallel.py (RESULTS §7).
        flags.FLAGS.mark_as_parsed()

        env_kwargs = dict(
            map_name=env_cfg["map_name"],
            players=[sc2_env.Agent(sc2_env.Race.terran)],
            agent_interface_format=features.AgentInterfaceFormat(
                feature_dimensions=features.Dimensions(
                    screen=env_cfg["screen"], minimap=env_cfg["minimap"]
                ),
            ),
            step_mul=env_cfg["step_mul"],
            game_steps_per_episode=0,
            visualize=False,
        )
        if env_cfg.get("save_replay_every", 0) > 0:
            env_kwargs["save_replay_episodes"] = env_cfg["save_replay_every"]
            env_kwargs["replay_dir"] = env_cfg["replay_dir"]

        env = sc2_env.SC2Env(**env_kwargs)
        obs = env.reset()[0]
        obs = _ensure_marine_selected(env, obs)
        child_remote.send(np.asarray(obs.observation["feature_screen"]))

        while True:
            cmd = child_remote.recv()
            if cmd[0] == "step":
                _, x, y = cmd
                obs = env.step([actions.FUNCTIONS.Move_screen("now", [int(x), int(y)])])[0]
                r = float(obs.reward)
                done = bool(obs.last())
                if done:
                    obs = env.reset()[0]
                obs = _ensure_marine_selected(env, obs)
                child_remote.send((np.asarray(obs.observation["feature_screen"]), r, done))
            elif cmd[0] == "close":
                break
            # comandos desconocidos: ignorar
    except (KeyboardInterrupt, EOFError, BrokenPipeError):
        pass
    except Exception as e:
        import traceback
        sys.stderr.write(f"[a2c_beacon worker {worker_id}] ERROR: {e}\n")
        traceback.print_exc()
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        try:
            child_remote.close()
        except Exception:
            pass


class VecSC2Env:
    """N envs SC2 en subprocesos (spawn), comunicados por Pipe.

    Cada worker es autónomo: reset, selecciona marine, ejecuta acciones y se auto-resetea
    al terminar episodio. El padre solo envía Move_screen([x, y]) y recibe (fs, r, done).
    """

    def __init__(self, num_envs, env_cfg, startup_timeout=120):
        self.num_envs = num_envs
        ctx = multiprocessing.get_context("spawn")

        self.parent_remotes = []
        child_remotes = []
        for _ in range(num_envs):
            parent_remote, child_remote = ctx.Pipe()
            self.parent_remotes.append(parent_remote)
            child_remotes.append(child_remote)

        self.procs = []
        for i, (parent_remote, child_remote) in enumerate(
                zip(self.parent_remotes, child_remotes)):
            p = ctx.Process(
                target=_worker,
                args=(child_remote, parent_remote, env_cfg, i),
                daemon=True,
            )
            p.start()
            self.procs.append(p)

        # El padre cierra su copia de los child remotes.
        for c in child_remotes:
            c.close()

        # Recibe el feature_screen inicial de cada worker, con timeout para detectar
        # workers muertos (SC2 que no arrancó, etc.) en vez de quedarse colgado.
        self.last_obs = []
        for i, remote in enumerate(self.parent_remotes):
            if not remote.poll(timeout=startup_timeout):
                raise RuntimeError(
                    f"VecSC2Env: worker {i} no envió obs inicial en {startup_timeout}s "
                    f"(¿SC2 falló al arrancar?)."
                )
            self.last_obs.append(remote.recv())

    def reset(self):
        # Los workers se auto-resetean; este reset devuelve el último estado conocido
        # (estado inicial al arrancar o el último new_fs recibido).
        return list(self.last_obs)

    def step(self, actions_list):
        """actions_list: list[(x, y)] por worker. Devuelve (list[fs], list[r], list[done])."""
        for remote, (x, y) in zip(self.parent_remotes, actions_list):
            remote.send(("step", x, y))
        new_fs, rewards, dones = [], [], []
        for remote in self.parent_remotes:
            fs, r, d = remote.recv()
            new_fs.append(fs)
            rewards.append(r)
            dones.append(d)
        self.last_obs = new_fs
        return new_fs, rewards, dones

    def close(self):
        for remote in self.parent_remotes:
            try:
                remote.send(("close",))
            except Exception:
                pass
        for p in self.procs:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()
                p.join(timeout=2)
        for remote in self.parent_remotes:
            try:
                remote.close()
            except Exception:
                pass


# --- modo 1 env con visor (debug visual) ---
def run_single(device):
    size = FLAGS.screen
    model = FullyConv(in_ch=2, size=size).to(device)
    opt = torch.optim.RMSprop(model.parameters(), lr=FLAGS.lr, eps=1e-5)

    start_update, total_steps, best = 0, 0, 0.0
    recent = collections.deque(maxlen=20)
    if FLAGS.load_checkpoint:
        start_update, total_steps, best, recent = load_checkpoint(
            FLAGS.load_checkpoint, model, opt)

    win = init_window(FLAGS.cell)
    font = pygame.font.Font(None, LABEL_H)
    clock = pygame.time.Clock()

    win.fill(BG)
    win.blit(font.render("Lanzando SC2...", True, FG), (MARGIN, MARGIN))
    pygame.display.flip()

    ep_reward = 0.0
    update = start_update  # por si el bucle no se ejecuta antes del finally

    env_kwargs = dict(
        map_name="MoveToBeacon",
        players=[sc2_env.Agent(sc2_env.Race.terran)],
        agent_interface_format=features.AgentInterfaceFormat(
            feature_dimensions=features.Dimensions(screen=size, minimap=FLAGS.minimap),
        ),
        step_mul=FLAGS.step_mul,
        game_steps_per_episode=0,
        visualize=False,
    )
    if FLAGS.save_replay_every > 0:
        env_kwargs["save_replay_episodes"] = FLAGS.save_replay_every
        env_kwargs["replay_dir"] = FLAGS.replay_dir

    try:
        with sc2_env.SC2Env(**env_kwargs) as env:
            obs = env.reset()[0]
            obs = select_if_needed(env, obs, win, font, FLAGS.cell)

            running = True
            for update in range(start_update + 1, FLAGS.updates + 1):
                if not running:
                    break
                logps, values, entropies, rewards, dones = [], [], [], [], []

                for _ in range(FLAGS.nsteps):
                    obs = select_if_needed(env, obs, win, font, FLAGS.cell)
                    fs = np.asarray(obs.observation["feature_screen"])
                    d_before = beacon_distance(fs, size) if FLAGS.shaped else None
                    logits, value = model(obs_to_tensor(fs, device))
                    dist = torch.distributions.Categorical(logits=logits)
                    idx = dist.sample()
                    y, x = int(idx.item()) // size, int(idx.item()) % size

                    obs = env.step([actions.FUNCTIONS.Move_screen("now", [x, y])])[0]
                    fs_new = np.asarray(obs.observation["feature_screen"])
                    r_native, done = float(obs.reward), obs.last()
                    total_steps += 1
                    ep_reward += r_native   # se reporta SIEMPRE el nativo (comparable a baselines)

                    # Shaping potential-based (Ng 1999): premia acercarse al beacon. Se salta
                    # al tocarlo (reaparece lejos, ese salto no es "alejarse") y al terminar.
                    r = r_native
                    if FLAGS.shaped and not done and r_native == 0.0 and d_before is not None:
                        d_after = beacon_distance(fs_new, size)
                        if d_after is not None:
                            r += FLAGS.shape_coef * (FLAGS.gamma * (-d_after) - (-d_before))

                    logps.append(dist.log_prob(idx).squeeze(0))
                    values.append(value.squeeze(0))
                    entropies.append(dist.entropy().squeeze(0))
                    rewards.append(r)
                    dones.append(1.0 if done else 0.0)

                    if FLAGS.render_every > 0 and total_steps % FLAGS.render_every == 0:
                        mean_r = (sum(recent) / len(recent)) if recent else 0.0
                        header = (f"A2C MoveToBeacon | update {update}/{FLAGS.updates} | "
                                  f"reward medio(20): {mean_r:.1f} | mejor: {best:.0f} | "
                                  f"steps {total_steps}")
                        render(win, font, fs_new, header, FLAGS.cell)
                    if not keep_running():
                        running = False
                        break
                    if FLAGS.fps > 0:
                        clock.tick(FLAGS.fps)

                    if done:
                        recent.append(ep_reward)
                        best = max(best, ep_reward)
                        ep_reward = 0.0
                        obs = env.reset()[0]
                        obs = select_if_needed(env, obs, win, font, FLAGS.cell)

                if not logps:
                    continue

                with torch.no_grad():
                    last_fs = np.asarray(obs.observation["feature_screen"])
                    _, last_v = model(obs_to_tensor(last_fs, device))
                    R = float(last_v.item()) * (0.0 if dones[-1] else 1.0)
                returns = []
                for r, d in zip(reversed(rewards), reversed(dones)):
                    R = r + FLAGS.gamma * R * (1.0 - d)
                    returns.insert(0, R)

                returns = torch.tensor(returns, dtype=torch.float32, device=device)
                values = torch.stack(values)
                logps = torch.stack(logps)
                entropies = torch.stack(entropies)
                advantages = returns - values.detach()

                policy_loss = -(logps * advantages).mean()
                value_loss = F.mse_loss(values, returns)
                loss = policy_loss + FLAGS.value_coef * value_loss - FLAGS.entropy * entropies.mean()

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), FLAGS.max_grad_norm)
                opt.step()

                if update % FLAGS.log_every == 0:
                    mean_r = (sum(recent) / len(recent)) if recent else 0.0
                    print(f"[a2c_beacon] update {update:>5} | envs 1 | "
                          f"reward medio(20) {mean_r:6.2f} | mejor {best:5.0f} | "
                          f"loss {loss.item():8.3f} | steps {total_steps}", flush=True)

                if (FLAGS.save_checkpoint_every > 0
                        and update % FLAGS.save_checkpoint_every == 0):
                    path = os.path.join(FLAGS.checkpoint_dir, f"checkpoint_{update:06d}.pt")
                    save_checkpoint(path, model, opt, update, total_steps, best, recent)
    finally:
        if FLAGS.save_checkpoint_every > 0 and update > start_update:
            try:
                path = os.path.join(FLAGS.checkpoint_dir,
                                    f"checkpoint_final_{update:06d}.pt")
                save_checkpoint(path, model, opt, update, total_steps, best, recent)
            except Exception as e:
                print(f"[a2c_beacon] guardado final del checkpoint falló: {e}", flush=True)
        pygame.quit()


# --- modo N envs en paralelo, headless (entrenamiento serio) ---
def run_parallel(device):
    size = FLAGS.screen
    num_envs = FLAGS.num_envs

    model = FullyConv(in_ch=2, size=size).to(device)
    opt = torch.optim.RMSprop(model.parameters(), lr=FLAGS.lr, eps=1e-5)

    start_update, total_steps, best = 0, 0, 0.0
    recent = collections.deque(maxlen=20)
    if FLAGS.load_checkpoint:
        start_update, total_steps, best, recent = load_checkpoint(
            FLAGS.load_checkpoint, model, opt)

    env_cfg = dict(
        map_name="MoveToBeacon",
        screen=size,
        minimap=FLAGS.minimap,
        step_mul=FLAGS.step_mul,
        save_replay_every=FLAGS.save_replay_every,
        replay_dir=FLAGS.replay_dir,
    )

    print(f"[a2c_beacon] lanzando {num_envs} envs SC2 en paralelo (headless)...", flush=True)
    t_launch = time.time()
    vec = VecSC2Env(num_envs, env_cfg)
    print(f"[a2c_beacon] {num_envs} envs listos en {time.time() - t_launch:.1f}s.", flush=True)

    ep_reward = [0.0] * num_envs
    obs_fs = vec.reset()
    update = start_update
    t0 = time.time()

    try:
        for update in range(start_update + 1, FLAGS.updates + 1):
            logps_buf, values_buf, ent_buf, rew_buf, done_buf = [], [], [], [], []

            for _ in range(FLAGS.nsteps):
                d_before = ([beacon_distance(fs, size) for fs in obs_fs]
                            if FLAGS.shaped else [None] * num_envs)

                inp = obs_to_tensor(np.stack(obs_fs, axis=0), device)
                logits, values = model(inp)
                dist = torch.distributions.Categorical(logits=logits)
                idx = dist.sample()
                logp = dist.log_prob(idx)
                ent = dist.entropy()

                idx_np = idx.cpu().numpy()
                ys = (idx_np // size).tolist()
                xs = (idx_np % size).tolist()
                actions_list = list(zip(xs, ys))

                new_obs_fs, rewards_native, dones = vec.step(actions_list)
                total_steps += num_envs

                # Tracking de episodios por env, en reward NATIVO (comparable a baselines).
                for i in range(num_envs):
                    ep_reward[i] += rewards_native[i]
                    if dones[i]:
                        recent.append(ep_reward[i])
                        if ep_reward[i] > best:
                            best = ep_reward[i]
                        ep_reward[i] = 0.0

                # Reward de entrenamiento (con shaping por env si aplica).
                train_rewards = np.array(rewards_native, dtype=np.float32)
                if FLAGS.shaped:
                    for i in range(num_envs):
                        if (not dones[i] and rewards_native[i] == 0.0
                                and d_before[i] is not None):
                            d_after = beacon_distance(new_obs_fs[i], size)
                            if d_after is not None:
                                train_rewards[i] += FLAGS.shape_coef * (
                                    FLAGS.gamma * (-d_after) - (-d_before[i])
                                )

                logps_buf.append(logp)
                values_buf.append(values)
                ent_buf.append(ent)
                rew_buf.append(torch.as_tensor(train_rewards, device=device))
                done_buf.append(torch.as_tensor(
                    np.array(dones, dtype=np.float32), device=device))

                obs_fs = new_obs_fs

            # Bootstrap: V del estado tras el último step, masked por done[T-1].
            with torch.no_grad():
                _, last_v = model(obs_to_tensor(np.stack(obs_fs, axis=0), device))

            returns = torch.zeros(FLAGS.nsteps, num_envs, device=device)
            R = last_v * (1.0 - done_buf[-1])
            for t in reversed(range(FLAGS.nsteps)):
                R = rew_buf[t] + FLAGS.gamma * R * (1.0 - done_buf[t])
                returns[t] = R

            values_t = torch.stack(values_buf)         # (T, N)
            logps_t = torch.stack(logps_buf)
            entropies_t = torch.stack(ent_buf)
            advantages = returns - values_t.detach()

            policy_loss = -(logps_t * advantages).mean()
            value_loss = F.mse_loss(values_t, returns)
            loss = (policy_loss + FLAGS.value_coef * value_loss
                    - FLAGS.entropy * entropies_t.mean())

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), FLAGS.max_grad_norm)
            opt.step()

            if update % FLAGS.log_every == 0:
                mean_r = (sum(recent) / len(recent)) if recent else 0.0
                elapsed = time.time() - t0
                sps = total_steps / max(elapsed, 1e-6)
                print(f"[a2c_beacon] update {update:>5} | envs {num_envs} | "
                      f"reward medio(20) {mean_r:6.2f} | mejor {best:5.0f} | "
                      f"loss {loss.item():8.3f} | steps {total_steps} | "
                      f"{sps:6.0f} step/s", flush=True)

            if (FLAGS.save_checkpoint_every > 0
                    and update % FLAGS.save_checkpoint_every == 0):
                path = os.path.join(FLAGS.checkpoint_dir, f"checkpoint_{update:06d}.pt")
                save_checkpoint(path, model, opt, update, total_steps, best, recent)
    except KeyboardInterrupt:
        print("\n[a2c_beacon] interrumpido (Ctrl+C); guardando último checkpoint...",
              flush=True)
    finally:
        if FLAGS.save_checkpoint_every > 0 and update > start_update:
            try:
                path = os.path.join(FLAGS.checkpoint_dir,
                                    f"checkpoint_final_{update:06d}.pt")
                save_checkpoint(path, model, opt, update, total_steps, best, recent)
            except Exception as e:
                print(f"[a2c_beacon] guardado final del checkpoint falló: {e}", flush=True)
        try:
            vec.close()
        except Exception:
            pass


def main(unused_argv):
    device = FLAGS.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[a2c_beacon] cuda no disponible; usando cpu.", flush=True)
        device = "cpu"

    if FLAGS.num_envs < 1:
        raise ValueError(f"--num_envs debe ser >= 1, dado {FLAGS.num_envs}.")
    if FLAGS.num_envs == 1:
        run_single(device)
    else:
        run_parallel(device)


if __name__ == "__main__":
    app.run(main)
