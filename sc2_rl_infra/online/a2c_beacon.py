"""A2C mínimo que APRENDE MoveToBeacon, visualizado EN VIVO sobre VNC (spike de Fase 3).

Una red conv pequeña (FullyConv, PyTorch) mira `feature_screen` y aprende DÓNDE mover
el marine (cabeza espacial screen×screen) más un crítico. A2C n-step en un solo env,
renderizando cada step en una ventana pygame de software (como `live_view`) para ver al
agente pasar de torpe a resolver el mapa en tiempo real. La cabecera muestra el reward
medio reciente, que debería subir de ~0-1 a ~20+ conforme aprende.

Es un ANTICIPO de la Fase 3 (RL online en minijuegos); no sustituye a las fases del plan.
RL es quisquilloso: si no converge, prueba a subir --entropy o bajar --lr.

Requisito: PyTorch en el env  ->  pip install torch   (CPU vale: la red es minúscula y
el cuello de botella es el propio SC2; la GPU apenas ayuda aquí).

Uso (en Brais, env `sc2-rl-infra` activo, VNC en :1 — ver tools/vnc.sh):
    DISPLAY=:1 python -m sc2_rl_infra.online.a2c_beacon
    DISPLAY=:1 python -m sc2_rl_infra.online.a2c_beacon --updates 5000 --fps 20
    DISPLAY=:1 python -m sc2_rl_infra.online.a2c_beacon --device cuda

Cierra la ventana (o Esc/Q) para detener; el reward medio se imprime en la terminal.
"""

import collections
import os

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
flags.DEFINE_float("lr", 7e-4, "Learning rate (RMSprop).")
flags.DEFINE_float("gamma", 0.99, "Factor de descuento.")
flags.DEFINE_float("entropy", 1e-3, "Coeficiente de entropía (exploración).")
flags.DEFINE_float("value_coef", 0.5, "Peso de la pérdida del crítico.")
flags.DEFINE_float("max_grad_norm", 0.5, "Recorte de norma de gradiente.")
flags.DEFINE_integer("screen", 84, "Resolución (px) de feature screen.")
flags.DEFINE_integer("minimap", 64, "Resolución (px) de feature minimap.")
flags.DEFINE_integer("step_mul", 8, "Game steps por agent step.")
flags.DEFINE_integer("fps", 0, "Cap de FPS del visor (0 = sin cap, entrena a tope).")
flags.DEFINE_integer("cell", 380, "Lado (px) de cada panel.")
flags.DEFINE_integer("log_every", 20, "Imprimir métricas cada N updates.")
flags.DEFINE_string("device", "cpu", "torch device: 'cpu' o 'cuda'.")

_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_SELECTED = features.SCREEN_FEATURES.selected.index

# --- visor (software, sin OpenGL; mismo enfoque que live_view) ---
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
    print(f"[a2c_beacon] SDL video driver = {pygame.display.get_driver()}")
    win = pygame.display.set_mode((2 * cell + 3 * MARGIN, HEADER_H + cell + LABEL_H + 2 * MARGIN))
    pygame.display.set_caption("sc2-rl-infra · A2C MoveToBeacon (entrenando, software)")
    return win


def render(win, font, obs, header, cell):
    win.fill(BG)
    win.blit(font.render(header, True, FG), (MARGIN, (HEADER_H - LABEL_H) // 2 + 2))
    fs = np.asarray(obs.observation["feature_screen"])
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


# --- modelo: FullyConv (encoder espacial -> política espacial + valor) ---
class FullyConv(nn.Module):
    def __init__(self, in_ch, size):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 16, 5, padding=2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.spatial = nn.Conv2d(32, 1, 1)     # logits espaciales -> dónde moverse
        self.value = nn.Linear(32, 1)          # crítico

    def forward(self, x):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        spatial_logits = self.spatial(h).flatten(1)        # (B, size*size)
        value = self.value(h.mean(dim=(2, 3))).squeeze(-1)  # (B,)
        return spatial_logits, value


def obs_to_tensor(obs, device):
    fs = np.asarray(obs.observation["feature_screen"])
    pr = (fs[_PLAYER_RELATIVE].astype(np.float32)) / 4.0    # 0..4 -> 0..1 (marine=.25, beacon=.75)
    sel = (fs[_SELECTED] > 0).astype(np.float32)
    x = np.stack([pr, sel], axis=0)[None]                   # (1, 2, size, size)
    return torch.as_tensor(x, device=device)


def select_if_needed(env, obs, win, font, cell):
    """MoveToBeacon necesita seleccionar el marine una vez por episodio."""
    while _MOVE_SCREEN not in obs.observation["available_actions"]:
        obs = env.step([actions.FUNCTIONS.select_army("select")])[0]
        render(win, font, obs, "seleccionando marine...", cell)
        if obs.last():
            break
    return obs


def main(unused_argv):
    device = FLAGS.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[a2c_beacon] cuda no disponible; usando cpu.")
        device = "cpu"

    size = FLAGS.screen
    model = FullyConv(in_ch=2, size=size).to(device)
    opt = torch.optim.RMSprop(model.parameters(), lr=FLAGS.lr, eps=1e-5)

    win = init_window(FLAGS.cell)
    font = pygame.font.Font(None, LABEL_H)
    clock = pygame.time.Clock()

    win.fill(BG)
    win.blit(font.render("Lanzando SC2...", True, FG), (MARGIN, MARGIN))
    pygame.display.flip()

    recent = collections.deque(maxlen=20)   # rewards de episodios recientes
    best = 0.0
    ep_reward, total_steps = 0.0, 0

    with sc2_env.SC2Env(
        map_name="MoveToBeacon",
        players=[sc2_env.Agent(sc2_env.Race.terran)],
        agent_interface_format=features.AgentInterfaceFormat(
            feature_dimensions=features.Dimensions(screen=size, minimap=FLAGS.minimap),
        ),
        step_mul=FLAGS.step_mul,
        game_steps_per_episode=0,
        visualize=False,
    ) as env:
        obs = env.reset()[0]
        obs = select_if_needed(env, obs, win, font, FLAGS.cell)

        running = True
        for update in range(1, FLAGS.updates + 1):
            if not running:
                break
            logps, values, entropies, rewards, dones = [], [], [], [], []

            for _ in range(FLAGS.nsteps):
                obs = select_if_needed(env, obs, win, font, FLAGS.cell)
                logits, value = model(obs_to_tensor(obs, device))
                dist = torch.distributions.Categorical(logits=logits)
                idx = dist.sample()
                y, x = int(idx.item()) // size, int(idx.item()) % size

                obs = env.step([actions.FUNCTIONS.Move_screen("now", [x, y])])[0]
                r, done = float(obs.reward), obs.last()
                total_steps += 1
                ep_reward += r

                logps.append(dist.log_prob(idx).squeeze(0))
                values.append(value.squeeze(0))
                entropies.append(dist.entropy().squeeze(0))
                rewards.append(r)
                dones.append(1.0 if done else 0.0)

                mean_r = (sum(recent) / len(recent)) if recent else 0.0
                header = (f"A2C MoveToBeacon | update {update}/{FLAGS.updates} | "
                          f"reward medio(20): {mean_r:.1f} | mejor: {best:.0f} | steps {total_steps}")
                render(win, font, obs, header, FLAGS.cell)
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

            # Bootstrap del valor del último estado (0 si terminal).
            with torch.no_grad():
                _, last_v = model(obs_to_tensor(obs, device))
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
                print(f"[a2c_beacon] update {update:>5} | reward medio(20) {mean_r:6.2f} | "
                      f"mejor {best:5.0f} | loss {loss.item():8.3f} | steps {total_steps}")

    pygame.quit()


if __name__ == "__main__":
    app.run(main)
