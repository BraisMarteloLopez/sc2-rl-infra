"""Visor en vivo por software (sin OpenGL propio), pensado para VNC.

Dos modos:
- Feature layers (por defecto): pinta las capas de PySC2 con su paleta oficial.
- RGB (`--rgb`): muestra el render REAL del juego (el framebuffer 3D). El render
  lo hace SC2 por dentro (EGL/OSMesa) y nos llega como arrays; nosotros solo los
  blitéamos. En Brais (GPU en MIG) el backend fiable es OSMesa (software): instala
  `libosmesa6`. Ver NOTES §7.2.

Por qué por software: el visor humano de PySC2 (`visualize=True`) crea un contexto
OpenGL en el hilo principal y lo usa desde otro hilo; Mesa/llvmpipe lo prohíbe y
aborta con BAD_ACCESS sobre VNC. Aquí corremos con `visualize=False` y dibujamos
nosotros en una ventana pygame de software, un solo hilo. El render RGB del juego
NO usa nuestro OpenGL: lo hace SC2 y nos pasa píxeles.

Uso (en Brais, env `sc2-rl-infra` activo, VNC en :1):
    DISPLAY=:1 python -m sc2_rl_infra.live_view                          # feature layers
    DISPLAY=:1 python -m sc2_rl_infra.live_view --rgb                    # render RGB real
    DISPLAY=:1 python -m sc2_rl_infra.live_view --rgb --record out.mp4   # graba a mp4
    DISPLAY=:1 python -m sc2_rl_infra.live_view --record out.mp4         # feature layers a mp4

Dependencias extra: `--rgb` necesita backend de render en SC2 (`sudo apt install
libosmesa6`); `--record` necesita `pip install imageio imageio-ffmpeg`.
Cierra la ventana (o Esc/Q) para detener.
"""

import math
import os

import numpy as np
from absl import app, flags
from pysc2.agents import random_agent
from pysc2.env import sc2_env
from pysc2.lib import features

# Sin mixer de audio: evita el ruido de ALSA ("cannot find card '0'") en headless.
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
import pygame  # noqa: E402  (tras fijar SDL_AUDIODRIVER)

FLAGS = flags.FLAGS
flags.DEFINE_string("map", "MoveToBeacon", "Minijuego PySC2 a jugar.")
flags.DEFINE_integer("episodes", 3, "Número de episodios a jugar.")
flags.DEFINE_integer("step_mul", 8, "Game steps por cada agent step (menor = más fluido).")
flags.DEFINE_integer("screen", 84, "Resolución (px) de las feature layers de pantalla.")
flags.DEFINE_integer("minimap", 64, "Resolución (px) de las feature layers de minimapa.")
flags.DEFINE_integer("max_steps", 0, "Máx. agent steps por episodio (0 = hasta que acabe).")
flags.DEFINE_integer("fps", 22, "Límite de FPS del visor (controla la velocidad de reproducción).")
flags.DEFINE_integer("cell", 320, "Lado (px) de cada panel en la ventana.")
flags.DEFINE_list(
    "layers",
    ["screen:player_relative", "screen:unit_type",
     "minimap:player_relative", "minimap:camera"],
    "Capas a mostrar (modo feature); tokens 'screen:nombre' o 'minimap:nombre'.",
)
flags.DEFINE_bool("rgb", False, "Mostrar el render RGB real del juego en vez de feature layers (necesita OSMesa/EGL en SC2).")
flags.DEFINE_integer("rgb_screen", 256, "Resolución (px) del render RGB de pantalla.")
flags.DEFINE_integer("rgb_minimap", 64, "Resolución (px) del render RGB de minimapa.")
flags.DEFINE_string("record", "", "Si se indica, graba la ventana a este .mp4 (necesita imageio-ffmpeg).")
flags.DEFINE_bool("force_osmesa", True, "En modo --rgb, forzar OSMesa quitando EGL de los renderers de PySC2 (EGL no funciona bajo MIG).")

COLS = 2
MARGIN = 8
LABEL_H = 22
HEADER_H = 30
BG = (24, 24, 28)
FG = (230, 230, 230)


def resolve_layers(specs):
    """Convierte ['screen:player_relative', ...] en [(which, Feature), ...]."""
    sets = {"screen": features.SCREEN_FEATURES, "minimap": features.MINIMAP_FEATURES}
    out = []
    for spec in specs:
        spec = spec.strip()
        if ":" not in spec:
            raise ValueError(f"Capa mal formada: {spec!r} (usa 'screen:nombre' o 'minimap:nombre').")
        which, name = (part.strip() for part in spec.split(":", 1))
        which = which.lower()
        if which not in sets:
            raise ValueError(f"Origen desconocido {which!r} en {spec!r}; usa 'screen' o 'minimap'.")
        feat_set = sets[which]
        if not hasattr(feat_set, name):
            avail = ", ".join(f.name for f in feat_set)
            raise ValueError(f"Capa {name!r} no existe en '{which}'. Disponibles: {avail}")
        out.append((which, getattr(feat_set, name)))
    return out


def colorize(plane, feature):
    """(H, W) de enteros -> (H, W, 3) uint8 con la paleta oficial de PySC2.

    Si la paleta falla, cae a escala de grises para no tumbar el visor.
    """
    try:
        palette = np.asarray(feature.palette)
        if palette.ndim != 2 or palette.shape[1] != 3:
            raise ValueError("la paleta no tiene forma (N, 3)")
        idx = np.clip(plane, 0, palette.shape[0] - 1).astype(np.int64)
        return palette[idx].astype(np.uint8)
    except Exception:
        scale = max(int(getattr(feature, "scale", 256)) - 1, 1)
        gray = (np.clip(plane, 0, scale) * (255.0 / scale)).astype(np.uint8)
        return np.dstack([gray, gray, gray])


def plane_surface(rgb, cell):
    """(H, W, 3) uint8 -> superficie pygame escalada a (cell, cell)."""
    # pygame.surfarray espera (ancho, alto, 3); nuestros arrays son (alto, ancho, 3).
    surf = pygame.surfarray.make_surface(np.ascontiguousarray(rgb.transpose(1, 0, 2)))
    return pygame.transform.scale(surf, (cell, cell))


def window_size(n_panels, cell):
    rows = max(1, math.ceil(n_panels / COLS))
    win_w = COLS * cell + (COLS + 1) * MARGIN
    win_h = HEADER_H + rows * (cell + LABEL_H + MARGIN) + MARGIN
    return win_w, win_h


def keep_running():
    """Procesa eventos; devuelve False si el usuario cierra la ventana."""
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return False
        if event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q):
            return False
    return True


def compose_panels(observation, panels):
    """Devuelve [(título, imagen (H, W, 3) uint8), ...] según el modo activo."""
    if FLAGS.rgb:
        return [
            ("rgb_screen", np.asarray(observation["rgb_screen"], dtype=np.uint8)),
            ("rgb_minimap", np.asarray(observation["rgb_minimap"], dtype=np.uint8)),
        ]
    planes = {
        "screen": np.asarray(observation["feature_screen"]),
        "minimap": np.asarray(observation["feature_minimap"]),
    }
    return [(f"{which}:{feat.name}", colorize(planes[which][feat.index], feat))
            for which, feat in panels]


def draw(screen, font, items, header, cell):
    screen.fill(BG)
    screen.blit(font.render(header, True, FG), (MARGIN, (HEADER_H - LABEL_H) // 2 + 2))
    for i, (title, image) in enumerate(items):
        col, row = i % COLS, i // COLS
        x = MARGIN + col * (cell + MARGIN)
        y = HEADER_H + row * (cell + LABEL_H + MARGIN)
        screen.blit(font.render(title, True, FG), (x, y))
        screen.blit(plane_surface(image, cell), (x, y + LABEL_H))
    pygame.display.flip()


def force_osmesa_renderer():
    """Quita EGL de la lista de renderers de PySC2 para que SC2 use OSMesa (software).

    EGL falla bajo MIG ("Failed to create a valid EGL display! Devices tried: 0"),
    así que dejamos solo las entradas -osmesapath de `known_gl_libs`.
    """
    from pysc2 import run_configs
    rc_cls = type(run_configs.get())
    libs = getattr(rc_cls, "known_gl_libs", None)
    if libs:
        rc_cls.known_gl_libs = [entry for entry in libs if entry[0] != "-eglpath"]
        print(f"[live_view] renderers tras forzar OSMesa: {rc_cls.known_gl_libs}")


def make_interface_format():
    if FLAGS.rgb:
        # Solo RGB: con un único espacio, PySC2 infiere el action_space (no hace
        # falta pasarlo). El agente actúa en coords RGB.
        return features.AgentInterfaceFormat(
            rgb_dimensions=features.Dimensions(screen=FLAGS.rgb_screen, minimap=FLAGS.rgb_minimap),
        )
    return features.AgentInterfaceFormat(
        feature_dimensions=features.Dimensions(screen=FLAGS.screen, minimap=FLAGS.minimap),
        use_feature_units=True,
    )


def main(unused_argv):
    panels = [] if FLAGS.rgb else resolve_layers(FLAGS.layers)
    n_items = 2 if FLAGS.rgb else len(panels)

    # En headless, importar pysc2 puede dejar SDL con un driver de vídeo invisible
    # ('dummy'). Forzamos x11 y reiniciamos el subsistema para pintar en el VNC.
    os.environ["SDL_VIDEODRIVER"] = "x11"
    pygame.display.quit()
    pygame.display.init()
    pygame.font.init()
    print(f"[live_view] SDL video driver = {pygame.display.get_driver()}")
    screen = pygame.display.set_mode(window_size(n_items, FLAGS.cell))
    mode = "RGB" if FLAGS.rgb else "feature layers"
    pygame.display.set_caption(f"sc2-rl-infra · visor {mode} (software)")
    font = pygame.font.Font(None, LABEL_H)
    clock = pygame.time.Clock()

    screen.fill(BG)
    screen.blit(font.render("Lanzando SC2...", True, FG), (MARGIN, MARGIN))
    pygame.display.flip()

    writer = None
    if FLAGS.record:
        import imageio
        writer = imageio.get_writer(FLAGS.record, fps=FLAGS.fps, macro_block_size=1)

    if FLAGS.rgb and FLAGS.force_osmesa:
        force_osmesa_renderer()

    agent = random_agent.RandomAgent()
    try:
        with sc2_env.SC2Env(
            map_name=FLAGS.map,
            players=[sc2_env.Agent(sc2_env.Race.random)],
            agent_interface_format=make_interface_format(),
            step_mul=FLAGS.step_mul,
            game_steps_per_episode=0,
            visualize=False,  # NO usamos el renderer GL de PySC2 (peta sobre VNC).
        ) as env:
            agent.setup(env.observation_spec()[0], env.action_spec()[0])

            running = True
            for episode in range(1, FLAGS.episodes + 1):
                if not running:
                    break
                timestep = env.reset()[0]
                agent.reset()
                total_reward, step = 0.0, 0
                while True:
                    total_reward += float(timestep.reward)
                    header = (f"{FLAGS.map} | ep {episode}/{FLAGS.episodes} | "
                              f"step {step} | reward {total_reward:.0f}{' | RGB' if FLAGS.rgb else ''}")
                    draw(screen, font, compose_panels(timestep.observation, panels), header, FLAGS.cell)
                    if writer is not None:
                        frame = pygame.surfarray.array3d(screen).transpose(1, 0, 2)
                        writer.append_data(np.ascontiguousarray(frame))
                    if not keep_running():
                        running = False
                        break
                    if timestep.last() or (FLAGS.max_steps and step >= FLAGS.max_steps):
                        break
                    timestep = env.step([agent.step(timestep)])[0]
                    step += 1
                    clock.tick(FLAGS.fps)
    finally:
        if writer is not None:
            writer.close()
            print(f"[live_view] vídeo escrito en {FLAGS.record}")
        pygame.quit()


if __name__ == "__main__":
    app.run(main)
