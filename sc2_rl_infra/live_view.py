"""Visor en vivo de feature layers por software (sin OpenGL), pensado para VNC.

Por qué existe: el visor humano de PySC2 (`visualize=True`) crea un contexto
OpenGL en el hilo principal y lo usa desde un hilo de render. El stack de Mesa
por software (llvmpipe) aplica de forma estricta la regla "un contexto GL solo
puede estar activo en un hilo a la vez" y aborta con BAD_ACCESS sobre TigerVNC
(da igual GLX o EGL). En vez de pelear con OpenGL, aquí corremos el agente con
`visualize=False` y dibujamos nosotros las feature layers en una ventana pygame
*de software* (sin el flag OPENGL) y en un único hilo. Eso esquiva el problema
por completo y no necesita GPU gráfica, así que se ve en directo en el VNC.

Uso (en Brais, env `sc2-rl-infra` activo, con el servidor VNC en :1):
    DISPLAY=:1 python -m sc2_rl_infra.live_view
    DISPLAY=:1 python -m sc2_rl_infra.live_view --map CollectMineralShards --episodes 3
    DISPLAY=:1 python -m sc2_rl_infra.live_view --step_mul 4 --fps 30 --cell 360

No necesita LIBGL_ALWAYS_SOFTWARE ni VirtualGL: no se usa OpenGL en absoluto.
Cierra la ventana (o pulsa Esc/Q) para detenerlo.
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
    "Capas a mostrar; tokens 'screen:nombre' o 'minimap:nombre' separados por comas.",
)

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

    Si la paleta falla por cualquier motivo, cae a escala de grises para no
    tumbar el visor por una capa concreta.
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


def draw(screen, font, panels, observation, header, cell):
    screen.fill(BG)
    screen.blit(font.render(header, True, FG), (MARGIN, (HEADER_H - LABEL_H) // 2 + 2))
    planes = {
        "screen": np.asarray(observation["feature_screen"]),
        "minimap": np.asarray(observation["feature_minimap"]),
    }
    for i, (which, feat) in enumerate(panels):
        col, row = i % COLS, i // COLS
        x = MARGIN + col * (cell + MARGIN)
        y = HEADER_H + row * (cell + LABEL_H + MARGIN)
        surf = plane_surface(colorize(planes[which][feat.index], feat), cell)
        screen.blit(font.render(f"{which}:{feat.name}", True, FG), (x, y))
        screen.blit(surf, (x, y + LABEL_H))
    pygame.display.flip()


def main(unused_argv):
    panels = resolve_layers(FLAGS.layers)

    pygame.display.init()
    pygame.font.init()
    screen = pygame.display.set_mode(window_size(len(panels), FLAGS.cell))  # SIN OPENGL
    pygame.display.set_caption("sc2-rl-infra · visor de feature layers (software)")
    font = pygame.font.Font(None, LABEL_H)
    clock = pygame.time.Clock()

    # Feedback inmediato mientras arranca SC2 (~varios segundos).
    screen.fill(BG)
    screen.blit(font.render("Lanzando SC2...", True, FG), (MARGIN, MARGIN))
    pygame.display.flip()

    agent = random_agent.RandomAgent()
    with sc2_env.SC2Env(
        map_name=FLAGS.map,
        players=[sc2_env.Agent(sc2_env.Race.random)],
        agent_interface_format=features.AgentInterfaceFormat(
            feature_dimensions=features.Dimensions(screen=FLAGS.screen, minimap=FLAGS.minimap),
            use_feature_units=True,
        ),
        step_mul=FLAGS.step_mul,
        game_steps_per_episode=0,
        visualize=False,  # NO usamos el renderer GL de PySC2 (es el que peta sobre VNC).
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
                header = (f"{FLAGS.map}  |  episodio {episode}/{FLAGS.episodes}  |  "
                          f"step {step}  |  reward {total_reward:.0f}")
                draw(screen, font, panels, timestep.observation, header, FLAGS.cell)
                if not keep_running():
                    running = False
                    break
                if timestep.last() or (FLAGS.max_steps and step >= FLAGS.max_steps):
                    break
                timestep = env.step([agent.step(timestep)])[0]
                step += 1
                clock.tick(FLAGS.fps)

    pygame.quit()


if __name__ == "__main__":
    app.run(main)
