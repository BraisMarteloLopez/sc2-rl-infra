"""Demo: un agente aleatorio juega un minijuego de PySC2.

Es el punto de entrada propio de Fase 0 (criterio de éxito de 01_PHASE0_infra.md §4),
y replica el smoke test ya validado a mano con `pysc2.bin.agent`.

Uso (en Brais, env `sc2-rl-infra` activo):
    python -m sc2_rl_infra.demo_random_agent
    python -m sc2_rl_infra.demo_random_agent --map CollectMineralShards --episodes 2

Brais es headless (sin display X), por eso `--render` está en False por defecto.
"""

from absl import app, flags
from pysc2.agents import random_agent
from pysc2.env import run_loop, sc2_env
from pysc2.lib import features

FLAGS = flags.FLAGS
flags.DEFINE_string("map", "MoveToBeacon", "Minijuego PySC2 a jugar.")
flags.DEFINE_integer("episodes", 1, "Número de episodios a jugar.")
flags.DEFINE_integer("step_mul", 8, "Game steps por cada agent step.")
flags.DEFINE_integer("screen", 84, "Resolución (px) de las feature layers de pantalla.")
flags.DEFINE_integer("minimap", 64, "Resolución (px) de las feature layers de minimapa.")
flags.DEFINE_bool("render", False, "Visualizar la partida (necesita display X; en Brais headless: False).")


def main(unused_argv):
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
        visualize=FLAGS.render,
    ) as env:
        run_loop.run_loop([agent], env, max_episodes=FLAGS.episodes)


if __name__ == "__main__":
    app.run(main)
