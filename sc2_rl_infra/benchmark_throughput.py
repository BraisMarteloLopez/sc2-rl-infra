"""Benchmark de throughput del entorno SC2/PySC2 (Fase 0, objetivo central).

Incremento 1: una sola instancia. Mide agent-steps/segundo en estado estable
enviando una acción trivial (no_op), para aislar el coste del ENTORNO del coste
del agente. Razón: en fases siguientes el agente será una red neuronal en GPU,
así que medir el sampling de un agente random en CPU no sería representativo —
lo que importa en Fase 0 es cuántos steps/s da el entorno en sí.

Reporta también el tiempo de arranque de SC2 (relevante para fases con muchos resets).
La paralelización a N instancias (N = {1,2,4,8,12}) llega en el incremento 2.

Uso (en Brais, env sc2-rl-infra activo, desde ~/sc2-rl-infra):
    python -m sc2_rl_infra.benchmark_throughput
    python -m sc2_rl_infra.benchmark_throughput --map MoveToBeacon --steps 4000
    python -m sc2_rl_infra.benchmark_throughput --use_feature_units   # con raw interface
"""

import time

from absl import app, flags
from pysc2.env import sc2_env
from pysc2.lib import actions, features

FLAGS = flags.FLAGS
flags.DEFINE_string("map", "MoveToBeacon", "Minijuego/mapa a medir.")
flags.DEFINE_integer("steps", 2000, "Número de agent-steps a cronometrar (excluye resets).")
flags.DEFINE_integer("step_mul", 8, "Game steps por agent step.")
flags.DEFINE_integer("screen", 84, "Resolución (px) de las feature layers de pantalla.")
flags.DEFINE_integer("minimap", 64, "Resolución (px) de las feature layers de minimapa.")
flags.DEFINE_bool("use_feature_units", False,
                  "Incluir feature_units (activa el raw interface; añade sobrecarga por step).")


def benchmark_one_env(map_name, total_steps, step_mul, screen, minimap, use_feature_units):
    """Mide el throughput de stepping de una instancia de SC2. Devuelve un dict de métricas."""
    aif = features.AgentInterfaceFormat(
        feature_dimensions=features.Dimensions(screen=screen, minimap=minimap),
        use_feature_units=use_feature_units,
    )
    no_op = actions.FUNCTIONS.no_op()

    t0 = time.perf_counter()
    with sc2_env.SC2Env(
        map_name=map_name,
        players=[sc2_env.Agent(sc2_env.Race.random)],
        agent_interface_format=aif,
        step_mul=step_mul,
        game_steps_per_episode=0,
        visualize=False,
    ) as env:
        launch_s = time.perf_counter() - t0

        tr = time.perf_counter()
        env.reset()
        first_reset_s = time.perf_counter() - tr

        steps = 0
        step_s = 0.0
        resets = 0
        while steps < total_steps:
            t = time.perf_counter()
            timesteps = env.step([no_op])
            step_s += time.perf_counter() - t
            steps += 1
            if timesteps[0].last():
                env.reset()  # turnover de episodio; no cuenta como step cronometrado
                resets += 1

    fps = steps / step_s if step_s > 0 else float("nan")
    return {
        "launch_s": launch_s,
        "first_reset_s": first_reset_s,
        "steps": steps,
        "step_s": step_s,
        "resets": resets,
        "fps": fps,
        "game_loops_per_s": fps * step_mul,
    }


def main(unused_argv):
    r = benchmark_one_env(
        FLAGS.map, FLAGS.steps, FLAGS.step_mul, FLAGS.screen, FLAGS.minimap, FLAGS.use_feature_units
    )
    raw = "ON" if FLAGS.use_feature_units else "OFF"
    print("\n=== Benchmark throughput — 1 instancia, headless ===")
    print(f"  Mapa:                {FLAGS.map}")
    print(f"  step_mul:            {FLAGS.step_mul}")
    print(f"  feature_units:       {FLAGS.use_feature_units}  (raw interface {raw})")
    print(f"  feature dims:        screen={FLAGS.screen} minimap={FLAGS.minimap}")
    print(f"  Lanzamiento SC2:     {r['launch_s']:.2f} s")
    print(f"  Primer reset:        {r['first_reset_s']:.2f} s")
    print(f"  Steps cronometrados: {r['steps']}  (resets durante la medición: {r['resets']})")
    print(f"  Tiempo de step:      {r['step_s']:.2f} s")
    print(f"  THROUGHPUT:          {r['fps']:.1f} agent-steps/s  |  {r['game_loops_per_s']:.0f} game-loops/s")
    print()


if __name__ == "__main__":
    app.run(main)
