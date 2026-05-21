"""Benchmark de throughput — incremento 2: N instancias de SC2 en paralelo.

Mide el throughput AGREGADO con N instancias corriendo a la vez (un proceso por
instancia), barriendo N hasta el techo de cores del contenedor. Es el dato que
cierra Fase 0: cuántas instancias paralelas sostiene la máquina y con qué eficiencia
de escalado.

Diseño:
- Un proceso hijo por instancia (multiprocessing 'spawn'; el padre no abre ningún
  SC2Env, así que no hay estado que herede mal el fork). Cada worker crea su propio
  SC2Env, arranca SC2, resetea, y cuenta cuántos steps (no_op) hace en `--duration` s.
- Throughput agregado = suma de steps de todos los workers / duration.
- Tolerante a fallos: si un worker peta o no arranca, se reporta como fallo y NO
  cuelga al resto (timeout en la recogida de resultados + terminación de procesos
  colgados). Esto importa: queremos descubrir el techo de N sin que la sesión se cuelgue.

Uso (en Brais, env sc2-rl-infra activo, desde ~/sc2-rl-infra):
    python -m sc2_rl_infra.benchmark_parallel
    python -m sc2_rl_infra.benchmark_parallel --n_envs 8 --duration 30
    python -m sc2_rl_infra.benchmark_parallel --n_envs 1,2,4,8,12
"""

import multiprocessing as mp
import os
import time

from absl import app, flags

FLAGS = flags.FLAGS
flags.DEFINE_list("n_envs", ["1", "2", "4", "8", "12"], "Lista de N (instancias) a barrer.")
flags.DEFINE_integer("duration", 20, "Segundos de stepping cronometrado por instancia.")
flags.DEFINE_string("map", "MoveToBeacon", "Minijuego/mapa.")
flags.DEFINE_integer("step_mul", 8, "Game steps por agent step.")
flags.DEFINE_integer("screen", 84, "Resolución (px) feature layers de pantalla.")
flags.DEFINE_integer("minimap", 64, "Resolución (px) feature layers de minimapa.")
flags.DEFINE_bool("use_feature_units", False, "Activar feature_units (raw interface).")
flags.DEFINE_integer("startup_timeout", 180, "Timeout (s) por barrido para que arranquen las instancias.")


def _read_cpu_times():
    """Tiempos agregados de CPU desde /proc/stat (Linux). Devuelve (total, idle)."""
    with open("/proc/stat") as f:
        vals = [int(x) for x in f.readline().split()[1:]]
    idle = vals[3] + (vals[4] if len(vals) > 4 else 0)  # idle + iowait
    return sum(vals), idle


def _worker(idx, duration, map_name, step_mul, screen, minimap, use_feature_units, q):
    """Arranca una instancia de SC2 y cuenta steps (no_op) durante `duration` s."""
    try:
        from pysc2.env import sc2_env
        from pysc2.lib import actions, features

        # Los procesos hijos (spawn) no parsean flags, pero pysc2 lee FLAGS.sc2_run_config
        # internamente al crear el SC2Env -> UnparsedFlagAccessError. Marcamos los flags como
        # parseados (toman sus defaults, que es lo que necesita pysc2). El worker no usa FLAGS
        # para sus propios parámetros: van por argumentos.
        flags.FLAGS.mark_as_parsed()

        no_op = actions.FUNCTIONS.no_op()
        aif = features.AgentInterfaceFormat(
            feature_dimensions=features.Dimensions(screen=screen, minimap=minimap),
            use_feature_units=use_feature_units,
        )
        t0 = time.perf_counter()
        with sc2_env.SC2Env(
            map_name=map_name,
            players=[sc2_env.Agent(sc2_env.Race.random)],
            agent_interface_format=aif,
            step_mul=step_mul,
            game_steps_per_episode=0,
            visualize=False,
        ) as env:
            startup = time.perf_counter() - t0
            env.reset()
            t_end = time.perf_counter() + duration
            steps = 0
            while time.perf_counter() < t_end:
                timesteps = env.step([no_op])
                steps += 1
                if timesteps[0].last():
                    env.reset()
        q.put((idx, steps, startup))
    except Exception as e:  # cualquier fallo se reporta; no debe colgar al resto
        q.put((idx, -1, repr(e)))


def run_for_n(n, ctx, duration, map_name, step_mul, screen, minimap, use_feature_units, startup_timeout):
    """Lanza n workers en paralelo y agrega sus resultados. Robusto a cuelgues."""
    q = ctx.Queue()
    procs = [
        ctx.Process(target=_worker,
                    args=(i, duration, map_name, step_mul, screen, minimap, use_feature_units, q))
        for i in range(n)
    ]
    cpu_total0, cpu_idle0 = _read_cpu_times()
    for p in procs:
        p.start()

    deadline = time.perf_counter() + startup_timeout + duration + 60
    results = []
    while len(results) < n and time.perf_counter() < deadline:
        try:
            results.append(q.get(timeout=max(1.0, deadline - time.perf_counter())))
        except Exception:  # queue.Empty: nadie reportó en el plazo
            break

    cpu_total1, cpu_idle1 = _read_cpu_times()
    for p in procs:
        p.join(timeout=5)
        if p.is_alive():
            p.terminate()

    ok = [(i, s, su) for (i, s, su) in results if s >= 0]
    errors = [(i, su) for (i, s, su) in results if s < 0]
    total_steps = sum(s for (_, s, _) in ok)
    agg_fps = total_steps / duration if (duration > 0 and ok) else float("nan")
    per_inst = agg_fps / len(ok) if ok else float("nan")
    mean_startup = sum(su for (_, _, su) in ok) / len(ok) if ok else float("nan")
    dtot = cpu_total1 - cpu_total0
    didle = cpu_idle1 - cpu_idle0
    cpu_pct = 100.0 * (1 - didle / dtot) if dtot > 0 else float("nan")
    return {
        "n": n, "ok": len(ok), "failed": n - len(ok),
        "agg_fps": agg_fps, "per_inst": per_inst, "mean_startup": mean_startup,
        "cpu_pct": cpu_pct, "errors": errors,
    }


def main(unused_argv):
    n_list = [int(x) for x in FLAGS.n_envs]
    cores = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    ctx = mp.get_context("spawn")
    raw = "ON" if FLAGS.use_feature_units else "OFF"

    print(f"\n=== Benchmark throughput — N instancias en paralelo (headless, {FLAGS.map}) ===")
    print(f"Cores disponibles (cpuset): {cores} | duration/instancia: {FLAGS.duration}s | "
          f"step_mul: {FLAGS.step_mul} | raw: {raw}\n")
    header = (f"{'N':>3} | {'agg steps/s':>11} | {'por inst':>9} | {'escalado':>8} | "
              f"{'efic.':>6} | {'CPU%':>5} | {'arranque':>8} | fallos")
    print(header)
    print("-" * len(header))

    base_per_inst = None
    all_errors = []
    for n in n_list:
        r = run_for_n(n, ctx, FLAGS.duration, FLAGS.map, FLAGS.step_mul,
                      FLAGS.screen, FLAGS.minimap, FLAGS.use_feature_units, FLAGS.startup_timeout)
        if base_per_inst is None and r["ok"] > 0:
            base_per_inst = r["agg_fps"] / n  # baseline: throughput/instancia del primer N
        if base_per_inst and r["ok"] > 0:
            scaling = r["agg_fps"] / base_per_inst  # nº efectivo de instancias
            eff = 100.0 * scaling / n
        else:
            scaling, eff = float("nan"), float("nan")
        print(f"{n:>3} | {r['agg_fps']:>11.1f} | {r['per_inst']:>9.1f} | {scaling:>7.2f}x | "
              f"{eff:>5.0f}% | {r['cpu_pct']:>4.0f}% | {r['mean_startup']:>7.1f}s | {r['failed']}")
        for (i, err) in r["errors"]:
            all_errors.append((n, i, err))

    if all_errors:
        print("\nFallos (posible techo de N o límite de recursos):")
        for (n, i, err) in all_errors:
            print(f"  N={n} inst={i}: {err}")
    print()


if __name__ == "__main__":
    app.run(main)
