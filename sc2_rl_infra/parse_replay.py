"""Parser mínimo de replays: extrae observaciones y acciones de un .SC2Replay.

Valida el camino de datos que usará Fase 1 (behaviour cloning sobre replays humanos):
abrir un replay, recorrerlo y sacar pares (observación, acciones) del jugador observado.

IMPORTANTE — versión: un replay solo se parsea con la MISMA versión de SC2 que lo grabó.
Este script lanza la versión instalada por defecto (4.10), así que sirve para replays 4.10
(p. ej. los que guarda nuestro propio agente). Los packs de Blizzard son 3.16.1: para
parsearlos hay que tener SC2 3.16.1 instalado.

Uso (en Brais, env sc2-rl-infra activo):
    python -m sc2_rl_infra.parse_replay --replay /home/master/StarCraftII/Replays/<...>.SC2Replay
"""

from absl import app, flags
from pysc2 import run_configs
from pysc2.lib import features
from s2clientprotocol import common_pb2
from s2clientprotocol import sc2api_pb2 as sc_pb

FLAGS = flags.FLAGS
flags.DEFINE_string("replay", None, "Ruta absoluta al fichero .SC2Replay.")
flags.DEFINE_integer("observed_player", 1, "Jugador a observar (1 o 2).")
flags.DEFINE_integer("step_mul", 8, "Game steps por observación.")
flags.DEFINE_integer("max_steps", 1000, "Máximo de observaciones a recorrer (0 = hasta el final).")
flags.DEFINE_integer("screen", 84, "Resolución (px) de feature screen.")
flags.DEFINE_integer("minimap", 64, "Resolución (px) de feature minimap.")
flags.mark_flag_as_required("replay")


def main(unused_argv):
    run_config = run_configs.get()
    replay_data = run_config.replay_data(FLAGS.replay)

    interface = sc_pb.InterfaceOptions(raw=True, score=True)
    interface.feature_layer.width = 24
    interface.feature_layer.resolution.x = FLAGS.screen
    interface.feature_layer.resolution.y = FLAGS.screen
    interface.feature_layer.minimap_resolution.x = FLAGS.minimap
    interface.feature_layer.minimap_resolution.y = FLAGS.minimap

    with run_config.start(want_rgb=False) as controller:
        info = controller.replay_info(replay_data)
        print("=== Replay info ===")
        print(f"  Mapa:     {info.map_name}")
        print(f"  Versión:  {info.game_version} (base build {info.base_build})")
        print(f"  Duración: {info.game_duration_loops} game loops (~{info.game_duration_seconds:.0f} s)")
        for p in info.player_info:
            race = common_pb2.Race.Name(p.player_info.race_actual)
            res = sc_pb.Result.Name(p.player_result.result) if p.HasField("player_result") else "?"
            print(f"  Jugador {p.player_info.player_id}: {race} -> {res}")

        controller.start_replay(sc_pb.RequestStartReplay(
            replay_data=replay_data,
            options=interface,
            observed_player_id=FLAGS.observed_player,
        ))

        feat = features.features_from_game_info(controller.game_info())

        steps, actions_seen = 0, 0
        agent_obs = None
        while True:
            controller.step(FLAGS.step_mul)
            obs = controller.observe()
            agent_obs = feat.transform_obs(obs)
            actions_seen += len(obs.actions)
            steps += 1
            if obs.player_result or (FLAGS.max_steps and steps >= FLAGS.max_steps):
                break

    print("\n=== Parseo ===")
    print(f"  Observaciones recorridas: {steps}")
    print(f"  Acciones del jugador observadas: {actions_seen}")
    if agent_obs is not None:
        print(f"  Claves de observación disponibles: {sorted(agent_obs.keys())}")
    print("  OK: replay parseado; observaciones (feature layers) y acciones extraídas.")


if __name__ == "__main__":
    app.run(main)
