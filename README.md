# sc2-rl-infra

Infraestructura para RL sobre StarCraft II (PySC2 + SC2 headless en Linux), de cara a reproducir AlphaStar a escala reducida. Contexto, decisiones y resultados en `00_OVERVIEW.md`, `01_PHASE0_infra.md`, `NOTES.md` y `RESULTS.md`.

Este README recoge **todos los comandos** del proyecto. Salvo que se indique lo contrario, asumen:
- entorno conda `sc2-rl-infra` **activo**, y
- directorio de trabajo `~/sc2-rl-infra` en Brais (la máquina Linux donde corre SC2).

---

## Entorno

```bash
conda env create -f environment.yml     # recrea el env (freeze en environment.yml)
conda activate sc2-rl-infra
pip install -e . --no-deps              # instala el paquete sc2_rl_infra (editable)
```

### Sideload de SC2 (build 4.10 headless)

El egress directo a Akamai está bloqueado desde Brais (`NOTES.md §4`), así que el binario se descarga fuera y se transfiere (`NOTES.md §6`):

```powershell
# 1) En una máquina CON salida a Akamai (Windows / PowerShell):
curl.exe -L -C - --connect-timeout 20 --retry 3 -o "$HOME\Downloads\SC2.4.10.zip" "https://blzdistsc2-a.akamaihd.net/Linux/SC2.4.10.zip"
```
```bash
# 2) En Brais: transferir, verificar integridad y descomprimir (zip cifrado):
scp SC2.4.10.zip master@<brais>:~/SC2_A/
stat -c %s ~/SC2_A/SC2.4.10.zip                       # debe dar 4115224017
unzip -P iagreetotheeula ~/SC2_A/SC2.4.10.zip -d ~/   # crea ~/StarCraftII/
```

---

## Sincronizar Brais con el repo

Brais solo **consume** el código (se desarrolla y se sube desde otra parte). Para traer lo último sin divergencias por merges locales, sincroniza a `main` a pelo:

```bash
cd ~/sc2-rl-infra
git fetch origin
git reset --hard origin/main
```

---

## Smoke test (sanity de PySC2 ↔ SC2)

```bash
python -m pysc2.bin.agent --map MoveToBeacon --agent pysc2.agents.random_agent.RandomAgent --norender --max_episodes 1
```

---

## Agente demo

```bash
python -m sc2_rl_infra.demo_random_agent
python -m sc2_rl_infra.demo_random_agent --map CollectMineralShards --episodes 2
```
Flags: `--map --episodes --step_mul --screen --minimap --render`.
No uses `--render` en Brais: activa el visor GL de PySC2, que **peta sobre VNC** (`NOTES.md §7`). Para ver al agente, usa `live_view` (abajo).

---

## Benchmarks de throughput (Fase 0)

```bash
# 1 instancia:
python -m sc2_rl_infra.benchmark_throughput
python -m sc2_rl_infra.benchmark_throughput --map MoveToBeacon --steps 4000
python -m sc2_rl_infra.benchmark_throughput --use_feature_units        # con raw interface
# flags: --map --steps --step_mul --screen --minimap --use_feature_units

# N instancias en paralelo:
python -m sc2_rl_infra.benchmark_parallel
python -m sc2_rl_infra.benchmark_parallel --n_envs 8 --duration 30
python -m sc2_rl_infra.benchmark_parallel --n_envs 1,2,4,8,12
# flags: --n_envs --duration --map --step_mul --screen --minimap --use_feature_units --startup_timeout
```

---

## Parsear replays

```bash
python -m sc2_rl_infra.parse_replay --replay /home/master/StarCraftII/Replays/<...>.SC2Replay
```
Flags: `--replay` (obligatorio) `--observed_player --step_mul --max_steps --screen --minimap`.
Regla de versión: un replay solo se parsea con la **misma build** de SC2 que lo grabó (4.10 hoy).

---

## Visualización en vivo (visor por software sobre VNC)

Brais es headless. El agente se ve en directo con un visor **pygame por software** (feature layers, sin OpenGL), mirado desde Windows con MobaXterm. El porqué (no es el visor GL oficial) y las decisiones están en `NOTES.md §7`.

```bash
# Instalar VNC + gestor de ventanas mínimo (una sola vez; requiere sudo):
sudo apt install -y tigervnc-standalone-server fluxbox
vncpasswd                       # fija la contraseña de VNC (una sola vez)

# Gestionar el servidor VNC (tools/vnc.sh crea ~/.vnc/xstartup con fluxbox si falta):
tools/vnc.sh start              # vncserver :1 -geometry 1600x900 -localhost yes
tools/vnc.sh status
tools/vnc.sh stop
tools/vnc.sh restart
# variables opcionales: VNC_DISPLAY (def. :1), VNC_GEOMETRY (def. 1600x900)

# Lanzar el visor (DISPLAY=:1 obligatorio):
DISPLAY=:1 python -m sc2_rl_infra.live_view
DISPLAY=:1 python -m sc2_rl_infra.live_view --map CollectMineralShards --step_mul 4 --fps 30
```
Flags: `--map --episodes --step_mul --screen --minimap --max_steps --fps --cell --layers --save_replay --replay_dir`.

**MobaXterm (cliente VNC en Windows):** sesión VNC → host `localhost:5901`; en *Network settings* activar *Connect through SSH gateway (jump host)* apuntando a Brais. El servidor solo escucha en loopback, por eso se entra por el túnel SSH (no por la IP directa).

---

## Exportar replays

```bash
DISPLAY=:1 python -m sc2_rl_infra.live_view --save_replay --episodes 1
# escribe el .SC2Replay en ~/StarCraftII/Replays/sc2-rl-infra/
```
El `.SC2Replay` es el artefacto portátil. Para verlo con **gráficos reales**: copiarlo a Windows (`scp` o el panel SFTP de MobaXterm) y abrirlo en el cliente de SC2 (en la misma versión que lo grabó). Ver `NOTES.md §7.3`.
