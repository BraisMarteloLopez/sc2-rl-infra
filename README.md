# sc2-rl-infra

Código propio para **replicar AlphaStar a escala reducida** sobre StarCraft II: implementar su arquitectura neuronal, entrenarla por *behaviour cloning* sobre replays humanos, y reproducir a pequeña escala los experimentos de AlphaStar Unplugged (offline RL) y de mini-juegos (online RL). Una sola H100, PyTorch, SC2 headless en Linux. El objetivo es **entendimiento + código funcional**, no competir.

**Monorepo:** todas las fases viven en este repo, organizadas por subpaquetes de `sc2_rl_infra/`. Plan estratégico completo en [`00_OVERVIEW.md`](00_OVERVIEW.md).

---

## Estado y fases

| Fase | Qué | Estado |
|---|---|---|
| **0 — Infra** | Entorno SC2/PySC2, throughput, pipeline de datos | **Cerrada** |
| **1 — Behaviour cloning** | Arquitectura AlphaStar + BC sobre replays humanos | Siguiente |
| **2 — Offline RL** | AlphaStar Unplugged a escala reducida | Planificada |
| **3 — Online RL** | RL en mini-juegos PySC2 vs baselines SC2LE | Planificada |
| **4 — World models** | Dreamer-family | Opcional |

**Fase 0 cerrada:** SC2 4.10 headless funcionando, throughput **~4 700 agent-steps/s a 12 instancias** (escala ~lineal; default recomendado **N=8**, ~2 766 steps/s), y `parse_replay` extrae observaciones + acciones de un replay. Detalle en [`RESULTS.md`](RESULTS.md). **Siguiente:** adquirir el dataset de replays humanos (primera tarea de Fase 1).

---

## Estructura del repo

```
sc2-rl-infra/
├── 00_OVERVIEW.md        Plan estratégico (objetivo, fases, convenciones, riesgos)
├── 01_PHASE0_infra.md    Definición y criterios de Fase 0
├── NOTES.md              Bitácora viva (decisiones, restricciones, visualización §7)
├── RESULTS.md            Resultados de Fase 0 (throughput, recomendaciones)
├── environment.yml       Entorno conda congelado
├── pyproject.toml        Paquete sc2_rl_infra
├── tools/
│   └── vnc.sh            Servidor VNC para la visualización en vivo
└── sc2_rl_infra/
    ├── demo_random_agent.py    Agente random en un mini-juego (smoke / demo)
    ├── benchmark_throughput.py Throughput de 1 instancia
    ├── benchmark_parallel.py   Throughput agregado de N instancias
    ├── parse_replay.py         Parser de replays (.SC2Replay → obs + acciones)
    └── live_view.py            Visor en vivo por software (feature layers sobre VNC)
```

Las carpetas de fases futuras (`env/`, `models/`, `imitation/`, `offline/`, `online/`, `worldmodels/`) se crean cuando llega su código, no antes (`00_OVERVIEW §3.1`).

---

## Convenciones del proyecto

- **Raza / matchup:** Protoss, PvP (`00_OVERVIEW §2.3`).
- **Perfil de interfaz:** `human` (cámara + feature layers, fidelidad AlphaStar) para BC/RL; `full` (raw, todo el mapa) como baseline de throughput (`00_OVERVIEW §2.4`).
- **Regla de versión:** SC2 corre siempre en Linux/Brais; un replay solo se parsea con la build que lo grabó (4.10 hoy).

---

## Entorno

Linux/Brais: NVIDIA H100 (MIG 2g.24gb, ~24 GB), 12 cores (cpuset LXC), Ubuntu 24.04, SC2 headless x86_64. (La DGX Spark ARM queda fuera: SC2 no tiene build ARM.) Detalle en `NOTES §1`.

Todos los comandos asumen el env conda `sc2-rl-infra` **activo** y `cwd = ~/sc2-rl-infra`.

```bash
conda env create -f environment.yml     # recrea el env (freeze)
conda activate sc2-rl-infra
pip install -e . --no-deps              # instala sc2_rl_infra (editable)
```

### Sideload de SC2 (build 4.10 headless)

El egress a Akamai está bloqueado desde Brais (`NOTES §4`), así que el binario se descarga fuera y se transfiere (`NOTES §6`):

```powershell
# En una máquina CON salida a Akamai (Windows / PowerShell):
curl.exe -L -C - --connect-timeout 20 --retry 3 -o "$HOME\Downloads\SC2.4.10.zip" "https://blzdistsc2-a.akamaihd.net/Linux/SC2.4.10.zip"
```
```bash
# En Brais: transferir, verificar y descomprimir (zip cifrado):
scp SC2.4.10.zip master@<brais>:~/SC2_A/
stat -c %s ~/SC2_A/SC2.4.10.zip                       # debe dar 4115224017
unzip -P iagreetotheeula ~/SC2_A/SC2.4.10.zip -d ~/   # crea ~/StarCraftII/
```

### Sincronizar Brais

Brais solo **consume** el código (se desarrolla y se sube desde otra parte):

```bash
cd ~/sc2-rl-infra && git fetch origin && git reset --hard origin/main
```

---

## Comandos

### Smoke test (PySC2 ↔ SC2)
```bash
python -m pysc2.bin.agent --map MoveToBeacon --agent pysc2.agents.random_agent.RandomAgent --norender --max_episodes 1
```

### Agente demo
```bash
python -m sc2_rl_infra.demo_random_agent
python -m sc2_rl_infra.demo_random_agent --map CollectMineralShards --episodes 2
```
Flags: `--map --episodes --step_mul --screen --minimap --render`.
No uses `--render` en Brais: activa el visor GL de PySC2, que **peta sobre VNC** (`NOTES §7`). Para ver al agente usa `live_view`.

### Benchmarks de throughput (Fase 0)
```bash
# 1 instancia:
python -m sc2_rl_infra.benchmark_throughput
python -m sc2_rl_infra.benchmark_throughput --steps 4000 --use_feature_units
# flags: --map --steps --step_mul --screen --minimap --use_feature_units

# N instancias en paralelo:
python -m sc2_rl_infra.benchmark_parallel --n_envs 1,2,4,8,12 --duration 30
# flags: --n_envs --duration --map --step_mul --screen --minimap --use_feature_units --startup_timeout
```

### Parsear replays
```bash
python -m sc2_rl_infra.parse_replay --replay ~/StarCraftII/Replays/<...>.SC2Replay
# flags: --replay (obligatorio) --observed_player --step_mul --max_steps --screen --minimap
```

### Visor en vivo (sobre VNC)
Brais es headless: el agente se ve con un visor pygame **por software** (feature layers, sin OpenGL) sobre VNC, mirado desde Windows con MobaXterm. Porqué y decisiones en `NOTES §7`.
```bash
# VNC + gestor de ventanas mínimo (una sola vez; requiere sudo):
sudo apt install -y tigervnc-standalone-server fluxbox
vncpasswd

# Servidor VNC (tools/vnc.sh crea ~/.vnc/xstartup con fluxbox si falta):
tools/vnc.sh start            # | stop | status | restart  (vars: VNC_DISPLAY, VNC_GEOMETRY)

# Visor (DISPLAY=:1 obligatorio):
DISPLAY=:1 python -m sc2_rl_infra.live_view                                              # agente aleatorio (baseline)
DISPLAY=:1 python -m sc2_rl_infra.live_view --agent pysc2.agents.scripted_agent.MoveToBeacon  # agente que SÍ resuelve el mapa
DISPLAY=:1 python -m sc2_rl_infra.live_view --map CollectMineralShards --step_mul 4 --fps 30
# flags: --agent --map --episodes --step_mul --screen --minimap --max_steps --fps --cell --layers --save_replay --replay_dir
```
**MobaXterm (cliente VNC en Windows):** sesión VNC → host `localhost:5901`; en *Network settings* activar *Connect through SSH gateway (jump host)* apuntando a Brais (el servidor solo escucha en loopback).

### Exportar replays
```bash
DISPLAY=:1 python -m sc2_rl_infra.live_view --save_replay --episodes 1   # → ~/StarCraftII/Replays/sc2-rl-infra/
```
El `.SC2Replay` es el artefacto portátil. Para verlo con **gráficos reales**: copiarlo a Windows (`scp` o SFTP de MobaXterm) y abrirlo en el cliente de SC2 (misma versión que lo grabó). Ver `NOTES §7.3`.

---

## Documentación

- [`00_OVERVIEW.md`](00_OVERVIEW.md) — plan estratégico (objetivo, fases, convenciones, riesgos, referencias).
- [`01_PHASE0_infra.md`](01_PHASE0_infra.md) — Fase 0: objetivos y criterios de cierre.
- [`NOTES.md`](NOTES.md) — bitácora viva: entornos (§1), decisiones (§2), restricciones (§4), sideload de SC2 (§6), visualización remota (§7).
- [`RESULTS.md`](RESULTS.md) — resultados de Fase 0 (throughput, escalado, recomendación de N).
