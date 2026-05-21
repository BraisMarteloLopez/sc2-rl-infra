# Notas de trabajo — Fase 0

Documento vivo de decisiones tomadas, restricciones detectadas y decisiones aparcadas durante Fase 0.

Última actualización: 2026-05-21.

---

## 1. Entornos disponibles

### 1.1 Linux/Brais (entorno donde se ejecuta Fase 0)

- **GPU:** NVIDIA H100 NVL en MIG slice GI=3 (~22 GB de los 94 GB totales del H100). Driver 580.105.08, CUDA 13.0.
- **CPU:** AMD EPYC 9534 (Zen 4). El host tiene 128 hilos, **el contenedor LXC nos asigna 12 cores fijos**: `6, 9, 20, 23, 30, 31, 80, 94, 96, 97, 100, 124` (cpuset cgroup, `Cpus_allowed_list`). No modificable desde dentro del contenedor. CPU max MHz visible: 2450.
- **OS:** Ubuntu 24.04.3 LTS, kernel 6.8.0, x86_64. Contenedor LXC confirmado (`systemd-detect-virt: lxc`).
- **RAM:** 59 GB, 56 GB libres.
- **Disco:** rootfs montado sobre `pool-zfs/containers/NVIDIA_Brais`. 467 GB total, 395 GB libres.
- **Python:** 3.13.11 en miniconda base (`/home/master/miniconda3/bin/python3`). conda disponible.
- **Sudo:** aparentemente sin contraseña (a confirmar con un comando inocuo antes de depender de ello).
- **Herramientas confirmadas:** git, curl, wget, unzip, gcc, make, tar, python3, pip3, conda.
- **MIG:** slice `2g.24gb` fijo (~24 GB de VRAM, 2 de 7 GPC slices). Sin permisos para reconfigurar MIG desde el contenedor.

### 1.2 DGX Spark (no usada en Fase 0)

- **GPU:** NVIDIA GB10 (Grace+Blackwell), memoria unificada 121 GB.
- **CPU:** Cortex-X925 + Cortex-A725, 20 cores, aarch64.
- **OS:** Ubuntu 24.04.4 LTS, kernel 6.17, aarch64.
- **Disco:** 3.7 TB, 3.5 TB libres.
- **Python:** 3.12.3 sistema.

---

## 2. Decisiones tomadas

1. **Fase 0 se ejecuta íntegramente en Linux/Brais.** Razón dura: SC2 headless oficial solo se distribuye para x86_64. aarch64 (Spark) requeriría emulación (QEMU/Box64/FEX-Emu) que degrada el throughput, que es justo lo que esta fase debe medir.

2. **La DGX Spark queda fuera del scope de Fase 0.** No se monta servicio de inferencia ni infra distribuida en ella ahora.

---

## 3. Decisiones aparcadas (no se tocan en Fase 0)

- **Spark como nodo de inferencia distribuido.** Solo procede revisar en Fase 2/3 si concurren dos condiciones: (a) el slice MIG de 22 GB se queda corto para el modelo de AlphaStar, o (b) el throughput de inferencia local se demuestra cuello de botella en rollouts de RL. AlphaStar original no usa esta arquitectura; añadirla introduce latencia de red, serialización de observaciones y un punto de fallo más. Decisión: revisitar al cierre de Fase 1.
- **Versión exacta de SC2 (Linux headless) y PySC2.** Pendiente. Última build Linux headless oficial conocida es 4.10.x (2019).
- **Gestor de entornos** (env conda nuevo vs venv vs uv). Pendiente. Python 3.13.11 actual no sirve para PySC2.
- **Mini-juego de referencia del benchmark.** El plan menciona `MoveToBeacon` en §7; pendiente ratificar formalmente.
- **Definición concreta de "full game PvP" en el benchmark** (random vs random, random vs built-in, etc.). Pendiente.
- **Mapa principal del proyecto.** Diferido al final de Fase 0/1 según disponibilidad real de replays.
- **Perfil de interfaz — detalles concretos.** Convención decidida (`full` / `human` + ablaciones, ver `00_OVERVIEW.md §2.4`); el wrapper se construye en Fase 1. Por ratificar al inicio de Fase 1: la interfaz de cámara exacta (estilo AlphaStar), resolución de feature layers, modelo de APM/retardo, qué ablaciones (sin minimapa, cuadrante fijo), y si `human` expone las entidades propias vía `use_raw_units` (AlphaStar veía una lista de entidades) o solo feature layers.

---

## 4. Restricciones duras confirmadas (no resueltas, pero asumidas)

| Restricción | Detalle | Implicación |
|---|---|---|
| 12 cores en Linux/Brais | cpuset cgroup del contenedor LXC. No modificable desde dentro. | Rango N del benchmark = 1, 2, 4, 8, 12 (no 16). Techo absoluto de paralelismo SC2. |
| MIG slice 2g.24gb fijo | ~24 GB VRAM, 2 de 7 GPC slices del H100 NVL. Sin permisos para reconfigurar. | Suficiente para Fase 0. Para Fase 1 (BC) será apretado; ratificar al cierre de esta fase. |
| Python 3.10 obligatorio (no 3.11) | `pysc2/lib/colors.py:121` usa `random.shuffle(seq, randfunc)`, API eliminada en Python 3.11. Verificado: 3.11.15 falla con `TypeError: Random.shuffle() takes 2 positional arguments but 3 were given` al importar `pysc2.lib.features`. 3.10.20 carga limpio. | Env conda fijado a `python=3.10`. 3.11 descartado. |
| protobuf < 4 obligatorio | `s2clientprotocol` (dep de PySC2) trae `_pb2.py` generados con layout pre-4.21. protobuf ≥ 4 revienta en `descriptor.py:1027` (`Descriptors cannot be created directly`). Verificado: 7.34.1 falla, 3.20.3 OK. | Pin duro `protobuf<4` en el env. Pip baja `googleapis-common-protos` a 1.73.0 automáticamente para resolver el conflicto. |
| Egress hacia Akamai bloqueado desde Brais | DNS resuelve normal. TCP/443 a cualquier IP de Akamai (probado `blzdistsc2-a.akamaihd.net` con sus dos IPs por geo-DNS, edges alternativos vía `--resolve`, `www.akamai.com`, `www.blizzard.com`) timeout silenciosos. GitHub y otros HTTPS no-Akamai funcionan (200 OK en ~60 ms). mtr no produce hops (probable filtrado ICMP). Causa exacta (firewall LXC / ISP / upstream) no determinable desde dentro del contenedor. | Bloqueante para descarga directa desde Brais de cualquier asset alojado por Blizzard. Workaround obligatorio: sideload (descargar fuera y `scp` a Brais). Afecta también al dataset de replays (Fase 1) — ver row siguiente. |
| Dataset de replays de Blizzard probablemente requiere sideload | Vive en el mismo CDN Akamai bloqueado desde Brais. El riesgo apuntado en `01_PHASE0_infra.md §5` ("Dataset puede no estar accesible") se concreta hoy: la causa probable no es que Blizzard lo haya retirado (al menos no el binario, que sigue vivo: el CDN responde a otras redes), sino que la red de Brais no llega a Akamai. | Verificar al inicio de Fase 1 con un URL del dataset. Mismo workaround que el binario: sideload. |

### 4.1 Banderas abiertas (no bloqueantes ahora, vigilar)

- **numpy 2.2.6 instalado en el env.** PySC2 es de 2019 y numpy 2 eliminó aliases (`np.int`, `np.bool`, `np.float`). Los imports top-level de PySC2 pasan limpios, pero podría romper en runtime al lanzar partidas reales. Si pasa: pin `numpy<2` y reinstalar. **Actualización 2026-05-21:** el smoke test de MoveToBeacon (1 episodio completo con feature layers) corrió sin `AttributeError`; las observaciones salen como `np.int32` (tipo válido). El camino de minijuego está limpio con numpy 2.2.6. Sigue como bandera abierta solo para full-game y observaciones más pesadas, aún sin probar.

---

## 5. Progreso de Fase 0

**Hecho:**
- Survey de hardware (§1, §4).
- Fase 0 íntegramente en Linux/Brais (§2.1).
- Versiones decididas y verificadas experimentalmente en Linux/Brais: Python 3.10.20, PySC2 4.0.0, protobuf 3.20.3, grpcio 1.80.0, numpy 2.2.6.
- Env conda `sc2-rl-infra` creado en Linux/Brais. Imports de PySC2 (`sc2_env`, `actions`, `features`, `colors`) cargan limpios.
- **SC2 4.10.0 (Base build 75689) instalado en Linux/Brais vía sideload** (procedimiento en §6). PySC2 4.0.0 lo reconoce sin forzar `--version`. El paquete oficial ya incluye los mapas `mini_games`, los Ladder 2017-2019 y Melee, así que **no hizo falta descargar mapas aparte**. Integridad verificada (4115224017 bytes idénticos en origen y destino). Ocupa ~4.3 GB en `~/StarCraftII/`, donde PySC2 lo busca por defecto (no se toca `SC2PATH`).
- **Smoke test PySC2↔SC2 superado (2026-05-21).** `python -m pysc2.bin.agent --map MoveToBeacon --agent pysc2.agents.random_agent.RandomAgent --norender --max_episodes 1` completó un episodio entero (1920 game steps) sin crashes — cierra el **objetivo #2 de Fase 0** (PySC2 lanza partidas y el agente las recorre). PySC2 guardó un replay solo en `~/StarCraftII/Replays/`. Las líneas de cierre `return code: -15` (SIGTERM) y `unable to parse websocket frame` son el teardown normal, no errores.
  - Números preliminares (**orientativos, no es el benchmark**; 1 instancia, headless, MoveToBeacon, step_mul=8): arranque en frío ~5,5 s; ~310 agent-steps/s en estado estable.
- **Scaffolding instalado y validado en Brais (2026-05-21).** `pip install -e .` OK; `python -m sc2_rl_infra.demo_random_agent` y `benchmark_throughput` corren end-to-end. El repo se clonó desde GitHub a `~/sc2-rl-infra` (paralelo a `~/StarCraftII/`); actualización vía script `pull_sc2.sh` local.
- **Benchmark de throughput — 1 instancia (incremento 1), medición controlada (2000 steps, `no_op`):**
  - Lanzamiento de SC2: **~5,4 s/instancia** (coste de arranque, métrica §7). Reset de episodio: **~17 ms** (barato; el coste real es lanzar el proceso, no resetear).
  - Throughput: **~210 agent-steps/s** (~1700 game-loops/s ≈ 76× tiempo real), MoveToBeacon headless, step_mul=8.
  - Coste del **raw interface** (`use_feature_units`): **~4%** (212→204 steps/s). Pequeño.
  - **Corrige el ~310 "orientativo" de arriba**: era un run corto y ruidoso; el número fiable es ~210. Justifica medir en serio en vez de fiarse de un run suelto.

- **Benchmark de throughput — barrido en paralelo (incremento 2): HECHO.** N={1,2,4,8,12} instancias, un proceso por instancia. El agregado escala ~lineal hasta 12: **~4744 agent-steps/s a N=12** (CPU 93%, 0 fallos). Hallazgo: la "eficiencia" aparente >100% es artefacto del governor **`schedutil`** (1 instancia corre a ~1.4 GHz; bajo carga, boost a ~2.7-3.7 GHz → ~395/s por instancia). Resultados completos y recomendación en `RESULTS.md`.
- **`RESULTS.md` escrito** con tabla de throughput, el artefacto de frecuencia y la recomendación de N (**default N=8**: ~2766 steps/s al 65% de CPU, deja margen para el entrenamiento de Fase 1).
- **Pipeline de datos validado.** `parse_replay` abre un replay, lo recorre y extrae observaciones (feature layers + estado: `feature_screen`, `feature_minimap`, `available_actions`, `player`…) y acciones del jugador. Probado en un replay 4.10 (240 observaciones, 158 acciones). Fix necesario: pasar `map_data` explícito (SC2 buscaba `maps/` y el dir real es `Maps/` — case-sensitivity en Linux).
- **Entorno congelado en `environment.yml`** (freeze del 2026-05-21; recrear con `conda env create -f environment.yml` + `pip install -e . --no-deps`).

**Fase 0: CERRADA (2026-05-21).** Throughput medido y pipeline de datos demostrado; suficiente para proceder a Fase 1. Único pendiente, **diferido conscientemente a Fase 1** (decisión B):
- **Adquisición del dataset humano** (≥100 replays, parsear ≥1 humano): el *parser* ya está validado; falta *bajar los datos*. Se decide en Fase 1 entre packs 3.16.1 (sideload + instalar SC2 3.16.1) y AlphaStar Unplugged (4.8.2+). Es la primera tarea de Fase 1.
- (Menor) Ratificar `sudo` no-interactivo si hace falta para libs de sistema.

---

## 6. Sideload de SC2 que funcionó (reproducible)

El egress directo a Akamai desde Brais está bloqueado (§4), así que el binario se descarga fuera y se transfiere. Procedimiento confirmado el 2026-05-21:

1. **Descarga** en una máquina con salida a Akamai (aquí Windows 11 / PowerShell). El `--connect-timeout` es clave: si esa red también tuviera el bloqueo, falla en 20 s en vez de colgarse indefinidamente.
   ```powershell
   curl.exe -L -C - --connect-timeout 20 --retry 3 -o "$HOME\Downloads\SC2.4.10.zip" "https://blzdistsc2-a.akamaihd.net/Linux/SC2.4.10.zip"
   ```
2. **Transferencia** a Brais por `scp` (~3.9 GB): `scp SC2.4.10.zip master@<brais>:~/SC2_A/`
3. **Integridad** por tamaño en bytes (debe coincidir origen/destino): `stat -c %s SC2.4.10.zip` → `4115224017`.
4. **Descompresión** (zip cifrado; contraseña oficial de Blizzard `iagreetotheeula`): `unzip -P iagreetotheeula ~/SC2_A/SC2.4.10.zip -d ~/` → crea `~/StarCraftII/`.

**Post-mortem del cuelgue del 2026-05-13:** la sesión anterior se colgó durante la Acción 6 al intentar descargar el binario *directamente desde Brais*; el egress a Akamai da timeout silencioso indefinido y la sesión esperó sin fin. No dejó corrupción en disco — el env conda y el repo sobrevivieron intactos. Lección aplicada arriba: descargar fuera con timeout corto y transferir.
