# Notas de trabajo — Fase 0

Documento vivo de decisiones tomadas, restricciones detectadas y decisiones aparcadas durante Fase 0.

Última actualización: 2026-05-28.

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

3. **Monorepo (revisión 2026-05-21).** Todas las fases viven en este repo (`sc2-rl-infra`), organizado por subpaquetes, con un `RESULTS` por fase. Revierte el multi-repo del plan original (`00_OVERVIEW §3.1`): Fase 1 reutiliza código de Fase 0 (parser, perfiles de interfaz, env), así que separar repos forzaría duplicación o `sc2-rl-common` prematuro. Una sesión nueva de Claude Code arranca Fase 1 leyendo esta bitácora.

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
| Egress a `download.pytorch.org` bloqueado desde Brais | `pip install torch --index-url https://download.pytorch.org/whl/cpu` da `Network is unreachable` (igual que Akamai). **PyPI sí funciona** (de ahí salieron las deps del env). | Instalar torch desde PyPI: `pip install torch` (sin `--index-url`). Si PyPI también fallara, sideload del wheel + deps. |

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

---

## 7. Visualización remota (Brais headless → Windows local)

**Contexto (2026-05-25).** Brais es headless (LXC sin monitor; GPU en slice MIG de cómputo). Se trabaja por SSH desde Windows y se quiere ver la salida gráfica (visor de feature layers de PySC2, replays) en el Windows local. La regla de versión sigue intacta: **SC2 corre siempre en Brais** con su build (4.10 hoy; 3.16.1 cuando se sideload-een los packs) y a Windows solo le llega el *resultado* — **nunca se ejecuta SC2 en Windows**.

Tres vías evaluadas:

- **A — Vídeo a fichero.** Brais renderiza a mp4 y `scp` a Windows. La más simple, sin GUI remota ni lag; no interactiva. Buena para clips sueltos.
- **B — Streaming en vivo por VNC (elegida ahora).** Servidor VNC + pantalla virtual (Xvfb/Xvnc) en Brais; túnel SSH (`ssh -L`); visor VNC en Windows. Interactiva y reutilizable para ver agentes en directo (Fases 1+). Coste: instalar VNC + WM mínimo en el contenedor (requiere `sudo`).
- **C — Datos a Windows (alternativa viable, aparcada).** Brais extrae las feature layers a `.npz`; `scp`; se animan/inspeccionan con numpy+matplotlib en Windows, **sin SC2 en Windows**. Interactiva, 100% local e inmune a la regla de versión (solo son arrays). Ideal para depurar fotograma a fotograma. No reproduce la vista oficial coloreada de PySC2 (la dibujamos nosotros). El extractor reutilizaría la lógica de carga de `parse_replay` añadiendo el guardado de `feature_screen`/`feature_minimap`.

**Decisión:** se eligió **B** (visión en vivo). En la práctica el visor *oficial* de PySC2 sobre VNC fue un callejón sin salida, así que B se implementó con un **visor propio por software** (§7.1). **A**/**C** quedan documentadas como alternativas. Las **decisiones finales** de visualización y juego están en **§7.3**.

### 7.1. Implementación de B — lo que funcionó (reproducible, 2026-05-26)

B quedó **operativa**: se ve un agente en directo sobre VNC. Pero **no** con el visor oficial de PySC2 (callejón sin salida, ver abajo) — la vista la dibujamos nosotros por software.

**Montaje VNC (Brais).** TigerVNC + fluxbox, servidor solo en loopback; el túnel lo hace MobaXterm como *SSH gateway (jump host)*. Script: `tools/vnc.sh {start|stop|status|restart}` (crea `~/.vnc/xstartup` con fluxbox si falta).

```
tools/vnc.sh start          # vncserver :1 -geometry 1600x900 -localhost yes
```

Detalle clave del cliente: como el servidor escucha solo en `127.0.0.1`, en MobaXterm el host VNC es `localhost:5901` y se entra por el *SSH gateway* a Brais. Ir directo a la IP de Brais:5901 falla ("cannot reach host on port 5901").

**Callejón sin salida — el visor GL de PySC2.** `pysc2.bin.agent --render` y `SC2Env(visualize=True)` **revientan sobre VNC**:

```
pygame.error: Could not make GL context current: BadAccess          (GLX)
pygame.error: ... eglMakeCurrent failed ... EGL_BAD_ACCESS          (con SDL_VIDEO_X11_FORCE_EGL=1)
```

Causa: el renderer humano crea el contexto OpenGL en el hilo principal y lo usa desde un hilo de render; Mesa software (llvmpipe) aplica de forma estricta la regla "un contexto GL solo activo en un hilo a la vez" → `BAD_ACCESS`. Los drivers NVIDIA son laxos con eso (por eso a otros les funciona); llvmpipe no. **No lo arreglan**: ninguna variable de entorno (GLX ni EGL), ni reiniciar el VNC, ni instalar `libgl1-mesa-dri`/`mesa-utils` (llvmpipe da `direct rendering: Yes`, pero el problema es de hilos, no del driver). **VirtualGL** tampoco vale aquí: la GPU está en **MIG de cómputo** y NVIDIA deshabilita OpenGL/Vulkan bajo MIG. (SC2 mostraba `Creating stub renderer` porque pedimos solo feature layers; el cliente Linux soporta RGB *en teoría*, pero en Brais resultó inviable — intentado y descartado en §7.2.)

**Solución que funciona — visor propio por software.** `sc2_rl_infra/live_view.py`: corre el agente con `visualize=False` (no toca el renderer GL) y pinta las feature layers en una ventana **pygame de software (sin OpenGL), un solo hilo**, reutilizando la paleta oficial de PySC2. Esquiva el muro por completo y no necesita GPU gráfica.

```
tools/vnc.sh start
DISPLAY=:1 python -m sc2_rl_infra.live_view
DISPLAY=:1 python -m sc2_rl_infra.live_view --map CollectMineralShards --step_mul 4 --fps 30
DISPLAY=:1 python -m sc2_rl_infra.live_view --save_replay      # además guarda el .SC2Replay
```

Gotcha imprescindible: al importar pysc2 en headless, SDL queda con un driver de vídeo invisible (`dummy`); `live_view.py` fuerza `SDL_VIDEODRIVER=x11` y reinicia el subsistema antes de abrir la ventana (si no, el bucle corre pero no aparece nada). Al arrancar imprime `[live_view] SDL video driver = x11`.

### 7.2. Gráficos reales (RGB) en Brais — intentado y DESCARTADO (2026-05-26)

El cliente Linux soporta *en teoría* el Rendered Interface (RGB de alta fidelidad, el framebuffer 3D del juego) pidiéndolo (`want_rgb=True` / `rgb_dimensions`) con backend **EGL** (hardware) u **OSMesa** (software). Se intentó en Brais (hubo un modo `--rgb` en `live_view`, ya retirado) y **no funciona aquí**, por tres muros independientes:

1. **EGL hardware** → `Failed to create a valid EGL display! Devices tried: 0`. La GPU está en **MIG de cómputo** y NVIDIA deshabilita los gráficos bajo MIG: no hay dispositivo EGL.
2. **EGL software** (`EGL_PLATFORM=surfaceless` + `LIBGL_ALWAYS_SOFTWARE=1`) → mismo `Devices tried: 0`. El binario de SC2 (2019) **enumera dispositivos EGL explícitamente** e ignora el hint surfaceless.
3. **OSMesa software** → `Failed to load library file!`. La Mesa moderna (25.x, gallium + LLVM) carga en un Python limpio, pero **no dentro del binario de 2019** (conflicto entre el stack pesado de OSMesa y las libs viejas bajo el `LD_LIBRARY_PATH` que PySC2 le pone a SC2).

Tensión de fondo: **binario de 2019 + GPU en MIG (sin gráficos) + Mesa moderna.** Salir exigiría quitar MIG (va contra su propósito de cómputo) o cirugía frágil de librerías. **Decisión: descartado.** No hace falta para RL —el visor de feature layers (§7.1) es la vista útil— y la cinemática se obtiene por otra vía (replays en el cliente de Windows, §7.3).

### 7.3. Decisiones finales de visualización y juego (2026-05-26)

1. **Vista en tiempo real → visor propio por software** (`sc2_rl_infra.live_view`, §7.1). Pinta las feature layers (lo que el agente ve) en una ventana pygame de software sobre VNC. Es la herramienta canónica para mirar agentes en directo en Fases 1+. Sin OpenGL ni GPU gráfica.
2. **Export de replays.** Cada partida puede guardar su `.SC2Replay` (`live_view --save_replay`, vía `save_replay_episodes` de `SC2Env`; van a `~/StarCraftII/Replays/`). Es el artefacto portátil: se copia a Windows (`scp` / SFTP de MobaXterm) y, además de archivarlo, **es la vía a los gráficos reales** — se abre en el cliente de SC2 de Windows y se ve con la cinemática completa (con el cliente en la misma versión que grabó el replay).
3. **Jugar humano-vs-agente → en Windows contra el agente en Brais (por LAN/SSH).** Humano en el cliente de Windows, agente headless en Brais, misma partida (`pysc2.bin.play_vs_agent`). Es **futuro**: requiere SC2 + PySC2 en Windows en versión casada y un agente que merezca la pena enfrentar; rompe la regla "SC2 solo en Brais" de forma **acotada** (solo para jugar/espectar, nunca para el pipeline de datos). La variante nativa estilo AlphaStar (humano en cliente retail + agente vía s2api, gráficos completos) queda como meta lejana.

**Resumen:** feature layers en vivo para depurar (en Brais) · replays como export y como cinemática (vía cliente de Windows) · **gráficos 3D reales vía PySC2 en Windows: validado (§7.4)** · juego humano-vs-IA en Windows (futuro). **RGB directo en Brais: descartado (§7.2); en Windows sí (§7.4).**

### 7.4. Gráficos reales vía PySC2 en Windows — VALIDADO (2026-05-27)

Lo que en Brais se descartó (§7.2) **funciona en Windows**: PySC2 conduce el cliente retail y renderiza el Rendered Interface (3D real) + feature layers, porque en Windows hay GPU con gráficos y display (sin MIG, sin el binario de 2019). Es la vía buena para la cinemática y la base del `play_vs_agent` futuro. La regla sigue intacta: el SC2 del **pipeline de datos** solo en Brais; Windows es para ver/jugar.

**Montaje que funcionó (Windows corporativo, sin admin):**
- **Python 3.10 por zip *embeddable*** (`python-3.10.11-embed-amd64.zip`), no por instalador: el MSI de python.org lo **bloquea la política corporativa** (`ExitCode 1603`, rollback). El embeddable es solo descomprimir (per-user). Para que pip funcione: en `python310._pth` dejar `Lib\site-packages` + `import site`, y bootstrap con `get-pip.py`. Se invoca por ruta (`& $py ...`), no por el launcher `py`.
- **`pip install pysc2 "protobuf<4"`** → pysc2 4.0.0, protobuf 3.20.3 (mismo pin que §4). Python **3.10 obligatorio** también aquí (3.11+ rompe pysc2).
- **Mapas:** el retail los guarda empaquetados (CASC), no sueltos; PySC2 los quiere sueltos en `SC2PATH\Maps\`. Copiados de Brais (`Maps/Melee/Simple64.SC2Map`, `Maps/mini_games/`) a la carpeta Maps de Windows (resultó **escribible sin admin**); las subcarpetas (`Melee/`, `mini_games/`) coinciden con lo que PySC2 espera.
- `$env:SC2PATH = "C:\Program Files (x86)\StarCraft II"`.

**Comandos validados** (PowerShell; `$py` = ruta al python embeddable):
```
& $py -m pysc2.bin.play  --map Simple64                                                       # juego 3D real
& $py -m pysc2.bin.agent --map MoveToBeacon --agent pysc2.agents.scripted_agent.MoveToBeacon  # agente en vivo, renderizado
& $py -m pysc2.bin.play  --replay "ruta\al.SC2Replay"                                          # ver un replay
```

**Version-lock (lo crítico de los replays).** Un `.SC2Replay` exige *exactamente* la build que lo grabó (entre versiones cambian datos/lógica). Brais = **4.10 / Base75689** (última build Linux de Blizzard, 2019); Windows retail = **Base96883** (actual, autoactualizada). Un replay de Brais da en Windows `ValueError: Unknown game version: 4.10.0. Known versions: ['latest']` salvo que exista `Versions\Base75689\`, que se consigue **dejando que el cliente retail la descargue** (abrir un replay 4.10 en el retail dispara el version-switcher); tras eso PySC2 también puede. No es problema de SO (un replay 4.10 se vería en un Windows con 4.10): es que **Linux está congelado en 4.10 y Windows va en la última**.

**Flujo resultante:** Brais = entrenar/datos (headless, sin gráficos por MIG) · Windows = ver/jugar con gráficos. Para revisar partidas con cinemática lo limpio es **grabar en la versión del visor** (Windows actual) y evitar el version-lock; transportar replays 4.10 de Brais exige la Base75689 en Windows. Pendiente: `play_vs_agent` (humano-vs-agente, §7.3.3).

---

## 8. Spike de RL online (Fase 3): A2C MoveToBeacon (2026-05-26)

`sc2_rl_infra/online/a2c_beacon.py` — A2C (FullyConv, PyTorch) que entrena MoveToBeacon renderizando en el visor software, como adelanto de Fase 3. Requiere torch (instalar **desde PyPI**: `pip install torch`; el índice de PyTorch está bloqueado en Brais, ver §4). Se conduce cualquier agente en el visor con `live_view --agent módulo.Clase`.

**Arranque frío (sin shaping).** Corre y entrena (torch entró por PyPI, visor en vivo OK), pero **con un solo env y reward nativo no converge**: a ~180 updates el reward seguía a nivel aleatorio (~0.4, mejor 1). Es **arranque frío** — la recompensa nativa es escasa al principio (el marine rara vez pisa el beacon por azar → pocos +1 → gradiente débil) y un solo env es un setup débil. Sin bug aparente (el agente se mueve, la selección funciona, el loss es no-nulo). Este caso se reproduce hoy con `--noshaped`.

**Reward shaping implementado (2026-05-27, `--shaped` default ON).** Shaping potential-based por distancia marine→beacon (`beacon_distance` sobre `feature_screen[player_relative]`: SELF=1, NEUTRAL=3): `F = shape_coef·(γ·Φ' − Φ)` con `Φ = −dist_norm`, premiando acercarse. Detalles: se **salta en el step en que se toca el beacon** (reaparece lejos → ese salto no es "alejarse") y al terminar el episodio; el reward que se **muestra y se compara con los baselines sigue siendo el nativo** (el shaping solo entra en el cómputo de returns/ventajas). Flags nuevos: `--shaped/--noshaped`, `--shape_coef` (1.0), `--render_every` (1). Techo de referencia: el agente scripted `pysc2.agents.scripted_agent.MoveToBeacon` (vía `live_view --agent`) resuelve el mapa (~25/episodio); el aleatorio (~1) es el suelo.

**Probado en Brais (2026-05-28, 1 env con visor).** Run de ~1000 updates con `--shape_coef 2 --entropy 0.01 --fps 30`: **no convergió**. `mejor=3` (mejoró sobre el random ~1) pero `reward medio(20)` osciló 0.5-0.8 y **bajó** al final del run (0.80 → 0.50 en las últimas ~150 updates). Conjetura: con un solo env esos valores agresivos amplifican el ruido del gradiente y descalibran la política. **Recomendación para el primer run del paralelo: defaults** (`--shape_coef 1`, `--entropy 1e-3`); subirlos solo si sigue plano.

**A2C paralelo + checkpoints + save_replay (2026-05-28, `cf28f08`).** Refactor de `a2c_beacon` con dos modos:
- `--num_envs N` (default 1): con N>1 lanza N subprocesos SC2 vía `multiprocessing.spawn` + pipes, forward batched `(N,2,H,W)`, returns/ventajas vectorizados `(T,N)`, shaping por env. **Headless** (sin visor). Sweet spot N=8 (RESULTS §6). El modo 1-env con visor se preserva intacto.
- `--save_checkpoint_every N` + `--checkpoint_dir`: guarda `{model, optimizer, update, total_steps, best, recent}` cada N updates y al salir (también en Ctrl+C, `try/finally`). `--load_checkpoint <ruta>` reanuda.
- `--save_replay_every N` + `--replay_dir`: `save_replay_episodes` al SC2Env (cada env guarda un `.SC2Replay` cada N episodios suyos).

**Probado en Brais (2026-05-28, paralelo N=12): SPIKE RESUELTO.** Run con `--num_envs 12 --save_checkpoint_every 100 --save_replay_every 200` (defaults para `entropy`/`shape_coef`): a **update 160** (~30 720 transiciones, ~2:30 de pared) el agente alcanza **`reward medio(20) = 25.4`** (techo del scripted ~25) y `mejor = 29`. **El spike de Fase 3 (un A2C aprendiendo MoveToBeacon a nivel del baseline SC2LE) queda demostrado.** Lo que con 1 env y shaping agresivo no convergía, en paralelo con defaults sale en minutos: el escalado a N envs (gradientes decorrelacionados) era la clave.

Throughput observado: **~214 step/s a N=12** — muy por debajo del techo de Fase 0 (4744 step/s con `no_op`, RESULTS §4). El cuello ya no es SC2 sino el padre Python (pickle de los 12 `feature_screen` por step + numpy + torch). Optimizable (mandar solo las 2 capas que usa el modelo, vectorizar `beacon_distance`, etc.), pero **ya no hace falta para el spike** — convergió en minutos igualmente. **Recomendación de receta**: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` delante del comando para que torch no acapare cores que necesitan los workers (RESULTS §6 daba N=8 como sweet spot asumiendo trainer multi-hilo; con torch en 1 hilo, N=12 satura los 12 cores del LXC y converge sin problemas).

Checkpoints en `<--checkpoint_dir>/`:
- `checkpoint_NNNNNN.pt` — snapshot periódico cada `--save_checkpoint_every` updates.
- `checkpoint_final_NNNNNN.pt` — al cerrar (también en Ctrl+C, `try/finally`).
- **`best.pt`** — se sobreescribe cada vez que el `reward medio(20)` supera el récord del run (ventana llena ≥20 episodios). Es **el .pt que querrás cargar para inferencia/demo**: protege de la degradación tardía que vimos en el run 1-env. El wrapper-agente (`A2CCheckpointAgent`) lo prefiere por defecto sobre los numerados. `best_avg` se persiste en el dict del `.pt`, así que `--load_checkpoint` reanuda manteniendo el récord (no sobreescribirá `best.pt` con un avg peor).

Reanudar entrenamiento con `--load_checkpoint <ruta>` (acepta cualquiera de los tres; lo natural es el último numerado o el `_final_`).

**Wrapper-agente para el checkpoint (2026-05-28, `sc2_rl_infra.online.checkpoint_agent.A2CCheckpointAgent`).** Carga un `.pt` y se conduce con el modelo entrenado; compatible con `live_view --agent` (Brais, feature layers + `--save_replay`) y `pysc2.bin.agent --agent` (Windows, 3D real §7.4). Configuración por variables de entorno: `A2C_CHECKPOINT` (ruta del `.pt`; si no se da, usa el más reciente del checkpoint_dir), `A2C_DETERMINISTIC=1` (argmax en vez de muestrear) y `A2C_DEVICE` (cpu/cuda). El módulo **duplica intencionadamente** el `FullyConv` de `a2c_beacon` en vez de importarlo: importar `a2c_beacon` lo carga con sus ~20 flags absl (`step_mul`, `screen`, …) que colisionarían con las de `live_view` / `pysc2.bin.agent`. Mientras el modelo no cambie, esa duplicación es aceptable. Con esto **cierra el ciclo entero del spike**: entrenar paralelo (Brais) → checkpoint → ver al agente entrenado (Brais feature layers o Windows 3D) → `.SC2Replay` portable.

**Extensión a otros minijuegos PySC2 (2026-05-28).** Dos cambios para que el mismo `a2c_beacon` entrene cualquier minijuego cuyo control sea `select_army` + `Move_screen` / `Attack_screen`:

- **Flag `--map <MAP>`** (default `MoveToBeacon`): se pasa a `SC2Env(map_name=...)`. Trivial; sin esto el script estaba cableado a MoveToBeacon.
- **Cabeza de "tipo de acción"** en `FullyConv`: la política ahora factoriza `P(action) = P(spatial) · P(type)` con `type ∈ {Move_screen, Attack_screen}`. Log-prob y entropía suman ambas cabezas (estándar A2C multi-discrete). Worker / `VecSC2Env.step` reciben `(x, y, type_idx)` por pipe. En minijuegos sin Attack disponible (no ocurre en los que nos interesan), fallback automático a Move. Backward-compatible: checkpoints viejos (solo cabeza espacial) cargan con `strict=False` y la cabeza nueva arranca aleatoria → el agente de MoveToBeacon sigue jugando porque `Attack_screen` sobre el beacon (NEUTRAL) equivale a moverse.

**Política por minijuego — qué esperamos** (validar en Brais):

| Mapa | Acciones | Shaping `beacon_distance` | Techo realista con esta arquitectura | Baseline SC2LE |
|---|---|---|---|---|
| **MoveToBeacon** ✓ | Move | Sí (marine→beacon) | ~25 (validado a 25.4) | ~26 |
| **CollectMineralShards** | Move | Sí (los shards son NEUTRAL → reutilizado tal cual como "ir al shard más cercano") | ~17 (limitado: `select_army` arrastra los 2 marines juntos; para ~100 hace falta selección por marine) | ~17 (FullyConv del paper) / ~100 top |
| **FindAndDefeatZerglings** | Move + **Attack** | No (no hay NEUTRAL → se desactiva solo; el `+1/kill -1/muerte` ya es denso) | ~baseline si el agente aprende a *atacar* (lo que con Move solo no podía hacer); el shaping de exploración por visibilidad sería la siguiente palanca si se atasca | ~45 |
| **DefeatZerglingsAndBanelings** | Move + **Attack** | No | Limitado: la arquitectura no separa marines, y vs banelings (AoE) la separación es la jugada óptima. Verá baseline-ish, no top. | ~75 (con split óptimo) |

**Limitación arquitectural pendiente** (no implementada, anotada como futuro): **selección por unidad** (`select_point` sobre cada marine), que es el escalón para acercarse a top-baselines en CollectMineralShards (asignar 1 marine a 1 shard) y DefeatZerglingsAndBanelings (split anti-baneling). Es un cambio sustancial: rompe la simetría "todas mis unidades como grupo", añade una nueva cabeza categórica "qué unidad selecciono" + posiblemente un orden temporal de selección/comando. No procede mientras estemos en spike; se anota como TODO real de Fase 3.

**Ojo:** es un spike, **no la arquitectura de AlphaStar** (Fase 1: encoder de entidades + espacial + LSTM + heads autoregresivos). No sustituye al plan — **Fase 1 (behaviour cloning) sigue siendo el siguiente paso oficial**; el dataset de replays humanos es su primera tarea (ver §5, RESULTS §9).
