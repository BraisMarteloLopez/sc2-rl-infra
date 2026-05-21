# RESULTS — Fase 0: caracterización del entorno y benchmark de throughput

**Fecha:** 2026-05-21
**Máquina:** Linux/Brais (contenedor LXC). NVIDIA H100 NVL (MIG 2g.24gb), AMD EPYC 9534, **12 cores** asignados (cpuset), 59 GB RAM.
**Stack:** SC2 4.10.0 (Base 75689) headless · PySC2 4.0.0 · Python 3.10.20 · protobuf 3.20.3 · numpy 2.2.6.

---

## 1. La pregunta de Fase 0 y la respuesta

> ¿Cuál es el throughput real de SC2 headless en esta máquina y cuántas instancias paralelas se sostienen sin saturar?

**Respuesta:** el throughput agregado **escala de forma aproximadamente lineal hasta las 12 instancias** (techo de cores del contenedor), alcanzando **~4 700 agent-steps/s** con CPU al 93% y **sin fallos**. Es throughput de sobra para Fases 1-3 a escala reducida. **Decisión: proceder a Fase 1.**

---

## 2. Metodología

- Mini-juego **MoveToBeacon**, headless (`--norender`), `step_mul=8`.
- Acción enviada en cada step: **`no_op`**, para medir el coste del *entorno* y no el del agente (en fases siguientes el agente será una red en GPU; medir el sampling de un agente random en CPU no sería representativo).
- 1 agent-step = 8 game-loops. SC2 a tiempo real corre a 22.4 loops/s; las cifras de abajo equivalen a decenas de veces el tiempo real.
- Herramientas (en el paquete `sc2_rl_infra`):
  - `benchmark_throughput` — 1 instancia, N steps fijos, cronometra `env.step()` aislando los resets.
  - `benchmark_parallel` — barrido de N instancias (un proceso por instancia, multiprocessing `spawn`), throughput agregado durante una ventana fija.

---

## 3. Resultados — 1 instancia

| Métrica | Valor |
|---|---|
| Throughput (estado estable) | ~210 agent-steps/s (~1 700 game-loops/s, ≈76× tiempo real) |
| Tiempo de arranque de SC2 | ~5.4 s |
| Reset de episodio | ~17 ms |
| Coste del raw interface (`use_feature_units`) | ~4% |

---

## 4. Resultados — barrido en paralelo

20 s de stepping por instancia, `no_op`, MoveToBeacon headless:

| N | agg steps/s | por instancia | CPU% (sistema) | arranque | fallos |
|---:|---:|---:|---:|---:|---:|
| 1  |  211.6 | 211.6 | 10% | 5.4 s | 0 |
| 2  |  670.9 | 335.4 | 19% | 5.3 s | 0 |
| 4  | 1594.0 | 398.5 | 36% | 5.3 s | 0 |
| 8  | 2766.1 | 345.8 | 65% | 5.7 s | 0 |
| 12 | 4744.1 | 395.3 | 93% | 6.4 s | 0 |

- **0 fallos en todo el barrido**: la máquina sostiene 12 instancias de SC2 concurrentes sin problemas de memoria ni de proceso.
- **CPU sube ~linealmente con N** hasta el 93% a N=12 → cerca de saturar los 12 cores.
- **GPU**: ~0% por diseño (no hay redes en Fase 0; si no fuera ~0, sería un bug).
- **RAM**: no fue cuello de botella (12 instancias OK con 56 GB libres).

---

## 5. Hallazgo importante: artefacto de frecuencia de CPU

La "eficiencia de escalado" calculada respecto a N=1 daba **>100%** (hasta 187%), físicamente imposible. Causa: el governor de CPU es **`schedutil`**, que escala la frecuencia con la carga. Con 1 instancia la máquina está casi ociosa y los cores corren a **~1.4 GHz**; bajo carga suben a boost (~2.7-3.7 GHz). Por eso una instancia *sola* va a ~210/s pero bajo carga cada una va a ~395/s — el ratio (1.87×) coincide con el de frecuencias.

**Implicación:** la métrica honesta es el **throughput agregado** (real, ver §4). El escalado, tomando ~395/s (pleno clock) como base, es **~lineal hasta 12** (4744 ≈ 12 × 395). No controlamos el governor (es del host, sin privilegios en el contenedor), pero en el escenario real de Fase 1+ la máquina estará cargada, así que las cifras "bajo carga" (N≥4) son las representativas.

---

## 6. Recomendación de N para fases siguientes

**Default recomendado: N = 8.**

- N=8 da ~2 766 steps/s con CPU al **65%**, dejando ~4 cores de margen para el proceso de entrenamiento de Fase 1 (red en GPU, data loading, el bucle Python del trainer).
- N=12 maximiza el throughput de rollouts (~4 744 steps/s) pero deja la CPU al 93%, sin margen para nada más en la misma máquina.

Como en Fase 1+ los envs comparten máquina con el entrenamiento, conviene empezar en **N=8** y subir hacia 12 solo si se confirma que el entrenamiento no necesita esos cores. **Revisar al montar el primer pipeline de entrenamiento.**

---

## 7. Problemas encontrados y workarounds (útil para Fase 1)

| Problema | Workaround |
|---|---|
| Egress a Akamai bloqueado desde Brais | Sideload del binario (descarga externa + `scp`). Ver NOTES §6. **Afectará igual al dataset de replays.** |
| Python 3.11 rompe PySC2 | Env fijado a Python 3.10. |
| protobuf ≥ 4 rompe `s2clientprotocol` | Pin `protobuf<4`. |
| Workers multiprocessing (`spawn`) no parsean flags → `UnparsedFlagAccessError` en pysc2 | `flags.FLAGS.mark_as_parsed()` en cada worker. |
| Escalado aparente superlineal | Artefacto de `schedutil`; reportar throughput agregado, no "eficiencia" vs N=1. |
| numpy 2 (riesgo apuntado) | Sin incidencias en minijuego; vigilar en full-game. |

---

## 8. Estado de Fase 0

- [x] SC2 headless instalado y funcional.
- [x] PySC2 lanza partidas; agente random juega (objetivo #2).
- [x] Throughput medido y documentado; recomendación de N.
- [x] `demo_random_agent` y `benchmark_*` reproducibles (`pip install -e .`).
- [ ] **Dataset de replays** (≥100 descargados, ≥1 parseado) — pendiente; mismo bloqueo de Akamai → sideload.
- [ ] Congelar el env en `environment.yml`.

**Decisión: throughput suficiente para proceder a Fase 1.** Único bloqueante restante de Fase 0: acceso al dataset de replays.
