# Fase 0: Infraestructura y caracterización del entorno

**Repositorio:** `sc2-rl-infra`
**Duración estimada:** 1-2 semanas (hasta 4 si hay fricciones)
**Bloqueante para:** todas las fases posteriores

---

## 1. Por qué existe esta fase

La razón principal de Fase 0 es **una sola hipótesis crítica que necesitamos validar antes de comprometernos a las siguientes fases**:

> ¿Cuál es el throughput real de SC2 headless en nuestra máquina, y cuántas instancias paralelas podemos sostener sin saturar CPU/RAM?

Esta pregunta determina si los planes de Fases 1-3 son siquiera factibles en escala razonable. Si el throughput es demasiado bajo, hay que ajustar expectativas o cambiar de hardware antes de invertir tiempo en arquitecturas neurales.

Razones secundarias:

- Establecer wrappers de observación/acción que se reutilizarán en todas las fases.
- Verificar acceso al dataset de replays de Blizzard.
- Construir un baseline trivial (agente random) contra el cual comparar todo lo demás.

---

## 2. Objetivos específicos

1. SC2 headless instalado y funcional en la máquina objetivo.
2. PySC2 instalado, comunicándose con SC2, capaz de lanzar partidas mini-juego y full game.
3. Métricas de throughput documentadas: steps/segundo en una instancia, en N instancias paralelas, con y sin renderizado.
4. Wrappers básicos de observación (espacial + entidades) y acción (estructura jerárquica) — **diseño definido, construcción diferida a Fase 1** (cuando la arquitectura los consume; no especulativamente en Fase 0, ver §5). Serán conscientes del **perfil de interfaz** (`full` / `human`, ver `00_OVERVIEW.md §2.4`).
5. Agente random capaz de jugar mini-juegos y partidas full game (perdiendo, pero sin crashes).
6. Acceso confirmado al dataset de replays de Blizzard, formato entendido, scripts de parseo básicos funcionando.

---

## 3. Entregables

- Repositorio `sc2-rl-infra` con código instalable (`pip install -e .` o similar).
- `README.md` con instrucciones reproducibles de instalación desde cero.
- `RESULTS.md` con:
  - Tabla de throughput medido (instancias × steps/segundo × utilización CPU/GPU).
  - Decisión sobre número de instancias paralelas a usar en fases siguientes.
  - Decisión sobre mapa/raza a usar como default.
  - Lista de problemas encontrados y workarounds aplicados (esto será valioso documentar para Fase 1).
- Notebook o script de demo: agente random jugando un mini-juego y un full game.

---

## 4. Criterios de éxito

Fase 0 está **terminada** cuando:

1. Se puede lanzar `python -m sc2_rl_infra.demo_random_agent` y ver al agente jugar un mini-juego sin errores.
2. Se puede lanzar `python -m sc2_rl_infra.benchmark_throughput --n-envs 8` y obtener números.
3. El dataset de replays se ha descargado parcialmente (al menos 100 replays) y se ha parseado al menos uno con éxito, extrayendo observaciones y acciones en el formato que usará Fase 1.
4. `RESULTS.md` contiene una recomendación explícita sobre cuántos envs paralelos usar en fases siguientes, justificada con datos.

---

## 5. Lo que NO se hace en Fase 0

Lista explícita para evitar scope creep:

- **No** se entrenan redes neuronales. Ni siquiera pequeñas.
- **No** se implementa la arquitectura de AlphaStar. Eso es Fase 1.
- **No** se hace imitation learning. Eso es Fase 1.
- **No** se optimiza el código más allá de lo necesario para medir throughput. Optimizar prematuramente sin saber dónde está el cuello de botella es desperdicio.
- **No** se diseñan abstracciones genéricas "para cuando hagan falta". Si no se usan en esta fase, no se construyen.

---

## 6. Riesgos específicos de esta fase

| Riesgo | Mitigación |
|---|---|
| SC2 Linux headless da problemas en la distribución concreta | Tener plan B con Docker oficial de Blizzard |
| Versión de PySC2 incompatible con versión de SC2 | Fijar versiones explícitamente en el primer commit |
| Throughput mucho menor de lo esperado | Es información válida; documentar y replantear escala de Fase 1 antes de seguir |
| Dataset de replays no accesible o el link oficial está roto | Verificar acceso en el día 1, no al final de la fase |

---

## 7. Métricas mínimas a reportar en `RESULTS.md`

- Steps/segundo con 1 instancia, sin renderizado, mini-juego MoveToBeacon.
- Steps/segundo con 1 instancia, full game PvP, sin renderizado.
- Steps/segundo total con N instancias paralelas (N a determinar empíricamente: 1, 2, 4, 8, 12 — techo de 12 por el cpuset del contenedor LXC, ver NOTES §4).
- Utilización media de CPU y RAM con N óptimo.
- Utilización de GPU (debería ser casi cero en esta fase; si no, hay un bug).
- Tiempo medio de inicio de una instancia de SC2 (relevante para fases con muchos resets).

---

## 8. Pregunta abierta para el final de Fase 0

Al cierre de la fase, hay una decisión a tomar explícitamente en `RESULTS.md`:

> Dado el throughput medido, ¿es razonable proceder con Fase 1 tal como está planificada, o hay que ajustar (menos replays, menor batch size, etc.)?

Si la respuesta es "no es razonable", la siguiente acción es **renegociar el plan general**, no empezar Fase 1 con expectativas que no se van a cumplir.
