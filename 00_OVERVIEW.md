# Plan de Trabajo: Replicación de AlphaStar y Experimentación RL en StarCraft II

**Versión:** 0.1 (draft)
**Fecha:** Mayo 2026
**Estado:** Plan estratégico — pendiente de validación experimental en Fase 0

---

## 1. Objetivo del proyecto

Construir un entendimiento profundo y código propio funcional que replique, a escala factible con recursos no-corporativos, las técnicas centrales de **AlphaStar (Vinyals et al., 2019)** y **AlphaStar Unplugged (Mathieu et al., 2023)**, los dos trabajos de referencia de DeepMind sobre StarCraft II.

**No es objetivo del proyecto:**

- Alcanzar nivel competitivo en SC2 full game contra humanos o bots top.
- Publicar resultados académicos.
- Producir un agente desplegable en AI Arena u otras escaleras competitivas.

**Sí es objetivo del proyecto:**

- Implementar la arquitectura neural de AlphaStar y entrenarla con behaviour cloning sobre replays humanos.
- Reproducir a escala reducida los experimentos principales de AlphaStar Unplugged (offline RL sobre dataset de replays).
- Entrenar agentes con RL online en mini-juegos PySC2 y compararlos con baselines publicados.
- Opcionalmente, explorar world models (Dreamer-family) como contribución original sobre la base anterior.

---

## 2. Restricciones y supuestos

### 2.1 Recursos

- **Cómputo:** 1× NVIDIA H100. Es el cuello de botella principal del proyecto.
- **Entorno:** Linux nativo, SC2 headless.
- **Framework:** PyTorch.

### 2.2 Implicaciones honestas del cómputo disponible

Una H100 es ~3-4 órdenes de magnitud por debajo de los recursos de AlphaStar original (16 TPU v3 pods × 44 días). Además, **el cuello de botella en SC2 no es la GPU sino el entorno**: SC2 corre en CPU y satura mucho antes que la H100. Esto significa:

- Pipeline de league training con self-play multi-agente a gran escala: **no viable.**
- Behaviour cloning sobre subset reducido del dataset de replays: **viable.**
- Offline RL al estilo AlphaStar Unplugged a escala reducida: **viable.**
- RL online sobre mini-juegos PySC2: **viable.**
- RL online sobre full game con resultado competitivo: **no viable, pero sí instructivo a escala reducida.**

### 2.3 Convenciones de raza y mapa

Para reducir el espacio de variables y mantener los experimentos comparables entre fases:

- **Raza principal:** Protoss. Justificación: menor número de unidades distintas, mecánicas más simples (sin queens larva injection, sin MULEs), composiciones de ejército más limpias.
- **Matchup principal:** Protoss vs Protoss (PvP). Justificación: simetría, dataset abundante.
- **Mapa principal:** Por definir tras Fase 0 (depende de cuáles tengan mejor representación en el subset del dataset que descarguemos).

Estas decisiones se revisan al final de Fase 1 si hay evidencia experimental para cambiarlas.

---

## 3. Estructura del proyecto

### 3.1 Multi-repo, un repositorio por fase

| Fase | Repositorio | Propósito |
|---|---|---|
| Fase 0 | `sc2-rl-infra` | Entorno, wrappers, benchmarking de throughput |
| Fase 1 | `sc2-rl-imitation` | Behaviour cloning sobre replays (arquitectura AlphaStar) |
| Fase 2 | `sc2-rl-offline` | Offline RL al estilo AlphaStar Unplugged |
| Fase 3 | `sc2-rl-online` | RL online en mini-juegos (PPO/IMPALA) |
| Fase 4 | `sc2-rl-worldmodels` | (Opcional) DreamerV3 y derivados en mini-juegos |

### 3.2 Código compartido entre repositorios

Habrá inevitablemente duplicación de wrappers de observación, encoders básicos y utilidades. Decisión: **aceptar duplicación inicial, refactorizar a librería compartida (`sc2-rl-common`) solo cuando el coste de mantener duplicado supere al coste de extraer**. Anti-patrón a evitar: crear `sc2-rl-common` en Fase 0 sin saber qué será realmente común.

### 3.3 Estado y entregables por fase

Cada fase produce:

- Código funcional reproducible (scripts de entrenamiento + evaluación).
- Un informe técnico interno (`RESULTS.md`) con métricas obtenidas, comparación con baselines publicados, y modos de fallo observados.
- Checkpoints de modelos relevantes.
- Decisión explícita: ¿avanzar a la siguiente fase, iterar en esta, o pivotar?

**Regla dura:** no se inicia una fase hasta que la anterior ha producido `RESULTS.md` y un agente evaluable. Si la fase se atasca, se documentan los motivos y se decide pivotar o cerrar.

---

## 4. Criterios de éxito globales

El proyecto se considera **exitoso a nivel global** si al cierre de Fase 3 se cumple:

1. Existe código propio que reproduce la arquitectura central de AlphaStar (encoder de entidades + encoder espacial + LSTM + heads autoregresivos de acción).
2. Existe un agente de imitation learning entrenado sobre replays humanos que supera consistentemente al bot interno de SC2 en dificultad "Very Easy" en al menos una raza.
3. Existe una replicación numérica (aunque sea a escala reducida) de al menos uno de los algoritmos de AlphaStar Unplugged, con resultados comparables cualitativamente a los del paper.
4. Existen agentes de RL online entrenados en al menos 3 mini-juegos PySC2 con métricas comparables a los baselines publicados en SC2LE (Vinyals et al., 2017).

Cualquier resultado por debajo de esto que aún produzca entendimiento documentado es un éxito parcial aceptable. Fase 4 es 100% opcional y no condiciona el éxito global.

---

## 5. Riesgos principales y mitigaciones

| Riesgo | Probabilidad | Impacto | Mitigación |
|---|---|---|---|
| Throughput de SC2 demasiado bajo en H100+CPU disponible | Alta | Alto | Fase 0 es enteramente sobre medir esto antes de comprometerse al resto |
| Dataset de replays inaccesible o formato cambia | Media | Alto | Verificar acceso al dataset de Blizzard ANTES de Fase 1 |
| Behaviour cloning no converge a algo jugable | Media | Medio | Empezar con subset muy pequeño y arquitectura mínima viable, escalar gradualmente |
| Offline RL no mejora sobre imitation baseline | Media | Bajo | Es resultado válido por sí mismo, no implica fracaso del proyecto |
| Sobre-ingeniería temprana | Alta | Alto | Regla explícita: no abstracciones especulativas hasta Fase 2 |
| Fatiga / abandono por proyecto demasiado largo | Alta | Alto | Hitos pequeños, cada fase produce algo evaluable en semanas no meses |

---

## 6. Estimaciones temporales (orientativas, no compromisos)

Asumiendo dedicación parcial (no full-time) y un solo desarrollador:

| Fase | Estimación | Rango realista |
|---|---|---|
| Fase 0 | 1-2 semanas | hasta 4 si hay problemas de entorno |
| Fase 1 | 3-4 semanas | hasta 8 si el dataset requiere mucho preprocesado |
| Fase 2 | 3-4 semanas | hasta 6 |
| Fase 3 | 2-3 semanas | hasta 5 |
| Fase 4 | 4-6 semanas | abierto, es investigación |

**Total realista hasta cerrar Fase 3:** 3-5 meses de trabajo efectivo. Si esto te parece poco, probablemente subestimas el tiempo que vas a perder con bugs de PySC2 y formato de replays.

---

## 7. Referencias técnicas base

Las dos referencias principales del proyecto, en orden de relevancia para implementación:

1. **Vinyals et al. (2019)** — *Grandmaster level in StarCraft II using multi-agent reinforcement learning*. Nature. Arquitectura, pipeline, decisiones de diseño.
2. **Mathieu et al. (2023)** — *AlphaStar Unplugged: Large-Scale Offline Reinforcement Learning*. arXiv:2308.03526. Benchmark formal, dataset estandarizado, algoritmos offline.

Referencias secundarias:

3. **Vinyals et al. (2017)** — *StarCraft II: A New Challenge for Reinforcement Learning* (SC2LE). Mini-juegos y baselines.
4. **mini-AlphaStar** (Liu et al., comunidad) — Referencia de implementación en PyTorch a escala reducida. Útil como guía pero NO como código a copiar verbatim.

Las referencias del informe original sobre Dreamer, NE-Dreamer, UniZero, Agentic RL, etc., son relevantes solo para Fase 4 y están sujetas a verificación independiente (algunas referencias del informe original tienen IDs de arXiv que no he validado todavía).

---

## 8. Decisiones abiertas pendientes

Esta lista vive en el overview y se actualiza al cierre de cada fase:

- [ ] Mapa concreto a usar (depende de Fase 0 + análisis del dataset).
- [ ] Tamaño del subset de replays a descargar (depende de Fase 0).
- [ ] Algoritmo offline RL específico a implementar primero en Fase 2 (Offline Actor-Critic vs MuZero Unplugged vs alternativa).
- [ ] Si en Fase 3 se usa PPO clásico, IMPALA, o algo más reciente.
- [ ] Si Fase 4 se ejecuta o se cierra el proyecto en Fase 3.

---

## 9. Anti-objetivos explícitos

Para evitar scope creep, declaro explícitamente qué NO va a hacer este proyecto:

- No se va a reproducir el "League Training" con poblaciones de agentes (main agents, league exploiters, main exploiters). Requiere computación no disponible.
- No se va a entrenar ningún agente en SC2 full game con expectativa de competitividad.
- No se va a desplegar nada en AI Arena ni participar en ladders públicos.
- No se va a integrar con LLMs ni hacer "agentic RL" salvo como exploración opcional muy posterior.
- No se va a portar nada a JAX por consistencia con el paper original. PyTorch es decisión cerrada.
