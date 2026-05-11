# Notas de trabajo — Fase 0

Documento vivo de decisiones tomadas, restricciones detectadas y decisiones aparcadas durante Fase 0.

Última actualización: 2026-05-11.

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

---

## 4. Restricciones duras confirmadas (no resueltas, pero asumidas)

| Restricción | Detalle | Implicación |
|---|---|---|
| 12 cores en Linux/Brais | cpuset cgroup del contenedor LXC. No modificable desde dentro. | Rango N del benchmark = 1, 2, 4, 8, 12 (no 16). Techo absoluto de paralelismo SC2. |
| MIG slice 2g.24gb fijo | ~24 GB VRAM, 2 de 7 GPC slices del H100 NVL. Sin permisos para reconfigurar. | Suficiente para Fase 0. Para Fase 1 (BC) será apretado; ratificar al cierre de esta fase. |
| Python 3.13.11 incompatible con PySC2 | PySC2 está estancado; soporta hasta ~3.11. | Crear env conda con Python 3.10 o 3.11. |
| Dataset de Blizzard puede no estar accesible en 2026 | Riesgo declarado en `01_PHASE0_infra.md §5`. | Pendiente verificar acceso (acción aún por planificar). |

---

## 5. Próximo paso pendiente

**Acción 3 — Decisiones de instalación y creación del entorno Python.**

A discutir:
1. Versión exacta de SC2 Linux headless (candidato fuerte: 4.10.0, última build oficial de Blizzard).
2. Versión exacta de PySC2 (candidato fuerte: 4.0.0).
3. Versión exacta de Python (candidato: 3.11).
4. Nombre del env conda y estrategia de pinning (`environment.yml` vs `requirements.txt`).
5. Verificación final de `sudo` no-interactivo con un comando inocuo, antes de depender de él para instalar libs de sistema.
