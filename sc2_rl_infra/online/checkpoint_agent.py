"""Agente PySC2 que carga un checkpoint A2C (`.pt`) y lo usa para actuar.

Cierra el ciclo del spike de Fase 3:

    Brais entrena (`a2c_beacon --num_envs 12 --save_checkpoint_every 100`)
        -> `.pt` en `~/sc2-rl-infra/checkpoints/a2c_beacon/`
        -> este agente lo carga
        -> `live_view --agent` (Brais, feature layers sobre VNC)
            o `pysc2.bin.agent --agent` (Windows, 3D real, NOTES §7.4)
        -> opcionalmente guarda un `.SC2Replay` portable (`live_view --save_replay`).

`live_view` y `pysc2.bin.agent` instancian los agentes **sin argumentos** (vía
`load_agent("módulo.Clase")`), así que la ruta al checkpoint se pasa por variable
de entorno:

    A2C_CHECKPOINT=<ruta.pt>     ruta a un .pt; si no se da, usa el más reciente
                                 de ~/sc2-rl-infra/checkpoints/a2c_beacon/.
    A2C_DETERMINISTIC=1          (default 0) usa argmax sobre los logits en vez
                                 de muestrear de la política Categorical.
    A2C_DEVICE=cpu               torch device (default cpu; "cuda" en máquinas con GPU).

Uso (Brais, con VNC):
    tools/vnc.sh start
    DISPLAY=:1 A2C_CHECKPOINT=~/sc2-rl-infra/checkpoints/a2c_beacon/checkpoint_000300.pt \\
        python -m sc2_rl_infra.live_view \\
            --agent sc2_rl_infra.online.checkpoint_agent.A2CCheckpointAgent \\
            --save_replay --episodes 3

Uso (Windows, render 3D real — requiere `pip install torch` y `sc2_rl_infra`
importable en el Python embeddable, p.ej. con `PYTHONPATH` apuntando al repo):
    $env:A2C_CHECKPOINT = "C:\\ruta\\al\\checkpoint.pt"
    & $py -m pysc2.bin.agent --map MoveToBeacon \\
        --agent sc2_rl_infra.online.checkpoint_agent.A2CCheckpointAgent --max_episodes 3

Nota técnica: el modelo (FullyConv) se duplica aquí en vez de importarse de
`a2c_beacon`, porque ese módulo define ~20 flags absl que colisionarían con las
de `live_view` / `pysc2.bin.agent` al hacer `importlib.import_module`. Si el
modelo cambia, hay que tocar los dos sitios (mientras el spike sea estable,
esa duplicación es aceptable).
"""

import glob
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pysc2.agents import base_agent
from pysc2.lib import actions, features

_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_SELECTED = features.SCREEN_FEATURES.selected.index

_DEFAULT_CHECKPOINT_DIR = os.path.expanduser("~/sc2-rl-infra/checkpoints/a2c_beacon")


# Modelo: duplicado intencional del FullyConv de a2c_beacon.py (ver "Nota técnica"
# arriba). Misma arquitectura exacta, por eso el state_dict del .pt encaja.
class _FullyConv(nn.Module):
    def __init__(self, in_ch, size):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 16, 5, padding=2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.spatial = nn.Conv2d(32, 1, 1)
        self.value = nn.Linear(32, 1)

    def forward(self, x):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        spatial_logits = self.spatial(h).flatten(1)
        value = self.value(h.mean(dim=(2, 3))).squeeze(-1)
        return spatial_logits, value


def _obs_to_tensor(fs, device):
    """feature_screen ndarray (L, H, W) -> tensor (1, 2, H, W) listo para el modelo."""
    fs = np.asarray(fs)
    pr = fs[_PLAYER_RELATIVE].astype(np.float32) / 4.0    # marine=0.25, beacon=0.75
    sel = (fs[_SELECTED] > 0).astype(np.float32)
    x = np.stack([pr, sel], axis=0)[None]                  # (1, 2, H, W)
    return torch.as_tensor(x, device=device)


def _latest_checkpoint(dirpath):
    """Devuelve la ruta al .pt más reciente en `dirpath`, o None si no hay."""
    if not os.path.isdir(dirpath):
        return None
    pts = sorted(glob.glob(os.path.join(dirpath, "*.pt")), key=os.path.getmtime)
    return pts[-1] if pts else None


def _truthy(val):
    return str(val).strip().lower() not in ("", "0", "false", "no", "off")


class A2CCheckpointAgent(base_agent.BaseAgent):
    """PySC2 agent que carga un checkpoint A2C y se conduce con el modelo.

    Compatible con `live_view --agent` y `pysc2.bin.agent --agent`. Toda la
    configuración (ruta del .pt, determinista vs muestreo, device) viene por
    variables de entorno — ver docstring del módulo.
    """

    def __init__(self):
        super().__init__()
        ckpt_path = os.environ.get("A2C_CHECKPOINT", "").strip()
        if not ckpt_path:
            ckpt_path = _latest_checkpoint(_DEFAULT_CHECKPOINT_DIR)
            if ckpt_path is None:
                raise RuntimeError(
                    f"No hay checkpoints en {_DEFAULT_CHECKPOINT_DIR} y no se pasó "
                    "A2C_CHECKPOINT. Pasa la ruta con esa variable de entorno."
                )
        ckpt_path = os.path.expanduser(ckpt_path)
        self.device = os.environ.get("A2C_DEVICE", "cpu")
        if self.device == "cuda" and not torch.cuda.is_available():
            print("[A2CCheckpointAgent] CUDA no disponible, usando CPU.", flush=True)
            self.device = "cpu"
        self.deterministic = _truthy(os.environ.get("A2C_DETERMINISTIC", "0"))

        ckpt = torch.load(ckpt_path, map_location="cpu")
        self._ckpt_state = ckpt["model"]
        # in_ch sale del state_dict: conv1.weight tiene shape (out_ch, in_ch, k, k).
        self._in_ch = int(self._ckpt_state["conv1.weight"].shape[1])

        update = ckpt.get("update", "?")
        best_raw = ckpt.get("best", float("nan"))
        try:
            best_str = f"{float(best_raw):.2f}"
        except (TypeError, ValueError):
            best_str = str(best_raw)
        print(
            f"[A2CCheckpointAgent] checkpoint cargado: {ckpt_path} "
            f"(update {update}, best {best_str}, "
            f"determinista={self.deterministic}, device={self.device})",
            flush=True,
        )
        self.size = None
        self.model = None

    def setup(self, obs_spec, action_spec):
        super().setup(obs_spec, action_spec)
        # Tamaño de feature_screen desde el obs_spec; obs_spec["feature_screen"]
        # es (num_layers, H, W); H == W para nosotros.
        screen_shape = obs_spec["feature_screen"]
        self.size = int(screen_shape[1])
        self.model = _FullyConv(in_ch=self._in_ch, size=self.size).to(self.device)
        self.model.load_state_dict(self._ckpt_state)
        self.model.eval()

    def step(self, obs):
        super().step(obs)
        # MoveToBeacon: si el marine aún no está seleccionado, hazlo (mismo patrón
        # que select_if_needed en a2c_beacon). Cualquier paso de selección no es
        # decisión del modelo, así que ni siquiera hacemos forward.
        if _MOVE_SCREEN not in obs.observation["available_actions"]:
            return actions.FUNCTIONS.select_army("select")
        fs = np.asarray(obs.observation["feature_screen"])
        with torch.no_grad():
            logits, _ = self.model(_obs_to_tensor(fs, self.device))
            if self.deterministic:
                idx = int(torch.argmax(logits, dim=-1).item())
            else:
                idx = int(torch.distributions.Categorical(logits=logits).sample().item())
        y, x = idx // self.size, idx % self.size
        return actions.FUNCTIONS.Move_screen("now", [int(x), int(y)])
