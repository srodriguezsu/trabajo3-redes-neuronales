"""Carga de configuración YAML con sobreescritura opcional desde CLI/papermill."""
from pathlib import Path

import yaml

RAIZ = Path(__file__).resolve().parents[1]


def cargar_config(nombre: str, **overrides) -> dict:
    with open(RAIZ / "config" / f"{nombre}.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for clave, valor in overrides.items():
        destino = cfg
        *ruta, hoja = clave.split(".")
        for p in ruta:
            destino = destino[p]
        destino[hoja] = valor
    return cfg
