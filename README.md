# Trabajo 3 — Redes neuronales para series de tiempo, imágenes y recomendación

Redes Neuronales y Algoritmos Bioinspirados · UNAL Medellín · 2026-02. Enunciado: [`docs/enunciado_trabajo3.pdf`](docs/enunciado_trabajo3.pdf).

## Estructura

| Carpeta | Contenido | Parte |
|---|---|---|
| `common/` | Semillas (`seeds.py`) y carga de configuración (`config.py`) compartidas | 1 |
| `config/` | Un YAML por módulo (`modulo1.yaml`, …): hiperparámetros, parámetros operativos, umbrales, K | todos |
| `modulo1_demanda/` | Pronóstico de demanda y flota (notebooks 01–05) | 1 |
| `modulo2_distraccion/` | Clasificación de conducción distractiva | 2 |
| `modulo3_recomendacion/` | Recomendación de destinos | 3 |
| `app/` | Aplicación web que integra los tres módulos | 3 |

Cada módulo sigue la misma convención: `notebooks/`, `data/{raw,processed,external}`, `models/`, `figures/`.

## Instalación

Requiere [uv](https://docs.astral.sh/uv/).

```bash
uv sync                      # crea .venv con versiones fijas (uv.lock)
# alternativa sin uv:
pip install -r requirements.txt
```

## Ejecución

Los notebooks tienen una celda etiquetada `parameters`, así que cada experimento corre con un solo comando y los parámetros se cambian sin editar código:

```bash
uv run papermill modulo1_demanda/notebooks/01_datos_y_series.ipynb /tmp/out.ipynb -p DEMO True
```

Los parámetros largos viven en `config/moduloN.yaml`.

## Reproducibilidad

- Semilla base `42`; las 30 corridas usan `42..71` (`common/seeds.py`).
- Los datos procesados se versionan; los crudos se descargan con el notebook 01 de cada módulo.
- Tras añadir dependencias: `uv add <pkg> && uv export --no-hashes > requirements.txt`.
