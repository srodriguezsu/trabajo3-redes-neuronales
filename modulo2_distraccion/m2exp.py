"""Experimentos del módulo 2. Cada uno se lanza con un solo comando y guarda en results/ y models/.

    python modulo2_distraccion/m2exp.py logistica
    python modulo2_distraccion/m2exp.py busqueda --familia cnn
    python modulo2_distraccion/m2exp.py busqueda --familia transfer
    python modulo2_distraccion/m2exp.py semillas --familia cnn
    python modulo2_distraccion/m2exp.py semillas --familia transfer
    python modulo2_distraccion/m2exp.py demo                    # todo en pequeño, < 2 min

Cualquier parámetro de config/modulo2.yaml se cambia con --set clave=valor, por ejemplo
--set transfer.lr=0.001 --set entrenamiento.epocas=10 --set alertas.umbral=0.3
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

AQUI = Path(__file__).resolve().parent
sys.path[:0] = [str(AQUI.parent), str(AQUI)]

import m2lib  # noqa: E402
from common.config import cargar_config  # noqa: E402
from common.seeds import fijar_semilla, semillas_experimento  # noqa: E402

RES = m2lib.RESULTADOS


def config(sets=(), demo=False) -> dict:
    over = {}
    for s in sets:
        k, v = s.split("=", 1)
        over[k] = yaml.safe_load(v)
    cfg = cargar_config("modulo2", **over)
    if demo:
        cfg["demo"]["activo"] = True
    return cfg


def datos(cfg, particion: str, nivel="ninguno"):
    meta = m2lib.validas()
    meta = meta[meta["particion"] == particion]
    if cfg["demo"]["activo"]:
        meta = meta.sample(frac=1, random_state=0).groupby("clase").head(cfg["demo"]["imagenes_por_clase"])
    return meta, m2lib.Imagenes(meta, cfg["datos"]["clases"], nivel)


def hp_final(cfg, familia: str) -> tuple[str, dict]:
    if familia == "cnn":
        return "cnn", dict(cfg["cnn"])
    hp = dict(cfg["transfer"])
    return hp["arquitectura"], hp


def sufijo(cfg):
    return "_demo" if cfg["demo"]["activo"] else ""


# ---------------------------------------------------------------- referencia

def logistica(cfg) -> pd.DataFrame:
    """Regresión logística multinomial sobre características de la red preentrenada congelada.
    C se elige por pérdida logarítmica en validación. Es determinista: las 30 semillas dan lo mismo."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    t0 = time.time()
    meta = m2lib.validas()
    F = m2lib.cargar_caracteristicas()
    clases = cfg["datos"]["clases"]
    y = np.array([clases.index(c) for c in meta["clase"]])
    idx = {p: np.where(meta["particion"] == p)[0] for p in ["entrenamiento", "validacion", "prueba"]}
    if cfg["demo"]["activo"]:
        idx = {p: i[:: max(1, len(i) // (cfg["demo"]["imagenes_por_clase"] * 5))] for p, i in idx.items()}
    filas, mejor = [], None
    for C in cfg["logistica"]["C"]:
        mod = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))
        mod.fit(F[idx["entrenamiento"]], y[idx["entrenamiento"]])
        pv = mod.predict_proba(F[idx["validacion"]])
        m = m2lib.metricas(y[idx["validacion"]], pv, clases)
        filas.append({"C": C, "perdida_val": m["perdida"], "f1_macro_val": m["f1_macro"],
                      "exactitud_val": m["exactitud"]})
        if mejor is None or m["perdida"] < mejor[0]:
            mejor = (m["perdida"], C, mod)
    _, C, mod = mejor
    pp = mod.predict_proba(F[idx["prueba"]])
    RES.mkdir(exist_ok=True)
    s = sufijo(cfg)
    pd.DataFrame(filas).to_csv(RES / f"logistica_busqueda{s}.csv", index=False)
    np.save(RES / f"prob_prueba_logistica{s}.npy", pp)
    m = m2lib.metricas(y[idx["prueba"]], pp, clases)
    lr = mod[-1]
    resumen = {"modelo": "logistica", "C": C, "parametros": int(lr.coef_.size + lr.intercept_.size),
               "configuraciones": len(filas), "segundos": time.time() - t0, **m}
    (RES / f"logistica{s}.json").write_text(json.dumps(resumen, indent=1))
    if not s:
        import joblib
        m2lib.MODELOS.mkdir(exist_ok=True)
        joblib.dump(mod, m2lib.MODELOS / "logistica.joblib")
    print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in resumen.items()}, indent=1))
    return pd.DataFrame(filas)


# ---------------------------------------------------------------- búsqueda

def busqueda(cfg, familia: str) -> pd.DataFrame:
    """Búsqueda aleatoria: n_config configuraciones por familia con el mismo presupuesto de épocas."""
    b = cfg["busqueda"]
    espacio = b[f"espacio_{familia}"]
    rng = random.Random(b["semilla"])
    n = 2 if cfg["demo"]["activo"] else b["n_config"]
    epocas = cfg["demo"]["epocas"] if cfg["demo"]["activo"] else b["epocas"]
    _, ds_e = datos(cfg, "entrenamiento")
    _, ds_v = datos(cfg, "validacion")
    filas = []
    for i in range(n):
        hp = {k: rng.choice(v) for k, v in espacio.items()}
        tipo = "cnn" if familia == "cnn" else hp["arquitectura"]
        fijar_semilla(b["semilla"])
        ds_e.nivel = hp["aumento"]
        red = m2lib.construir(tipo, hp, len(cfg["datos"]["clases"]))
        t0 = time.time()
        h = m2lib.entrenar(red, ds_e, ds_v, hp, epocas, b["paciencia"], cfg["entrenamiento"]["tamano_lote"],
                           b["semilla"])
        mejor = h.loc[h["perdida_val"].idxmin()]
        fila = {"config": i, **{k: str(v) for k, v in hp.items()},
                "parametros": m2lib.parametros_entrenables(red), "epocas": len(h),
                "perdida_val": mejor["perdida_val"], "exactitud_val": mejor["exactitud_val"],
                "segundos": time.time() - t0}
        filas.append(fila)
        print(f"[{familia} {i + 1}/{n}] {hp} -> perdida_val {fila['perdida_val']:.4f} "
              f"exactitud_val {fila['exactitud_val']:.3f} ({fila['segundos']:.0f} s)", flush=True)
        del red
    df = pd.DataFrame(filas).sort_values("perdida_val")
    RES.mkdir(exist_ok=True)
    df.to_csv(RES / f"busqueda_{familia}{sufijo(cfg)}.csv", index=False)
    return df


# ---------------------------------------------------------------- semillas

def semillas(cfg, familia: str) -> pd.DataFrame:
    """Entrena la configuración final con 30 semillas; guarda métricas de prueba, historiales,
    probabilidades de prueba y el modelo de la semilla con menor pérdida de validación."""
    import torch

    tipo, hp = hp_final(cfg, familia)
    clases = cfg["datos"]["clases"]
    demo = cfg["demo"]["activo"]
    lista = semillas_experimento(2 if demo else cfg["semillas"]["n"])
    epocas = cfg["demo"]["epocas"] if demo else cfg["entrenamiento"]["epocas"]
    _, ds_e = datos(cfg, "entrenamiento", hp["aumento"])
    _, ds_v = datos(cfg, "validacion")
    _, ds_p = datos(cfg, "prueba")
    s = sufijo(cfg)
    RES.mkdir(exist_ok=True)
    filas, hists, mejor = [], [], (float("inf"), None)
    for sem in lista:
        fijar_semilla(sem)
        red = m2lib.construir(tipo, hp, len(clases))
        t0 = time.time()
        h = m2lib.entrenar(red, ds_e, ds_v, hp, epocas, cfg["entrenamiento"]["paciencia"],
                           cfg["entrenamiento"]["tamano_lote"], sem)
        seg = time.time() - t0
        pp, yp = m2lib.predecir(red, m2lib.cargador(ds_p, 256, False))
        m = m2lib.metricas(yp, pp, clases)
        lv = h["perdida_val"].min()
        filas.append({"semilla": sem, "perdida_val": lv, "epocas": len(h), "segundos": seg, **m})
        hists.append(h.assign(semilla=sem))
        print(f"[{familia} semilla {sem}] val {lv:.4f}  prueba exactitud {m['exactitud']:.3f} "
              f"f1 {m['f1_macro']:.3f} ({seg:.0f} s)", flush=True)
        if lv < mejor[0]:
            mejor = (lv, sem)
            np.save(RES / f"prob_prueba_{familia}{s}.npy", pp)
            if not demo:
                m2lib.MODELOS.mkdir(exist_ok=True)
                torch.save({"tipo": tipo, "hp": hp, "clases": clases, "semilla": sem,
                            "estado": {k: v.cpu() for k, v in red.state_dict().items()}},
                           m2lib.MODELOS / f"{familia}.pt")
        del red
        torch.cuda.empty_cache()
    df = pd.DataFrame(filas)
    df.to_csv(RES / f"semillas_{familia}{s}.csv", index=False)
    pd.concat(hists).to_csv(RES / f"historial_{familia}{s}.csv", index=False)
    resumen = {"modelo": familia, "tipo": tipo, "hp": hp, "semilla_guardada": mejor[1],
               "parametros": m2lib.parametros_entrenables(m2lib.construir(tipo, hp, len(clases))),
               "segundos_total": float(df["segundos"].sum())}
    (RES / f"semillas_{familia}{s}.json").write_text(json.dumps(resumen, indent=1))
    return df


def cargar_modelo(familia: str):
    import torch

    d = torch.load(m2lib.MODELOS / f"{familia}.pt", map_location="cpu", weights_only=False)
    red = m2lib.construir(d["tipo"], d["hp"], len(d["clases"]))
    red.load_state_dict(d["estado"])
    return red.eval(), d


def demo(cfg):
    """Corrida corta de todo el módulo (< 2 min con GPU): referencia, búsqueda y semillas."""
    t0 = time.time()
    logistica(cfg)
    for fam in ["cnn", "transfer"]:
        busqueda(cfg, fam)
        semillas(cfg, fam)
    print(f"demo terminada en {time.time() - t0:.0f} s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experimento", choices=["logistica", "busqueda", "semillas", "demo"])
    ap.add_argument("--familia", choices=["cnn", "transfer"], default="transfer")
    ap.add_argument("--set", action="append", default=[], metavar="clave=valor")
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()
    cfg = config(a.set, demo=a.demo or a.experimento == "demo")
    {"logistica": lambda: logistica(cfg), "busqueda": lambda: busqueda(cfg, a.familia),
     "semillas": lambda: semillas(cfg, a.familia), "demo": lambda: demo(cfg)}[a.experimento]()
