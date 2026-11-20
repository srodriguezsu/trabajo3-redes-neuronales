"""Funciones reutilizadas por los notebooks del Módulo 1 (datos, métricas, modelos, flota)."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests

URL_SOCRATA = "https://www.datos.gov.co/resource/{id}.json"


def filtro_fechas(desde: str, hasta: str) -> str:
    """El conjunto trae fechas imposibles (p. ej. año 6202): se acota explícitamente."""
    return f"fecha_despacho between '{desde}T00:00:00' and '{hasta}T23:59:59'"


# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------
def soql(dataset_id: str, params: dict, pagina: int = 50_000, reintentos: int = 3) -> pd.DataFrame:
    """Consulta SoQL paginada. La agregación se hace en el servidor (el crudo tiene ~49M filas)."""
    url = URL_SOCRATA.format(id=dataset_id)
    lotes, offset = [], 0
    while True:
        q = {**params, "$limit": pagina, "$offset": offset}
        for intento in range(reintentos):
            try:
                r = requests.get(url, params=q, timeout=300)
                r.raise_for_status()
                break
            except requests.RequestException:
                if intento == reintentos - 1:
                    raise
                time.sleep(5 * (intento + 1))
        lote = r.json()
        if not lote:
            break
        lotes.append(pd.DataFrame(lote))
        if len(lote) < pagina:
            break
        offset += pagina
    return pd.concat(lotes, ignore_index=True) if lotes else pd.DataFrame()


def rutas_principales(dataset_id: str, n: int, fechas: str) -> pd.DataFrame:
    """Top-n rutas (origen→destino) por pasajeros despachados en origen."""
    df = soql(dataset_id, {
        "$select": "municipio_origen_ruta, municipio_destino_ruta, sum(pasajeros) as pasajeros, "
                   "sum(despachos) as despachos, min(fecha_despacho) as desde, max(fecha_despacho) as hasta",
        "$where": f"tipo_despacho = 'ORIGEN' AND {fechas}",
        "$group": "municipio_origen_ruta, municipio_destino_ruta",
        "$order": "pasajeros DESC",
    }, pagina=n)
    df = df.head(n)
    for c in ("pasajeros", "despachos"):
        df[c] = pd.to_numeric(df[c])
    return df


def diario_por_ruta(dataset_id: str, pares: list[tuple[str, str]], fechas: str) -> pd.DataFrame:
    """Pasajeros y despachos por ruta, día y clase de vehículo (agregado en el servidor)."""
    filtro_rutas = " OR ".join(
        f"(municipio_origen_ruta = '{o}' AND municipio_destino_ruta = '{d}')" for o, d in pares)
    df = soql(dataset_id, {
        "$select": "municipio_origen_ruta, municipio_destino_ruta, fecha_despacho, clase_vehiculo, "
                   "sum(pasajeros) as pasajeros, sum(despachos) as despachos",
        "$where": f"tipo_despacho = 'ORIGEN' AND {fechas} AND ({filtro_rutas})",
        "$group": "municipio_origen_ruta, municipio_destino_ruta, fecha_despacho, clase_vehiculo",
        "$order": "fecha_despacho",
    })
    df["fecha"] = pd.to_datetime(df["fecha_despacho"]).dt.normalize()
    for c in ("pasajeros", "despachos"):
        df[c] = pd.to_numeric(df[c])
    df["ruta"] = df["municipio_origen_ruta"] + "-" + df["municipio_destino_ruta"]
    return df.drop(columns="fecha_despacho")


def dias_con_reporte(dataset_id: str, fechas: str) -> pd.DatetimeIndex:
    """Días en que el sistema recibió algún reporte (para distinguir 'cero real' de 'sin datos')."""
    df = soql(dataset_id, {"$select": "fecha_despacho, count(*) as n", "$where": fechas, "$group": "fecha_despacho"})
    return pd.DatetimeIndex(pd.to_datetime(df["fecha_despacho"]).dt.normalize()).sort_values()


def completar_calendario(serie: pd.DataFrame, dias_validos: pd.DatetimeIndex) -> pd.DataFrame:
    """Reindexa a frecuencia diaria. Día sin registro de la ruta pero con reporte del sistema → 0;
    día sin ningún reporte en el sistema → NaN (falta de datos, no de demanda)."""
    idx = pd.date_range(serie.index.min(), serie.index.max(), freq="D")
    out = serie.reindex(idx)
    sin_sistema = ~idx.isin(dias_validos)
    out.loc[~sin_sistema] = out.loc[~sin_sistema].fillna(0)
    return out


def marcar_atipicos(y: pd.Series, ventana: int = 28, k: float = 5.0) -> pd.Series:
    """Atípicos por desviación robusta (MAD) respecto a la mediana móvil del mismo día de la semana."""
    ref = y.groupby(y.index.dayofweek).transform(
        lambda s: s.rolling(ventana // 7, center=True, min_periods=2).median())
    resid = y - ref
    mad = resid.abs().rolling(ventana, center=True, min_periods=7).median() * 1.4826
    return (resid.abs() > k * mad.replace(0, np.nan)).fillna(False)


def imputar_estacional(y: pd.Series) -> pd.Series:
    """Imputa NaN con el valor de la semana anterior (y, si falta, la siguiente)."""
    y = y.copy()
    for _ in range(4):
        faltan = y.isna()
        if not faltan.any():
            break
        y[faltan] = y.shift(7)[faltan]
        y[y.isna()] = y.shift(-7)[y.isna()]
    return y.interpolate(limit_direction="both")


# ---------------------------------------------------------------------------
# Calendario (variables exógenas conocidas con 30 días de anticipación)
# ---------------------------------------------------------------------------
def tabla_calendario(idx: pd.DatetimeIndex, festivos: pd.DataFrame) -> pd.DataFrame:
    """festivos: DataFrame con columna `fecha` (y opcional `nombre`). Todas las variables son
    deterministas a partir de la fecha → conocidas en el momento de la decisión."""
    f = set(pd.to_datetime(festivos["fecha"]).dt.normalize())
    cal = pd.DataFrame(index=idx)
    cal["dow"] = idx.dayofweek
    cal["festivo"] = idx.isin(f).astype(int)
    cal["puente"] = (cal["festivo"].eq(1) & cal["dow"].isin([0, 4])).astype(int)
    # Días previos/posteriores a festivo (salida y retorno de puentes)
    cal["pre_festivo"] = pd.Series(idx + pd.Timedelta(days=1), index=idx).isin(f).astype(int)
    cal["post_festivo"] = pd.Series(idx - pd.Timedelta(days=1), index=idx).isin(f).astype(int)
    doy = idx.dayofyear
    cal["sin_anual"] = np.sin(2 * np.pi * doy / 365.25)
    cal["cos_anual"] = np.cos(2 * np.pi * doy / 365.25)
    cal["fin_de_anio"] = (((idx.month == 12) & (idx.day >= 15)) | ((idx.month == 1) & (idx.day <= 15))).astype(int)
    cal["vacaciones_mitad"] = (((idx.month == 6) & (idx.day >= 15)) | ((idx.month == 7) & (idx.day <= 15))).astype(int)
    nombres = festivos.set_index(pd.to_datetime(festivos["fecha"]).dt.normalize()).get("nombre")
    cal["semana_santa"] = 0
    if nombres is not None:
        jueves = nombres[nombres.str.contains("Jueves Santo", case=False, na=False)].index
        for j in jueves:
            cal.loc[(idx >= j - pd.Timedelta(days=4)) & (idx <= j + pd.Timedelta(days=4)), "semana_santa"] = 1
    for d in range(7):
        cal[f"dow_{d}"] = (cal["dow"] == d).astype(int)
    return cal.drop(columns="dow")


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------
def mae(y, p):
    return float(np.nanmean(np.abs(np.asarray(y) - np.asarray(p))))


def rmse(y, p):
    return float(np.sqrt(np.nanmean((np.asarray(y) - np.asarray(p)) ** 2)))


def escala_mase(y_train: np.ndarray, m: int = 7) -> float:
    """Denominador del MASE: MAE in-sample del ingenuo estacional de periodo m."""
    y_train = np.asarray(y_train, float)
    return float(np.nanmean(np.abs(y_train[m:] - y_train[:-m])))


def mase(y, p, escala: float):
    return mae(y, p) / escala


def perdida_cuantil(y, q_pred, tau: float):
    """Pinball loss: ρ_τ(u) = u(τ − 1{u<0}), u = y − q."""
    u = np.asarray(y) - np.asarray(q_pred)
    return float(np.nanmean(np.maximum(tau * u, (tau - 1) * u)))


def cobertura(y, q_pred):
    """Proporción de observaciones por debajo del cuantil pronosticado (ideal: τ)."""
    y, q = np.asarray(y), np.asarray(q_pred)
    ok = ~np.isnan(y)
    return float(np.mean(y[ok] <= q[ok]))


BLOQUES_HORIZONTE = {"1-7": (1, 7), "8-14": (8, 14), "15-30": (15, 30)}


# ---------------------------------------------------------------------------
# Evaluación con origen rodante
# ---------------------------------------------------------------------------
def origenes_rodantes(idx: pd.DatetimeIndex, inicio_prueba: pd.Timestamp, h: int, paso: int) -> list[pd.Timestamp]:
    """Orígenes t (último día observado) tales que [t+1, t+h] cae dentro del periodo de prueba."""
    ult = idx.max() - pd.Timedelta(days=h)
    t0 = inicio_prueba - pd.Timedelta(days=1)
    return list(pd.date_range(t0, ult, freq=f"{paso}D"))


def tabla_errores(registros: pd.DataFrame, escalas: dict[str, float], tau: float) -> pd.DataFrame:
    """registros: columnas modelo, ruta, origen, h, y, pred, q_alto."""
    filas = []
    for (mod, ruta), g in registros.groupby(["modelo", "ruta"]):
        filas.append({"modelo": mod, "ruta": ruta, "MAE": mae(g.y, g.pred), "RMSE": rmse(g.y, g.pred),
                      "MASE": mase(g.y, g.pred, escalas[ruta]),
                      f"cobertura_P{int(tau*100)}": cobertura(g.y, g.q_alto),
                      "pinball": perdida_cuantil(g.y, g.q_alto, tau)})
    return pd.DataFrame(filas)


def errores_por_bloque(registros: pd.DataFrame) -> pd.DataFrame:
    filas = []
    for nombre, (a, b) in BLOQUES_HORIZONTE.items():
        g = registros[registros.h.between(a, b)]
        for mod, gm in g.groupby("modelo"):
            filas.append({"modelo": mod, "bloque": nombre, "MAE": mae(gm.y, gm.pred), "RMSE": rmse(gm.y, gm.pred)})
    return pd.DataFrame(filas).pivot(index="modelo", columns="bloque", values="MAE")[list(BLOQUES_HORIZONTE)]


# ---------------------------------------------------------------------------
# Modelos de referencia
# ---------------------------------------------------------------------------
def ingenuo_estacional(hist: pd.Series, h: int, tau: float, m: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """Repite la última semana. Cuantil alto: cuantil empírico de los errores históricos del método."""
    ult = hist.values[-m:]
    pred = np.array([ult[i % m] for i in range(h)], float)
    err = (hist - hist.shift(m)).dropna().values
    q = pred + np.quantile(err, tau)
    return pred, np.maximum(q, pred)


def ets_semanal(hist: pd.Series, h: int, tau: float):
    """Holt-Winters aditivo sobre log1p(y) con estacionalidad semanal. Cuantil alto por residuos."""
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    z = np.log1p(hist.clip(lower=0))
    fit = ExponentialSmoothing(z, trend="add", damped_trend=True, seasonal="add", seasonal_periods=7).fit()
    zf = fit.forecast(h).values
    sd = np.std(fit.resid)
    from scipy.stats import norm
    pred = np.expm1(zf + 0.5 * sd**2)  # corrección de sesgo log-normal → media
    q = np.expm1(zf + norm.ppf(tau) * sd * np.sqrt(1 + np.arange(h) / 7))
    return np.clip(pred, 0, None), np.clip(q, 0, None)


# ---------------------------------------------------------------------------
# Redes neuronales
# ---------------------------------------------------------------------------
def ventanas(Y: pd.DataFrame, cal: pd.DataFrame, origenes, ventana: int, h: int, escala: pd.Series):
    """Construye tensores para un modelo GLOBAL (todas las rutas comparten pesos).
    X_hist: [N, ventana, 1] demanda escalada; X_fut: [N, h, n_cal] calendario futuro;
    y: [N, h] demanda en conteos; s: [N] escala de la ruta; r: [N] índice de ruta."""
    rutas = list(Y.columns)
    Xh, Xf, ys, ss, rr, oo = [], [], [], [], [], []
    pos = {d: i for i, d in enumerate(Y.index)}
    C = cal.reindex(Y.index).values.astype(np.float32)
    for t in origenes:
        i = pos.get(pd.Timestamp(t))
        if i is None or i + 1 < ventana or i + h >= len(Y):
            continue
        for j, ruta in enumerate(rutas):
            hist = Y[ruta].values[i + 1 - ventana:i + 1]
            fut = Y[ruta].values[i + 1:i + 1 + h]
            if np.isnan(hist).any():
                continue
            Xh.append((hist / escala[ruta])[:, None])
            Xf.append(C[i + 1:i + 1 + h])
            ys.append(fut)
            ss.append(escala[ruta])
            rr.append(j)
            oo.append(t)
    return (np.array(Xh, np.float32), np.array(Xf, np.float32), np.array(ys, np.float32),
            np.array(ss, np.float32), np.array(rr), np.array(oo))


def construir_red(tipo: str, ventana: int, h: int, n_cal: int, n_rutas: int, capas: int, unidades: int,
                  dropout: float, emb_ruta: int = 4):
    """Salida: (log_tasa [B,h], q_alto [B,h]) en escala de la ruta.
    - tipo 'mlp': ventana aplanada + calendario futuro aplanado.
    - tipo 'lstm' / 'gru': codificador recurrente de la ventana; cabeza densa con calendario futuro."""
    import torch
    from torch import nn

    class Red(nn.Module):
        def __init__(self):
            super().__init__()
            self.tipo = tipo
            self.emb = nn.Embedding(n_rutas, emb_ruta)
            if tipo == "mlp":
                d_in = ventana
                bloques, d = [], d_in
                for _ in range(capas):
                    bloques += [nn.Linear(d, unidades), nn.ReLU(), nn.Dropout(dropout)]
                    d = unidades
                self.enc = nn.Sequential(*bloques)
            else:
                rnn = nn.LSTM if tipo == "lstm" else nn.GRU
                self.enc = rnn(1, unidades, num_layers=capas, batch_first=True,
                               dropout=dropout if capas > 1 else 0.0)
            d_cab = unidades + emb_ruta + h * n_cal
            self.cab = nn.Sequential(nn.Linear(d_cab, unidades), nn.ReLU(), nn.Dropout(dropout),
                                     nn.Linear(unidades, 2 * h))
            self.h = h

        def forward(self, xh, xf, r):
            if self.tipo == "mlp":
                z = self.enc(xh.squeeze(-1))
            else:
                out, _ = self.enc(xh)
                z = out[:, -1]
            z = torch.cat([z, self.emb(r), xf.flatten(1)], dim=1)
            o = self.cab(z)
            log_tasa = o[:, :self.h]
            # q_alto = media + margen ≥ 0 → garantiza que el P90 no cruce la media
            q_alto = torch.exp(log_tasa) + nn.functional.softplus(o[:, self.h:])
            return log_tasa, q_alto

    return Red()


def perdida_poisson_cuantil(log_tasa, q_alto, y_esc, tau: float, peso_q: float = 1.0):
    """NLL Poisson (con log-tasa, sin término log y!) + pinball sobre el cuantil alto.
    Todo en escala de la ruta (y / escala) para que rutas grandes no dominen el gradiente."""
    import torch

    nll = (torch.exp(log_tasa) - y_esc * log_tasa).mean()
    u = y_esc - q_alto
    pin = torch.maximum(tau * u, (tau - 1) * u).mean()
    return nll + peso_q * pin


def contar_parametros(modelo) -> int:
    return sum(p.numel() for p in modelo.parameters() if p.requires_grad)


@dataclass
class ResultadoEntrenamiento:
    modelo: object
    hist_train: list
    hist_val: list
    mejor_epoca: int
    segundos: float


def entrenar(modelo, datos_tr, datos_va, tau: float, lr: float, epocas: int, paciencia: int,
             batch: int = 64, optimizador: str = "adam", weight_decay: float = 0.0, seed: int = 42):
    import copy
    import torch

    torch.manual_seed(seed)
    def tens(d):
        xh, xf, y, s, r, _ = d
        return (torch.tensor(xh), torch.tensor(xf), torch.tensor(y / s[:, None]), torch.tensor(r))
    tr, va = tens(datos_tr), tens(datos_va)
    Opt = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW, "rmsprop": torch.optim.RMSprop}[optimizador]
    opt = Opt(modelo.parameters(), lr=lr, weight_decay=weight_decay)
    g = torch.Generator().manual_seed(seed)
    mejor, mejor_ep, estado, ht, hv = math.inf, 0, None, [], []
    t0 = time.time()
    for ep in range(epocas):
        modelo.train()
        perm = torch.randperm(len(tr[0]), generator=g)
        acum = 0.0
        for k in range(0, len(perm), batch):
            b = perm[k:k + batch]
            opt.zero_grad()
            lt, q = modelo(tr[0][b], tr[1][b], tr[3][b])
            loss = perdida_poisson_cuantil(lt, q, tr[2][b], tau)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), 5.0)
            opt.step()
            acum += loss.item() * len(b)
        ht.append(acum / len(perm))
        modelo.eval()
        with torch.no_grad():
            lt, q = modelo(va[0], va[1], va[3])
            lv = perdida_poisson_cuantil(lt, q, va[2], tau).item()
        hv.append(lv)
        if lv < mejor - 1e-5:
            mejor, mejor_ep, estado = lv, ep, copy.deepcopy(modelo.state_dict())
        elif ep - mejor_ep >= paciencia:
            break
    modelo.load_state_dict(estado)
    return ResultadoEntrenamiento(modelo, ht, hv, mejor_ep, time.time() - t0)


def predecir(modelo, datos):
    import torch

    xh, xf, _, s, r, _ = datos
    modelo.eval()
    with torch.no_grad():
        lt, q = modelo(torch.tensor(xh), torch.tensor(xf), torch.tensor(r))
    return np.exp(lt.numpy()) * s[:, None], q.numpy() * s[:, None]


# ---------------------------------------------------------------------------
# Dimensionamiento de flota y conductores
# ---------------------------------------------------------------------------
def despachos_requeridos(demanda, capacidad: float, ocupacion: float):
    """D_t = ⌈ demanda_t / (capacidad · ocupación) ⌉."""
    return np.ceil(np.asarray(demanda, float) / (capacidad * ocupacion))


def vueltas_por_vehiculo(horas_operacion: float, ciclo_horas: float) -> int:
    """Una vuelta = ida + regreso. Vueltas = ⌊ horas_operación / tiempo_ciclo ⌋ (≥ 1)."""
    return max(1, int(horas_operacion // ciclo_horas))


def vehiculos_requeridos(desp_ida, desp_regreso, vueltas: int):
    """Cada vuelta cubre un despacho en cada sentido → el sentido más cargado manda.
    Si no se modela el regreso, pasar desp_regreso = desp_ida (supuesto simétrico)."""
    return np.ceil(np.maximum(desp_ida, desp_regreso) / vueltas)


def conductores_requeridos(vehiculos, horas_operacion: float, horas_max_conductor: float,
                           ausentismo: float, dias_trabajo_semana: int = 6):
    """Turnos por vehículo = ⌈ horas_operación / jornada_máx ⌉; se ajusta por descanso semanal
    (7/días_trabajo) y ausentismo (1/(1−a))."""
    turnos = math.ceil(horas_operacion / horas_max_conductor)
    return np.ceil(np.asarray(vehiculos) * turnos * (7 / dias_trabajo_semana) / (1 - ausentismo))


def costo_operativo(y_real, demanda_plan, capacidad, ocupacion, vueltas):
    """Pasajeros sin atender y vehículos-día sobrantes al dimensionar con `demanda_plan`."""
    desp_plan = despachos_requeridos(demanda_plan, capacidad, ocupacion)
    sillas = desp_plan * capacidad
    sin_atender = np.clip(np.asarray(y_real) - sillas, 0, None)
    veh_plan = np.ceil(desp_plan / vueltas)
    veh_necesarios = np.ceil(despachos_requeridos(y_real, capacidad, ocupacion) / vueltas)
    sobrantes = np.clip(veh_plan - veh_necesarios, 0, None)
    return float(np.nansum(sin_atender)), float(np.nansum(sobrantes))


def ruta_raiz() -> Path:
    return Path(__file__).resolve().parent
