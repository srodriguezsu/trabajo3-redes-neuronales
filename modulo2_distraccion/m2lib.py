"""Funciones del módulo 2: datos, partición sin fuga, modelos, entrenamiento, evaluación y Grad-CAM.

Los notebooks solo llaman a estas funciones; los parámetros salen de config/modulo2.yaml.
"""
from __future__ import annotations

import hashlib
import json
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

AQUI = Path(__file__).resolve().parent
RAW = AQUI / "data" / "raw"
PROC = AQUI / "data" / "processed"
MODELOS = AQUI / "models"
FIGURAS = AQUI / "figures"
RESULTADOS = AQUI / "results"

NOMBRES = {
    "safe_driving": "Conducción segura",
    "turning": "Giro",
    "texting_phone": "Escribir en el teléfono",
    "talking_phone": "Hablar por teléfono",
    "other_activities": "Otras actividades",
}


# ---------------------------------------------------------------- datos

def descargar(cfg: dict) -> Path:
    """Baja el zip de Mendeley (2,5 GB) si no está y verifica su sha256."""
    import requests

    RAW.mkdir(parents=True, exist_ok=True)
    zip_ = RAW / "driver_behavior.zip"
    if not zip_.exists():
        with requests.get(cfg["datos"]["url_zip"], stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(zip_, "wb") as f:
                for trozo in r.iter_content(1 << 20):
                    f.write(trozo)
    h = hashlib.sha256()
    with open(zip_, "rb") as f:
        for trozo in iter(lambda: f.read(1 << 20), b""):
            h.update(trozo)
    if h.hexdigest() != cfg["datos"]["sha256_zip"]:
        raise ValueError("El zip descargado no coincide con el sha256 de config/modulo2.yaml")
    carpeta = RAW / "imagenes"
    if not carpeta.exists():
        with zipfile.ZipFile(zip_) as z:
            z.extractall(RAW)
        (RAW / "Multi-Class Driver Behavior Image Dataset").rename(carpeta)
    return carpeta


def dhash(im: Image.Image, lado: int = 8) -> int:
    """Hash perceptual por diferencias (64 bits): compara cada píxel con su vecino derecho."""
    g = np.asarray(im.convert("L").resize((lado + 1, lado), Image.LANCZOS), dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def encajar(im: Image.Image, ancho: int, alto: int) -> Image.Image:
    """Reduce la imagen para que quepa en ancho x alto sin deformarla y rellena con gris."""
    im = im.copy()
    im.thumbnail((ancho, alto), Image.LANCZOS)
    lienzo = Image.new("RGB", (ancho, alto), (114, 114, 114))
    lienzo.paste(im, ((ancho - im.width) // 2, (alto - im.height) // 2))
    return lienzo


def preparar(cfg: dict, carpeta_raw: Path | None = None) -> pd.DataFrame:
    """Reduce todas las imágenes a ancho x alto (JPG) y escribe metadatos.csv.

    Guarda por imagen: clase, tamaño original, sha1 de los bytes (duplicados exactos)
    y el dHash de la imagen original (casi duplicados).
    """
    carpeta_raw = carpeta_raw or RAW / "imagenes"
    d = cfg["datos"]
    salida = PROC / "imagenes"
    filas = []
    for clase in d["clases"]:
        (salida / clase).mkdir(parents=True, exist_ok=True)
        for i, ruta in enumerate(sorted((carpeta_raw / clase).iterdir())):
            datos = ruta.read_bytes()
            try:
                im = Image.open(ruta)
                im.load()
            except Exception:  # 10 archivos del zip publicado vienen llenos de ceros
                filas.append({"archivo": None, "original": ruta.name, "clase": clase,
                              "bytes": len(datos), "sha1": hashlib.sha1(datos).hexdigest(),
                              "corrupta": True})
                continue
            ancho0, alto0 = im.size
            exif_rotada = ImageOps.exif_transpose(im).size != im.size
            im = ImageOps.exif_transpose(im).convert("RGB")
            nombre = f"{clase}_{i:04d}.jpg"
            encajar(im, d["ancho"], d["alto"]).save(salida / clase / nombre, quality=d["calidad_jpg"])
            filas.append({
                "archivo": f"{clase}/{nombre}", "original": ruta.name, "clase": clase,
                "ancho_original": ancho0, "alto_original": alto0, "rotada_exif": exif_rotada,
                "bytes": len(datos), "sha1": hashlib.sha1(datos).hexdigest(),
                "dhash": f"{dhash(im):016x}", "corrupta": False,
            })
    meta = pd.DataFrame(filas)
    meta["corrupta"] = meta["corrupta"].astype(bool)
    meta.to_csv(PROC / "metadatos.csv", index=False)
    return meta


def cargar_metadatos() -> pd.DataFrame:
    return pd.read_csv(PROC / "metadatos.csv", dtype={"dhash": str})


def validas() -> pd.DataFrame:
    """Metadatos sin las imágenes corruptas, con la partición si ya existe."""
    meta = cargar_metadatos()
    meta = meta[~meta["corrupta"]].reset_index(drop=True)
    ruta = PROC / "particion.csv"
    if ruta.exists():
        meta = meta.merge(pd.read_csv(ruta), on="archivo", how="left")
    return meta


# ---------------------------------------------------------------- torch: datos y modelos

def dispositivo():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


MEDIA = (0.485, 0.456, 0.406)   # normalización de ImageNet (la usan las redes preentrenadas)
DESV = (0.229, 0.224, 0.225)


AUMENTOS = {  # (grados, traslación, escala mín, escala máx, brillo, contraste, saturación, gris, borrado)
    "ninguno": None,
    "suave": (5, 0.03, 0.95, 1.05, 0.15, 0.15, 0.0, 0.0, 0.0),
    "medio": (10, 0.06, 0.90, 1.10, 0.30, 0.30, 0.20, 0.0, 0.0),
    "fuerte": (15, 0.10, 0.85, 1.15, 0.40, 0.40, 0.30, 0.10, 0.25),
}


def a_tensor(im: Image.Image, ancho=224, alto=168):
    """Imagen PIL -> tensor normalizado (1, 3, alto, ancho) para predecir una imagen suelta."""
    import torch

    im = encajar(ImageOps.exif_transpose(im).convert("RGB"), ancho, alto)
    x = torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1).float().div(255)
    return normalizar(x.unsqueeze(0))


def normalizar(x):
    import torch

    m = torch.tensor(MEDIA, device=x.device).view(1, 3, 1, 1)
    d = torch.tensor(DESV, device=x.device).view(1, 3, 1, 1)
    return (x - m) / d


def aumentar(x, nivel: str, gen):
    """Aumento de datos por lotes en la GPU; cada imagen recibe sus propios parámetros.

    Transformación afín (rotación, traslación, escala) con relleno gris, brillo, contraste,
    saturación y, en el nivel fuerte, gris y borrado aleatorio. Sin volteo horizontal: el
    conductor va siempre del mismo lado y voltear crearía escenas que la cámara nunca vería.
    """
    import math

    import torch
    import torch.nn.functional as F

    par = AUMENTOS[nivel]
    if par is None:
        return x
    grados, tras, s0, s1, br, co, sa, gris, borrado = par
    n, dev = len(x), x.device
    u = lambda *forma: torch.rand(*forma, generator=gen, device=dev)
    ang = (u(n) * 2 - 1) * math.radians(grados)
    esc = s0 + u(n) * (s1 - s0)
    tx, ty = (u(n) * 2 - 1) * tras * 2, (u(n) * 2 - 1) * tras * 2
    cos, sen = torch.cos(ang) / esc, torch.sin(ang) / esc
    theta = torch.stack([torch.stack([cos, -sen, tx], 1), torch.stack([sen, cos, ty], 1)], 1)
    malla = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x - 0.45, malla, align_corners=False, padding_mode="zeros") + 0.45
    x = x * (1 + (u(n, 1, 1, 1) * 2 - 1) * br)
    media = x.mean(dim=(1, 2, 3), keepdim=True)
    x = (x - media) * (1 + (u(n, 1, 1, 1) * 2 - 1) * co) + media
    g = (0.299 * x[:, :1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:])
    if sa:
        x = g + (x - g) * (1 + (u(n, 1, 1, 1) * 2 - 1) * sa)
    if gris:
        x = torch.where(u(n, 1, 1, 1) < gris, g.expand_as(x), x)
    if borrado:
        h, w = x.shape[-2:]
        lado = (u(n) * 0.2 + 0.1)
        cy, cx = u(n) * h, u(n) * w
        yy = torch.arange(h, device=dev).view(1, h, 1); xx = torch.arange(w, device=dev).view(1, 1, w)
        caja = ((yy - cy.view(n, 1, 1)).abs() < lado.view(n, 1, 1) * h / 2) &                ((xx - cx.view(n, 1, 1)).abs() < lado.view(n, 1, 1) * w / 2)
        caja = caja.unsqueeze(1) & (u(n, 1, 1, 1) < borrado)
        x = torch.where(caja, torch.full_like(x, 0.45), x)
    return x.clamp(0, 1)


class Imagenes:
    """Las imágenes procesadas como un tensor uint8 en la GPU (7 276 x 3 x 168 x 224 = 0,8 GB)."""

    def __init__(self, meta: pd.DataFrame, clases: list[str], nivel: str = "ninguno"):
        import torch

        arr = np.stack([np.asarray(Image.open(PROC / "imagenes" / a).convert("RGB")) for a in meta["archivo"]])
        self.x = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous().to(dispositivo())
        self.y = torch.tensor([clases.index(c) for c in meta["clase"]], device=dispositivo())
        self.nivel = nivel

    def __len__(self):
        return len(self.y)


class cargador:
    """Itera lotes (x normalizado, y) sobre un Imagenes; con mezclar=True cambia el orden en cada
    época de forma reproducible a partir de la semilla."""

    def __init__(self, ds, tamano_lote, mezclar, semilla=0, num_workers=0):
        import torch

        self.ds, self.b, self.mezclar = ds, tamano_lote, mezclar
        self.gen_cpu = torch.Generator().manual_seed(semilla)
        self.gen = torch.Generator(device=ds.x.device).manual_seed(semilla)

    def __len__(self):
        return (len(self.ds) + self.b - 1) // self.b

    def __iter__(self):
        import torch

        n = len(self.ds)
        orden = torch.randperm(n, generator=self.gen_cpu) if self.mezclar else torch.arange(n)
        orden = orden.to(self.ds.x.device)
        for i in range(0, n, self.b):
            idx = orden[i:i + self.b]
            x = self.ds.x[idx].float().div_(255)
            if self.mezclar:
                x = aumentar(x, self.ds.nivel, self.gen)
            yield normalizar(x), self.ds.y[idx]


def red_desde_cero(n_clases: int, filtros=(32, 64, 128, 256), dropout=0.3):
    """CNN propia: bloques [conv3x3-BN-ReLU]x2 + maxpool, promedio global y una capa lineal."""
    import torch.nn as nn

    capas, entrada = [], 3
    for f in filtros:
        capas += [nn.Conv2d(entrada, f, 3, padding=1, bias=False), nn.BatchNorm2d(f), nn.ReLU(inplace=True),
                  nn.Conv2d(f, f, 3, padding=1, bias=False), nn.BatchNorm2d(f), nn.ReLU(inplace=True),
                  nn.MaxPool2d(2)]
        entrada = f
    return nn.Sequential(*capas, nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                         nn.Dropout(dropout), nn.Linear(entrada, n_clases))


def red_preentrenada(nombre: str, n_clases: int, dropout=0.2, descongelar="todo"):
    """Red de torchvision con pesos de ImageNet y una cabeza nueva de n_clases."""
    import torch.nn as nn
    from torchvision import models as M

    if nombre == "resnet18":
        red = M.resnet18(weights=M.ResNet18_Weights.IMAGENET1K_V1)
        red.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(red.fc.in_features, n_clases))
        cabeza, ultimo = red.fc, red.layer4
    elif nombre == "mobilenet_v3_small":
        red = M.mobilenet_v3_small(weights=M.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        red.classifier[2] = nn.Dropout(dropout)
        red.classifier[3] = nn.Linear(red.classifier[3].in_features, n_clases)
        cabeza, ultimo = red.classifier, red.features[-3:]
    elif nombre == "efficientnet_b0":
        red = M.efficientnet_b0(weights=M.EfficientNet_B0_Weights.IMAGENET1K_V1)
        red.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(red.classifier[1].in_features, n_clases))
        cabeza, ultimo = red.classifier, red.features[-2:]
    else:
        raise ValueError(nombre)
    if descongelar != "todo":
        for p in red.parameters():
            p.requires_grad = False
        for p in cabeza.parameters():
            p.requires_grad = True
        if descongelar == "ultimo_bloque":
            for p in ultimo.parameters():
                p.requires_grad = True
    return red


def capa_gradcam(red, tipo: str):
    """Última capa convolucional, sobre la que se calcula Grad-CAM."""
    if tipo == "cnn":
        return [m for m in red if m.__class__.__name__ == "Conv2d"][-1]
    if tipo == "resnet18":
        return red.layer4[-1]
    return red.features[-1]


def parametros_entrenables(red) -> int:
    return sum(p.numel() for p in red.parameters() if p.requires_grad)


_EXTRACTORES = {}


def _extractor(nombre: str):
    import torch.nn as nn

    if nombre not in _EXTRACTORES:
        red = red_preentrenada(nombre, 1000)
        if nombre == "resnet18":
            red.fc = nn.Identity()
        else:
            red.classifier = nn.Identity()
        _EXTRACTORES[nombre] = red.eval().to(dispositivo())
    return _EXTRACTORES[nombre]


def caracteristicas(meta: pd.DataFrame, nombre="resnet18", tamano_lote=128) -> np.ndarray:
    """Vector de la penúltima capa de la red preentrenada congelada para cada imagen."""
    import torch

    ds = Imagenes(meta, sorted(meta["clase"].unique()))
    red, dev, salida = _extractor(nombre), dispositivo(), []
    with torch.no_grad():
        for x, _ in cargador(ds, tamano_lote, False):
            salida.append(red(x.to(dev)).float().cpu().numpy())
    return np.concatenate(salida)


# ---------------------------------------------------------------- partición sin fuga

def _bits_hash(meta: pd.DataFrame) -> np.ndarray:
    h = np.array([int(x, 16) for x in meta["dhash"]], dtype=">u8")
    return np.unpackbits(h.view(np.uint8).reshape(len(meta), 8), axis=1)


def grupos_casi_iguales(meta: pd.DataFrame, feats: np.ndarray, umbral_sim: float, umbral_hash: int,
                        iniciales=None):
    """Une en un mismo grupo las imágenes casi iguales (componentes conexas de un grafo).

    Hay arista entre dos imágenes si su dHash difiere en <= umbral_hash bits o si el coseno entre
    sus características preentrenadas es >= umbral_sim.
    """
    n = len(meta)
    padre = np.arange(n)
    if iniciales is not None:  # imágenes que ya pertenecen a un mismo grupo (p. ej. mismo conductor)
        primero = {}
        for i, g in enumerate(iniciales):
            if g is not None and g == g:
                padre[i] = primero.setdefault(g, i)

    def raiz(i):
        while padre[i] != i:
            padre[i] = padre[padre[i]]
            i = padre[i]
        return i

    bits = _bits_hash(meta).astype(np.int8)
    f = feats / np.linalg.norm(feats, axis=1, keepdims=True)
    pares = {"hash": 0, "coseno": 0}
    for i in range(0, n, 256):
        fin = min(i + 256, n)
        ham = (bits[i:fin, None, :] != bits[None, :, :]).sum(-1)
        sim = f[i:fin] @ f.T
        for tipo, mascara in (("hash", ham <= umbral_hash), ("coseno", sim >= umbral_sim)):
            for a, b in zip(*np.nonzero(mascara)):
                a += i
                if a < b:
                    pares[tipo] += 1
                    ra, rb = raiz(a), raiz(b)
                    if ra != rb:
                        padre[rb] = ra
    ids = pd.factorize(pd.Series([raiz(i) for i in range(n)]))[0]
    return ids, pares


CLASES_STATEFARM = {
    "c0": "conducción segura", "c1": "escribir (mano derecha)", "c2": "hablar (mano derecha)",
    "c3": "escribir (mano izquierda)", "c4": "hablar (mano izquierda)", "c5": "manipular el radio",
    "c6": "beber", "c7": "alcanzar algo atrás", "c8": "peinarse o maquillarse",
    "c9": "hablar con el pasajero",
}


def origen_statefarm(meta: pd.DataFrame) -> pd.DataFrame:
    """Cruza los nombres de archivo con driver_imgs_list.csv de State Farm (Kaggle, 2016).

    Agrega 'conductor' (p002…p081) y 'clase_statefarm' (c0…c9) a las imágenes que vienen de ahí.
    """
    sf = pd.read_csv(AQUI / "data" / "external" / "statefarm_driver_imgs_list.csv")
    sf = sf.rename(columns={"img": "original", "subject": "conductor", "classname": "clase_statefarm"})
    meta = meta.drop(columns=[c for c in ("conductor", "clase_statefarm") if c in meta])
    return meta.merge(sf, on="original", how="left")


def grupos_particion(meta: pd.DataFrame, feats: np.ndarray, umbral_sim: float, umbral_hash: int):
    """Grupo de cada imagen para partir sin fuga.

    Parte de los conductores de State Farm (la misma persona nunca cae en dos particiones) y
    además une cualquier par de imágenes casi iguales (dHash o coseno), sean o no de State Farm.
    """
    return grupos_casi_iguales(meta, feats, umbral_sim, umbral_hash,
                               iniciales=meta["conductor"].tolist())


def particionar(meta: pd.DataFrame, grupos: np.ndarray, frac_val: float, frac_prueba: float, semilla=42):
    """Partición estratificada por clase y agrupada: cada grupo completo cae en una sola partición."""
    from sklearn.model_selection import StratifiedGroupKFold

    y = meta["clase"].values
    resto, prueba = next(StratifiedGroupKFold(round(1 / frac_prueba), shuffle=True,
                                              random_state=semilla).split(meta, y, grupos))
    k2 = round((1 - frac_prueba) / frac_val)
    ent, val = next(StratifiedGroupKFold(k2, shuffle=True, random_state=semilla + 1)
                    .split(resto, y[resto], grupos[resto]))
    part = np.empty(len(meta), dtype=object)
    part[resto[ent]], part[resto[val]], part[prueba] = "entrenamiento", "validacion", "prueba"
    return part


def verificar_fuga(meta: pd.DataFrame, feats: np.ndarray, umbral_sim: float, umbral_hash: int) -> pd.DataFrame:
    """Cuenta los pares casi iguales que cruzan particiones (debe dar 0) y qué tan parecido es el
    vecino más cercano entre particiones."""
    f = feats / np.linalg.norm(feats, axis=1, keepdims=True)
    bits = _bits_hash(meta).astype(np.int8)
    filas = []
    for a, b in [("entrenamiento", "validacion"), ("entrenamiento", "prueba"), ("validacion", "prueba")]:
        ia, ib = np.where(meta["particion"] == a)[0], np.where(meta["particion"] == b)[0]
        sim = f[ib] @ f[ia].T
        ham = (bits[ib, None, :] != bits[None, ia, :]).sum(-1)
        filas.append({"particiones": f"{a} / {b}", "pares_coseno": int((sim >= umbral_sim).sum()),
                      "pares_hash": int((ham <= umbral_hash).sum()),
                      "coseno_maximo": round(float(sim.max()), 4), "hamming_minimo": int(ham.min()),
                      "coseno_vecino_mediana": round(float(np.median(sim.max(1))), 4)})
    return pd.DataFrame(filas)


def guardar_particion(meta: pd.DataFrame) -> None:
    cols = ["archivo", "conductor", "clase_statefarm", "grupo", "particion"]
    meta[cols].to_csv(PROC / "particion.csv", index=False)


def cargar_caracteristicas(meta: pd.DataFrame | None = None, nombre="resnet18") -> np.ndarray:
    """Características guardadas (mismo orden que validas())."""
    return np.load(PROC / f"caracteristicas_{nombre}.npy").astype(np.float32)


# ---------------------------------------------------------------- entrenamiento

def optimizador(params, nombre: str, lr: float, weight_decay: float):
    import torch

    if nombre == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if nombre == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, nesterov=True, weight_decay=weight_decay)
    raise ValueError(nombre)


def construir(tipo: str, hp: dict, n_clases: int):
    """tipo: 'cnn' (desde cero) o el nombre de una red preentrenada."""
    if tipo == "cnn":
        return red_desde_cero(n_clases, hp.get("filtros", (32, 64, 128, 256)), hp["dropout"])
    return red_preentrenada(tipo, n_clases, hp["dropout"], hp.get("descongelar", "todo"))


def entrenar(red, ds_ent, ds_val, hp: dict, epocas: int, paciencia: int, tamano_lote: int,
             semilla: int, num_workers=0, registro=None):
    """Entrena con entropía cruzada, AMP en GPU, coseno en la tasa y parada temprana por pérdida
    de validación. Devuelve el historial por época y deja en la red los pesos de la mejor época."""
    import copy

    import torch
    import torch.nn as nn

    dev = dispositivo()
    red = red.to(dev)
    opt = optimizador([p for p in red.parameters() if p.requires_grad], hp["optimizador"], hp["lr"],
                      hp["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epocas)
    escala = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")
    perdida = nn.CrossEntropyLoss()
    dl_ent = cargador(ds_ent, tamano_lote, True, semilla, num_workers)
    dl_val = cargador(ds_val, 256, False, 0, num_workers)
    hist, mejor, mejor_estado, sin_mejora = [], float("inf"), None, 0
    for ep in range(1, epocas + 1):
        t0 = time.time()
        red.train()
        suma, n, aciertos = 0.0, 0, 0
        for x, y in dl_ent:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(dev.type, enabled=dev.type == "cuda"):
                out = red(x)
                l = perdida(out, y)
            escala.scale(l).backward()
            escala.step(opt)
            escala.update()
            suma += l.item() * len(y); n += len(y); aciertos += (out.argmax(1) == y).sum().item()
        sched.step()
        p_val, y_val = predecir(red, dl_val)
        l_val = float(-np.log(np.clip(p_val[np.arange(len(y_val)), y_val], 1e-7, 1)).mean())
        fila = {"epoca": ep, "perdida_ent": suma / n, "exactitud_ent": aciertos / n,
                "perdida_val": l_val, "exactitud_val": float((p_val.argmax(1) == y_val).mean()),
                "segundos": time.time() - t0}
        hist.append(fila)
        if registro:
            registro(fila)
        if l_val < mejor - 1e-4:
            mejor, mejor_estado, sin_mejora = l_val, copy.deepcopy(red.state_dict()), 0
        else:
            sin_mejora += 1
            if sin_mejora >= paciencia:
                break
    red.load_state_dict(mejor_estado)
    h = pd.DataFrame(hist)
    h["mejor"] = h["perdida_val"] == h["perdida_val"].min()
    return h


def predecir(red, dl) -> tuple[np.ndarray, np.ndarray]:
    """Probabilidades softmax y etiquetas reales."""
    import torch

    dev = dispositivo()
    red.eval()
    ps, ys = [], []
    with torch.no_grad():
        for x, y in dl:
            with torch.autocast(dev.type, enabled=dev.type == "cuda"):
                out = red(x.to(dev))
            ps.append(torch.softmax(out.float(), 1).cpu().numpy()); ys.append(y.cpu().numpy())
    return np.concatenate(ps), np.concatenate(ys)


def metricas(y: np.ndarray, p: np.ndarray, clases: list[str]) -> dict:
    from sklearn.metrics import accuracy_score, f1_score, log_loss, precision_score, recall_score

    pred = p.argmax(1)
    m = {"exactitud": accuracy_score(y, pred), "f1_macro": f1_score(y, pred, average="macro"),
         "perdida": log_loss(y, p, labels=range(len(clases)))}
    for i, c in enumerate(clases):
        m[f"precision_{c}"] = precision_score(y, pred, labels=[i], average="macro", zero_division=0)
        m[f"sensibilidad_{c}"] = recall_score(y, pred, labels=[i], average="macro", zero_division=0)
    return m


# ---------------------------------------------------------------- Grad-CAM

def gradcam(red, capa, x, clase: int | None = None) -> tuple[np.ndarray, int]:
    """Grad-CAM (Selvaraju et al., 2017): pondera los mapas de activación de la capa por el
    promedio espacial del gradiente de la puntuación de la clase, ReLU y normaliza a [0, 1]."""
    import torch
    import torch.nn.functional as F

    act, grad = {}, {}
    h1 = capa.register_forward_hook(lambda m, i, o: act.__setitem__("a", o))
    h2 = capa.register_full_backward_hook(lambda m, gi, go: grad.__setitem__("g", go[0]))
    red.eval()
    x = x.to(dispositivo()).unsqueeze(0).requires_grad_(True)
    out = red(x)
    clase = int(out.argmax(1)) if clase is None else clase
    red.zero_grad()
    out[0, clase].backward()
    h1.remove(); h2.remove()
    pesos = grad["g"].mean(dim=(2, 3), keepdim=True)
    mapa = F.relu((pesos * act["a"]).sum(1, keepdim=True))
    mapa = F.interpolate(mapa, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    mapa = mapa.detach().cpu().numpy()
    return (mapa - mapa.min()) / (mapa.max() - mapa.min() + 1e-8), clase


# ---------------------------------------------------------------- alertas

def prob_alerta(p: np.ndarray, clases: list[str], con_alerta: list[str]) -> np.ndarray:
    idx = [clases.index(c) for c in con_alerta]
    return p[:, idx].sum(1)


def tabla_umbrales(y: np.ndarray, p: np.ndarray, clases, con_alerta, umbrales, costo_fn=5.0, costo_fp=1.0):
    """Falsas alarmas y distracciones no detectadas por cada 1 000 imágenes según el umbral."""
    real = np.isin(np.array(clases)[y], con_alerta)
    pa = prob_alerta(p, clases, con_alerta)
    filas = []
    for u in umbrales:
        alerta = pa >= u
        fp = int((alerta & ~real).sum()); fn = int((~alerta & real).sum())
        filas.append({"umbral": u, "falsas_alarmas_por_1000": 1000 * fp / len(y),
                      "no_detectadas_por_1000": 1000 * fn / len(y),
                      "sensibilidad_alerta": float((alerta & real).sum() / real.sum()),
                      "precision_alerta": float((alerta & real).sum() / max(alerta.sum(), 1)),
                      "costo_por_1000": 1000 * (costo_fn * fn + costo_fp * fp) / len(y)})
    return pd.DataFrame(filas)
