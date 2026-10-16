"""Lógica del comparador de snatch (sin Streamlit ni YOLO, fácil de testear)."""
import cv2
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

CONF = 0.3
ANGULOS = ["Genou", "Hanche", "Tronc", "Coude"]

# Índices COCO de cada lado
LADOS = {
    "izquierdo": dict(sho=5, elb=7, wri=9, hip=11, kne=13, ank=15),
    "derecho": dict(sho=6, elb=8, wri=10, hip=12, kne=14, ank=16),
}
# Segmentos del cuerpo implicados en cada ángulo (para pintarlos en rojo)
SEGMENTOS = {
    "Genou": [("hip", "kne"), ("kne", "ank")],
    "Hanche": [("sho", "hip"), ("hip", "kne")],
    "Tronc": [("sho", "hip")],
    "Coude": [("sho", "elb"), ("elb", "wri")],
}
LIMBS = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (11, 12), (5, 11), (6, 12),
         (11, 13), (13, 15), (12, 14), (14, 16)]


# ----------------------------------------------------------------- métricas
def angulo(a, b, c):
    """Ángulo en b formado por a-b-c (arrays Nx2). Devuelve grados."""
    ba, bc = a - b, c - b
    cos = (ba * bc).sum(-1) / (np.linalg.norm(ba, axis=-1) * np.linalg.norm(bc, axis=-1) + 1e-6)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def lado_visible(kps):
    """Lado del cuerpo con mejor confianza media (el más cercano a la cámara)."""
    conf = np.nan_to_num(kps[:, :, 2])
    izq = conf[:, [5, 7, 9, 11, 13, 15]].mean()
    der = conf[:, [6, 8, 10, 12, 14, 16]].mean()
    return "izquierdo" if izq >= der else "derecho"


def cobertura(kps, lado):
    """Fracción de frames con hombro, cadera, rodilla y tobillo detectados."""
    s = LADOS[lado]
    ok = np.ones(len(kps), bool)
    for n in ("sho", "hip", "kne", "ank"):
        ok &= np.nan_to_num(kps[:, s[n], 2]) > CONF
    return float(ok.mean())


def ratio_lateral(kps, lado):
    """Ancho de hombros / longitud del torso. ~0.1-0.3 = lateral; >0.45 = de frente o diagonal."""
    s = LADOS[lado]
    ok = (np.nan_to_num(kps[:, 5, 2]) > CONF) & (np.nan_to_num(kps[:, 6, 2]) > CONF) \
        & (np.nan_to_num(kps[:, s["hip"], 2]) > CONF)
    if ok.sum() < 5:
        return None
    ancho = np.abs(kps[ok, 5, 0] - kps[ok, 6, 0])
    torso = np.linalg.norm(kps[ok, s["sho"], :2] - kps[ok, s["hip"], :2], axis=1)
    r = ancho / np.maximum(torso, 10)
    return float(np.median(r))


def _limpiar(x):
    x = np.asarray(x, float)
    idx = np.arange(len(x))
    ok = ~np.isnan(x)
    if ok.sum() == 0:
        return np.zeros_like(x)
    if ok.sum() < len(x):
        x = np.interp(idx, idx[ok], x[ok])
    if len(x) >= 9:
        x = savgol_filter(x, 9, 2)
    return x


def calcular_metricas(kps, lado):
    """DataFrame con 4 ángulos (grados) y 2 alturas relativas (independientes de zoom/cámara)."""
    s = LADOS[lado]

    def P(n):
        k = kps[:, s[n], :]
        return np.where((np.nan_to_num(k[:, 2]) > CONF)[:, None], k[:, :2], np.nan)

    sho, elb, wri, hip, kne, ank = (P(n) for n in ("sho", "elb", "wri", "hip", "kne", "ank"))
    v = sho - hip
    torso = np.linalg.norm(v, axis=1)
    torso = np.where(torso < 10, np.nan, torso)

    m = {
        "Genou": angulo(hip, kne, ank),
        "Hanche": angulo(sho, hip, kne),
        "Coude": angulo(sho, elb, wri),
        "Tronc": np.degrees(np.arctan2(np.abs(v[:, 0]), -v[:, 1])),  # 0° = vertical
        "_hip_rel": (ank[:, 1] - hip[:, 1]) / torso,   # altura de cadera sobre tobillo, en torsos
        "_wri_rel": (ank[:, 1] - wri[:, 1]) / torso,   # altura de muñeca (≈ barra), en torsos
    }
    return pd.DataFrame({k: _limpiar(x) for k, x in m.items()})


def detectar_recepcion(df):
    """Frame de la recepción: punto más bajo de cadera mientras la barra está arriba
    (muñeca por encima del hombro: el hombro está ~1 torso sobre la cadera)."""
    wri, hip = df["_wri_rel"].values, df["_hip_rel"].values
    idx = np.where(wri - hip > 1.0)[0]
    if len(idx) == 0:
        return int(np.argmin(hip))
    return int(idx[np.argmin(hip[idx])])


# --------------------------------------------------------------------- DTW
def _features(df):
    return np.column_stack([df["Genou"] / 90, df["Hanche"] / 90, df["Tronc"] / 90,
                            df["Coude"] / 90, df["_hip_rel"] / 1.5, df["_wri_rel"] / 1.5])


def alinear_dtw(df_ref, df_al):
    """DTW entre modelo y alumno. Devuelve lista de pares (i_ref, j_alumno)."""
    a, b = _features(df_ref), _features(df_al)
    n, m = len(a), len(b)
    cost = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0
    for i in range(1, n + 1):
        Di, Dp, ci = D[i], D[i - 1], cost[i - 1]
        for j in range(1, m + 1):
            Di[j] = ci[j - 1] + min(Dp[j], Di[j - 1], Dp[j - 1])
    i, j, path = n, m, []
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        k = int(np.argmin([D[i - 1, j - 1], D[i - 1, j], D[i, j - 1]]))
        if k == 0:
            i, j = i - 1, j - 1
        elif k == 1:
            i -= 1
        else:
            j -= 1
    return path[::-1]


def mapa_alumno_a_ref(path, m):
    """Para cada frame del alumno, el frame del modelo equivalente."""
    buckets = {}
    for i, j in path:
        buckets.setdefault(j, []).append(i)
    res = np.array([int(np.median(buckets[j])) for j in range(m)])
    return np.maximum.accumulate(res)


def valor_en(df, col, idx, r=2):
    return float(df[col].iloc[max(0, idx - r): idx + r + 1].mean())


# ---------------------------------------------------------------- feedback
def pista(ang, momento, d):
    """Interprétation indicative d'un écart (élève - modèle)."""
    mas = d > 0
    t = {
        ("Genou", "Réception"): ("Squat moins profond que le modèle (genou plus tendu).",
                                 "Genou plus fléchi que le modèle : squat plus profond ou bassin plus en arrière."),
        ("Hanche", "Réception"): ("Hanche plus ouverte : buste plus droit ou bassin plus haut à la réception.",
                                  "Hanche plus fermée : buste plus penché sur les cuisses à la réception."),
        ("Tronc", "Réception"): ("Buste plus penché vers l'avant que le modèle à la réception.",
                                 "Buste plus vertical que le modèle à la réception."),
        ("Coude", "Réception"): ("Coudes plus tendus que le modèle.",
                                 "Coudes moins tendus : barre peut-être non verrouillée à la réception."),
        ("Genou", "Position de départ"): ("Départ avec les jambes plus tendues (bassin plus haut) que le modèle.",
                                          "Départ avec plus de flexion des genoux (bassin plus bas) que le modèle."),
        ("Hanche", "Position de départ"): ("Hanche plus ouverte au départ (buste plus droit).",
                                           "Hanche plus fermée au départ (buste plus penché)."),
        ("Tronc", "Position de départ"): ("Buste plus penché vers l'avant au départ.",
                                          "Buste plus vertical au départ."),
        ("Genou", "Fin"): ("Genoux plus tendus que le modèle en fin de mouvement.",
                           "Genoux plus fléchis (moins verrouillés) que le modèle en fin de mouvement."),
        ("Coude", "Fin"): ("Coudes plus tendus que le modèle.",
                           "Coudes moins verrouillés que le modèle en fin de mouvement."),
    }
    if (ang, momento) in t:
        return t[(ang, momento)][0 if mas else 1]
    return f"{ang} {'plus grand' if mas else 'plus petit'} que le modèle."


# ------------------------------------------------------------------ vídeo
def _dibujar(img, k, lado, rojos, grosor):
    s = LADOS[lado]
    ok = lambda i: np.nan_to_num(k[i, 2]) > CONF and not np.isnan(k[i, 0])
    pt = lambda i: (int(k[i, 0]), int(k[i, 1]))
    for a, b in LIMBS:
        if ok(a) and ok(b):
            cv2.line(img, pt(a), pt(b), (0, 220, 0), grosor)
    for ang in rojos:
        for na, nb in SEGMENTOS[ang]:
            a, b = s[na], s[nb]
            if ok(a) and ok(b):
                cv2.line(img, pt(a), pt(b), (0, 0, 255), grosor * 2)
    for i in range(5, 17):
        if ok(i):
            cv2.circle(img, pt(i), grosor + 1, (255, 255, 255), -1)


def render_comparacion(path_ref, kps_ref, df_ref, lado_ref, seg_ref,
                       path_al, kps_al, df_al, lado_al, seg_al,
                       mapa, umbral, salida, alto=480, progreso=None):
    """Vídeo lado a lado (modelo | alumno) sincronizado, con segmentos fuera de umbral en rojo.
    seg_* = (frame_inicio, frame_fin) absolutos; mapa = frame modelo (relativo) por frame alumno (relativo)."""
    cap_r, cap_a = cv2.VideoCapture(path_ref), cv2.VideoCapture(path_al)
    cap_r.set(cv2.CAP_PROP_POS_FRAMES, seg_ref[0])
    cap_a.set(cv2.CAP_PROP_POS_FRAMES, seg_al[0])
    fps = cap_a.get(cv2.CAP_PROP_FPS) or 25
    writer, i_leido, frame_r = None, -1, None
    total = seg_al[1] - seg_al[0] + 1

    for j in range(total):
        ok_a, fa = cap_a.read()
        if not ok_a:
            break
        i = int(mapa[j])
        while i_leido < i:
            ok_r, fr = cap_r.read()
            if not ok_r:
                break
            frame_r, i_leido = fr, i_leido + 1
        if frame_r is None:
            break
        ir, ia = seg_ref[0] + i, seg_al[0] + j

        rojos, txt = [], []
        for ang in ANGULOS:
            d = df_al[ang].iloc[ia] - df_ref[ang].iloc[ir]
            if abs(d) > umbral:
                rojos.append(ang)
                txt.append(f"{ang} {d:+.0f} deg")

        paneles = []
        for img, k, lado, etiqueta in ((frame_r.copy(), kps_ref[ir], lado_ref, "MODELE"),
                                       (fa.copy(), kps_al[ia], lado_al, "ELEVE")):
            h, w = img.shape[:2]
            _dibujar(img, k, lado, rojos, max(2, h // 220))
            img = cv2.resize(img, (int(w * alto / h), alto))
            cv2.rectangle(img, (0, 0), (190, 34), (0, 0, 0), -1)
            cv2.putText(img, etiqueta, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            if etiqueta == "ELEVE":
                for n, t in enumerate(txt):
                    cv2.putText(img, t, (10, alto - 15 - 28 * n), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            paneles.append(img)

        frame = np.hstack(paneles)
        if writer is None:
            writer = cv2.VideoWriter(salida, cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                     (frame.shape[1], frame.shape[0]))
        writer.write(frame)
        if progreso:
            progreso((j + 1) / total)

    cap_r.release(), cap_a.release()
    if writer:
        writer.release()
