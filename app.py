import os
import shutil
import subprocess
import tempfile

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from ultralytics import YOLO

import snatch_utils as su

REF_PATH = "referencia_snatch.mp4"  # súbelo al repo con este nombre

st.set_page_config(page_title="Comparador de Snatch", page_icon="🏋️", layout="wide")
st.title("🏋️ Comparador de Squat Snatch")
st.write("Compara la técnica de un alumno con un vídeo modelo: ángulos articulares, fases y vídeo sincronizado.")

with st.sidebar:
    st.header("Ajustes")
    modelo_nombre = st.selectbox("Modelo de pose", ["yolov8m-pose.pt", "yolov8l-pose.pt", "yolov8s-pose.pt"])
    imgsz = st.select_slider("Resolución de análisis", [320, 480, 640, 960], value=640)
    umbral = st.slider("Umbral de diferencia (grados)", 5, 30, 15)
    st.caption("Más resolución y modelo más grande = más preciso pero más lento.")

st.info("📹 **Cómo grabar:** vista **lateral** (90° respecto al plano del movimiento), cámara fija a la altura de la "
        "cadera, a 3-4 m, cuerpo entero visible y el mismo lado que el vídeo modelo.")


# ------------------------------------------------------------------ helpers
@st.cache_resource
def cargar_modelo(nombre):
    return YOLO(nombre)


def guardar_subida(f):
    clave = f"subida_{f.name}_{f.size}"
    if clave not in st.session_state:
        t = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        t.write(f.getvalue())
        t.close()
        st.session_state[clave] = t.name
    return st.session_state[clave]


def extraer_keypoints(path, modelo, imgsz, barra):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
    w, h = int(cap.get(3)), int(cap.get(4))
    kps = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        k = np.full((17, 3), np.nan, dtype=np.float32)
        r = modelo(frame, verbose=False, imgsz=imgsz)[0]
        if r.keypoints is not None and len(r.boxes) > 0:
            cajas = r.boxes.xyxy.cpu().numpy()
            areas = (cajas[:, 2] - cajas[:, 0]) * (cajas[:, 3] - cajas[:, 1])
            idx = int(np.argmax(areas))  # persona más grande = atleta principal
            k[:, :2] = r.keypoints.xy[idx].cpu().numpy()
            k[:, 2] = r.keypoints.conf[idx].cpu().numpy() if r.keypoints.conf is not None else 1.0
        kps.append(k)
        barra.progress(min(len(kps) / total, 1.0))
    cap.release()
    return dict(kps=np.array(kps), fps=fps, size=(w, h))


def leer_frame(path, idx):
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, f = cap.read()
    cap.release()
    return cv2.cvtColor(f, cv2.COLOR_BGR2RGB) if ok else None


def a_h264(path):
    """Reencodea a H.264 (si hay ffmpeg) para que el navegador lo reproduzca."""
    if shutil.which("ffmpeg") is None:
        return path
    out = path.replace(".mp4", "_h264.mp4")
    r = subprocess.run(["ffmpeg", "-y", "-i", path, "-vcodec", "libx264", "-pix_fmt", "yuv420p",
                        "-loglevel", "error", out])
    return out if r.returncode == 0 and os.path.exists(out) else path


# -------------------------------------------------------------------- entrada
col1, col2 = st.columns(2)
with col1:
    st.subheader("Vídeo modelo")
    if os.path.exists(REF_PATH):
        path_ref = REF_PATH
        st.success(f"Usando `{REF_PATH}` del repositorio")
    else:
        f = st.file_uploader("Sube el vídeo modelo", type=["mp4", "mov", "avi"], key="ref")
        path_ref = guardar_subida(f) if f else None
with col2:
    st.subheader("Vídeo del alumno")
    f = st.file_uploader("Sube el vídeo del alumno", type=["mp4", "mov", "avi"], key="al")
    path_al = guardar_subida(f) if f else None

if not (path_ref and path_al):
    st.stop()

if st.button("🚀 Analizar vídeos (extraer poses)", type="primary"):
    modelo = cargar_modelo(modelo_nombre)
    for rol, path, nombre in (("ref", path_ref, "modelo"), ("al", path_al, "alumno")):
        st.write(f"Analizando vídeo del {nombre}...")
        barra = st.progress(0)
        d = extraer_keypoints(path, modelo, imgsz, barra)
        d["path"] = path
        st.session_state[f"datos_{rol}"] = d

R, A = st.session_state.get("datos_ref"), st.session_state.get("datos_al")
if not (R and A and R["path"] == path_ref and A["path"] == path_al):
    st.warning("Pulsa **Analizar vídeos** para extraer las poses.")
    st.stop()

# ---------------------------------------------------------- métricas y avisos
lado_r, lado_a = su.lado_visible(R["kps"]), su.lado_visible(A["kps"])
df_r, df_a = su.calcular_metricas(R["kps"], lado_r), su.calcular_metricas(A["kps"], lado_a)

for nombre, d, lado in (("modelo", R, lado_r), ("alumno", A, lado_a)):
    rat, cob = su.ratio_lateral(d["kps"], lado), su.cobertura(d["kps"], lado)
    if rat is not None and rat > 0.45:
        st.warning(f"⚠️ El vídeo del **{nombre}** no parece lateral (ratio hombros/torso = {rat:.2f}). "
                   "Los ángulos 2D estarán distorsionados y la comparación será solo orientativa.")
    if cob < 0.7:
        st.warning(f"⚠️ En el vídeo del **{nombre}** solo se detecta bien el cuerpo en el {cob:.0%} de los frames "
                   "(¿oclusión por la barra o el cuerpo cortado?).")
st.caption(f"Lado analizado → modelo: {lado_r} · alumno: {lado_a}")

# ------------------------------------------------------------------- recorte
st.subheader("1️⃣ Recorta el intento")
st.write("Deja solo desde la salida hasta que el atleta termina de pie con la barra arriba "
         "(quita celebraciones, tiros de la barra, etc.).")
c1, c2 = st.columns(2)
segs, fps_ = {}, {}
for col, rol, d, nombre in ((c1, "ref", R, "Modelo"), (c2, "al", A, "Alumno")):
    with col:
        n, fps = len(d["kps"]), d["fps"]
        dur = n / fps
        t0, t1 = st.slider(f"{nombre} (s)", 0.0, float(round(dur, 1)), (0.0, float(round(dur, 1))), 0.1, key=f"s_{rol}")
        i0, i1 = int(t0 * fps), min(n - 1, int(t1 * fps))
        segs[rol] = (i0, i1)
        a, b = st.columns(2)
        a.image(leer_frame(d["path"], i0), caption="inicio", use_container_width=True)
        b.image(leer_frame(d["path"], i1), caption="fin", use_container_width=True)

(r0, r1), (a0, a1) = segs["ref"], segs["al"]
if r1 - r0 < 15 or a1 - a0 < 15:
    st.error("El segmento es demasiado corto.")
    st.stop()
if (r1 - r0) > 1200 or (a1 - a0) > 1200:
    st.error("Segmento demasiado largo para alinear (máx. ~1200 frames). Recórtalo más.")
    st.stop()

sr, sa = df_r.iloc[r0:r1 + 1].reset_index(drop=True), df_a.iloc[a0:a1 + 1].reset_index(drop=True)

# ---------------------------------------------------------------- alineación
with st.spinner("Alineando movimientos (DTW)..."):
    path = su.alinear_dtw(sr, sa)
    mapa = su.mapa_alumno_a_ref(path, len(sa))
cr, ca = su.detectar_recepcion(sr), su.detectar_recepcion(sa)

st.subheader("2️⃣ Resultados")
m1, m2, m3 = st.columns(3)
m1.metric("Recepción del modelo", f"{cr / len(sr):.0%} del movimiento")
m2.metric("Recepción del alumno", f"{ca / len(sa):.0%} del movimiento")
m3.metric("Diferencia de timing", f"{(ca / len(sa) - cr / len(sr)) * 100:+.0f} pp")
st.caption("El tiempo se compara en proporción (no en segundos) porque el modelo puede estar en cámara lenta.")

# ---- curvas alineadas
dfp = pd.DataFrame(path, columns=["i", "j"]).groupby("i")["j"].mean()
x = np.arange(len(sr))
curvas = st.columns(2)
for n, ang in enumerate(su.ANGULOS):
    al = np.interp(dfp.reindex(x).interpolate().bfill().ffill().values, np.arange(len(sa)), sa[ang].values)
    with curvas[n % 2]:
        st.markdown(f"**{ang}** (°)")
        st.line_chart(pd.DataFrame({"Modelo": sr[ang].values, "Alumno": al}, index=x / (len(sr) - 1) * 100),
                      height=200)

# ---- tabla de momentos clave
momentos = {"Posición inicial": (0, 0), "Recepción": (cr, ca), "Final": (len(sr) - 1, len(sa) - 1)}
filas, pistas = [], []
for mom, (i, j) in momentos.items():
    for ang in su.ANGULOS:
        vr, va = su.valor_en(sr, ang, i), su.valor_en(sa, ang, j)
        d = va - vr
        flag = abs(d) > umbral
        filas.append({"Momento": mom, "Ángulo": ang, "Modelo (°)": round(vr), "Alumno (°)": round(va),
                      "Dif. (°)": round(d), "": "⚠️" if flag else "✅"})
        if flag:
            pistas.append(f"**{mom} · {ang} ({d:+.0f}°):** {su.pista(ang, mom, d)}")
st.markdown("#### Momentos clave")
st.dataframe(pd.DataFrame(filas), use_container_width=True, hide_index=True)

# ---- por fases (según el camino DTW)
st.markdown("#### Diferencia media por fase")
fases = {"Subida y recepción": [(i, j) for i, j in path if i <= cr],
         "Levantada": [(i, j) for i, j in path if i > cr]}
filas_f = []
for fase, pares in fases.items():
    if not pares:
        continue
    ii, jj = np.array([p[0] for p in pares]), np.array([p[1] for p in pares])
    fila = {"Fase": fase}
    for ang in su.ANGULOS:
        d = sa[ang].values[jj] - sr[ang].values[ii]
        fila[f"{ang} (media ±)"] = f"{d.mean():+.0f}° (|{np.abs(d).mean():.0f}|)"
    filas_f.append(fila)
st.dataframe(pd.DataFrame(filas_f), use_container_width=True, hide_index=True)
st.caption("Signo: alumno − modelo. Entre paréntesis, el error absoluto medio.")

st.markdown("#### 🧠 Pistas para el entrenador")
if pistas:
    for p in pistas:
        st.markdown(f"- {p}")
else:
    st.success(f"Ninguna diferencia supera {umbral}°.")
st.caption("Son diferencias respecto al modelo, no errores confirmados: las proporciones y la movilidad "
           "de cada atleta cambian, y el análisis 2D depende del ángulo de cámara.")

# --------------------------------------------------------------------- vídeo
st.subheader("3️⃣ Vídeo comparativo")
clave_seg = (r0, r1, a0, a1, umbral)
if st.button("🎬 Generar vídeo lado a lado"):
    with st.spinner("Renderizando..."):
        out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        barra = st.progress(0)
        su.render_comparacion(R["path"], R["kps"], df_r, lado_r, (r0, r1),
                              A["path"], A["kps"], df_a, lado_a, (a0, a1),
                              mapa, umbral, out, progreso=barra.progress)
        st.session_state["video_cmp"] = (clave_seg, a_h264(out))
v = st.session_state.get("video_cmp")
if v and v[0] == clave_seg:
    st.video(v[1])
    st.caption("En rojo: segmentos del cuerpo cuyo ángulo se desvía más del umbral en ese momento.")
