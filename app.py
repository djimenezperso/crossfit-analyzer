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

REF_PATH = "referencia_snatch.mp4"

st.set_page_config(page_title="CrossFit Analyzer Pro", page_icon="🏋️‍♂️", layout="wide")

# --- MENÚ LATERAL PARA ELEGIR MODO ---
st.sidebar.title("🏋️‍♂️ Menú Principal")
modo = st.sidebar.selectbox("Selecciona la herramienta:", [
    "⚔️ Comparador Élite (Lü Xiaojun)",
    "🔢 Contador de Repeticiones (Pull-ups / Snatch)"
])

@st.cache_resource
def cargar_modelo(nombre="yolov8n-pose.pt"):
    return YOLO(nombre)

def guardar_subida(f):
    clave = f"subida_{f.name}_{f.size}"
    if clave not in st.session_state:
        t = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        t.write(f.getvalue())
        t.close()
        st.session_state[clave] = t.name
    return st.session_state[clave]


# =========================================================================
# MODO 1: COMPARADOR ÉLITE
# =========================================================================
if modo == "⚔️ Comparador Élite (Lü Xiaojun)":
    st.title("🏋️‍♂️ Comparador de Squat Snatch (Técnica Élite)")
    st.write("Compara la técnica de tu alumno frente al estándar olímpico.")

    with st.sidebar:
        st.header("Ajustes Comparador")
        modelo_nombre = st.selectbox("Modelo de pose", ["yolov8m-pose.pt", "yolov8l-pose.pt", "yolov8s-pose.pt"])
        imgsz = st.select_slider("Resolución de análisis", [320, 480, 640, 960], value=640)
        umbral = st.slider("Umbral de diferencia (grados)", 5, 30, 15)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Vídeo modelo (Olímpico)")
        if os.path.exists(REF_PATH):
            path_ref = REF_PATH
            st.success("Vídeo de referencia cargado correctamente.")
        else:
            st.error(f"Falta el archivo '{REF_PATH}' en el repositorio de GitHub.")
            path_ref = None

    with col2:
        st.subheader("Vídeo del alumno")
        f = st.file_uploader("Sube el vídeo del alumno", type=["mp4", "mov", "avi"], key="al_comp")
        path_al = guardar_subida(f) if f else None

    if path_ref and path_al and st.button("🚀 Analizar y Comparar", type="primary"):
        with st.spinner("Extrayendo poses y alineando movimientos..."):
            modelo = cargar_modelo(modelo_nombre)
            
            # Extraer keypoints referencia
            cap_r = cv2.VideoCapture(path_ref)
            total_r = int(cap_r.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
            fps_r = cap_r.get(cv2.CAP_PROP_FPS) or 25
            w_r, h_r = int(cap_r.get(3)), int(cap_r.get(4))
            kps_r = []
            while cap_r.isOpened():
                ok, frame = cap_r.read()
                if not ok: break
                k = np.full((17, 3), np.nan, dtype=np.float32)
                r = modelo(frame, verbose=False, imgsz=imgsz)[0]
                if r.keypoints is not None and len(r.boxes) > 0:
                    idx = int(np.argmax((r.boxes.xyxy[:, 2] - r.boxes.xyxy[:, 0]) * (r.boxes.xyxy[:, 3] - r.boxes.xyxy[:, 1])))
                    k[:, :2] = r.keypoints.xy[idx].cpu().numpy()
                    k[:, 2] = r.keypoints.conf[idx].cpu().numpy() if r.keypoints.conf is not None else 1.0
                kps_r.append(k)
            cap_r.release()
            R = dict(kps=np.array(kps_r), fps=fps_r, size=(w_r, h_r), path=path_ref)

            # Extraer keypoints alumno
            cap_a = cv2.VideoCapture(path_al)
            total_a = int(cap_a.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
            fps_a = cap_a.get(cv2.CAP_PROP_FPS) or 25
            w_a, h_a = int(cap_a.get(3)), int(cap_a.get(4))
            kps_a = []
            while cap_a.isOpened():
                ok, frame = cap_a.read()
                if not ok: break
                k = np.full((17, 3), np.nan, dtype=np.float32)
                r = modelo(frame, verbose=False, imgsz=imgsz)[0]
                if r.keypoints is not None and len(r.boxes) > 0:
                    idx = int(np.argmax((r.boxes.xyxy[:, 2] - r.boxes.xyxy[:, 0]) * (r.boxes.xyxy[:, 3] - r.boxes.xyxy[:, 1])))
                    k[:, :2] = r.keypoints.xy[idx].cpu().numpy()
                    k[:, 2] = r.keypoints.conf[idx].cpu().numpy() if r.keypoints.conf is not None else 1.0
                kps_a.append(k)
            cap_a.release()
            A = dict(kps=np.array(kps_a), fps=fps_a, size=(w_a, h_a), path=path_al)

            st.session_state["datos_ref"] = R
            st.session_state["datos_al"] = A

        st.success("¡Análisis completado! Desplázate hacia abajo para ver las métricas.")

    # Si ya hay datos analizados, mostrar resultados del comparador
    if "datos_ref" in st.session_state and "datos_al" in st.session_state:
        R, A = st.session_state["datos_ref"], st.session_state["datos_al"]
        lado_r, lado_a = su.lado_visible(R["kps"]), su.lado_visible(A["kps"])
        df_r, df_a = su.calcular_metricas(R["kps"], lado_r), su.calcular_metricas(A["kps"], lado_a)

        st.subheader("1️⃣ Recorta el intento")
        c1, c2 = st.columns(2)
        segs = {}
        for col, rol, d, nombre in ((c1, "ref", R, "Modelo"), (c2, "al", A, "Alumno")):
            with col:
                n, fps = len(d["kps"]), d["fps"]
                t0, t1 = st.slider(f"{nombre} (s)", 0.0, float(round(n/fps, 1)), (0.0, float(round(n/fps, 1))), 0.1, key=f"s_{rol}")
                i0, i1 = int(t0 * fps), min(n - 1, int(t1 * fps))
                segs[rol] = (i0, i1)

        (r0, r1), (a0, a1) = segs["ref"], segs["al"]
        sr, sa = df_r.iloc[r0:r1 + 1].reset_index(drop=True), df_a.iloc[a0:a1 + 1].reset_index(drop=True)

        path_dtw = su.alinear_dtw(sr, sa)
        mapa = su.mapa_alumno_a_ref(path_dtw, len(sa))
        cr, ca = su.detectar_recepcion(sr), su.detectar_recepcion(sa)

        st.subheader("2️⃣ Resultados y Gráficas")
        m1, m2, m3 = st.columns(3)
        m1.metric("Recepción del modelo", f"{cr / len(sr):.0%}")
        m2.metric("Recepción del alumno", f"{ca / len(sa):.0%}")
        m3.metric("Diferencia timing", f"{(ca / len(sa) - cr / len(sr)) * 100:+.0f} pp")

        # Curvas
        dfp = pd.DataFrame(path_dtw, columns=["i", "j"]).groupby("i")["j"].mean()
        x = np.arange(len(sr))
        curvas = st.columns(2)
        for n, ang in enumerate(su.ANGULOS):
            al = np.interp(dfp.reindex(x).interpolate().bfill().ffill().values, np.arange(len(sa)), sa[ang].values)
            with curvas[n % 2]:
                st.markdown(f"**{ang}** (°)")
                st.line_chart(pd.DataFrame({"Modelo": sr[ang].values, "Alumno": al}, index=x / max(len(sr) - 1, 1) * 100), height=200)

        # Vídeo comparativo lado a lado
        clave_seg = (r0, r1, a0, a1, umbral)
        if st.button("🎬 Generar vídeo lado a lado con errores en rojo"):
            with st.spinner("Renderizando vídeo comparativo..."):
                out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
                barra = st.progress(0)
                su.render_comparacion(R["path"], R["kps"], df_r, lado_r, (r0, r1),
                                      A["path"], A["kps"], df_a, lado_a, (a0, a1),
                                      mapa, umbral, out, progreso=barra.progress)
                
                # Reencode h264 si hay ffmpeg
                out_final = out
                if shutil.which("ffmpeg") is not None:
                    h264_out = out.replace(".mp4", "_h264.mp4")
                    r = subprocess.run(["ffmpeg", "-y", "-i", out, "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-loglevel", "error", h264_out])
                    if r.returncode == 0 and os.path.exists(h264_out):
                        out_final = h264_out

                st.session_state["video_cmp"] = (clave_seg, out_final)

        v = st.session_state.get("video_cmp")
        if v and v[0] == clave_seg:
            st.video(v[1])


# =========================================================================
# MODO 2: CONTADOR DE REPETICIONES (PULL-UPS Y SNATCH)
# =========================================================================
elif modo == "🔢 Contador de Repeticiones (Pull-ups / Snatch)":
    st.title("🏋️‍♂️️ Contador Automático de Repeticiones")
    modalidad = st.selectbox("Selecciona el ejercicio:", ["Pull-ups (Dominadas)", "Squat Snatch"])
    
    video_file = st.file_uploader("Sube tu archivo de vídeo (.mp4, .mov, .avi)", type=["mp4", "mov", "avi"])
    
    if video_file is not None and st.button("🚀 Contar Repeticiones", type="primary"):
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(video_file.read())
        
        with st.spinner("Contando repeticiones con IA..."):
            model = cargar_modelo('yolov8n-pose.pt')
            cap = cv2.VideoCapture(tfile.name)
            
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            out_clean = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
            out_skeleton = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
            
            writer_clean = cv2.VideoWriter(out_clean, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
            writer_skel = cv2.VideoWriter(out_skeleton, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

            repeticiones = 0
            contador_frames = 0
            fase = "ABAJO"
            ultimo_frame_rep = -1000
            
            from collections import deque
            historial_h = deque(maxlen=3)
            ventana_h = deque(maxlen=int(4.0 * max(fps, 1)))

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                contador_frames += 1

                resultados = model(frame, verbose=False, imgsz=320)
                fotograma_limpio = frame.copy()
                fotograma_esqueleto = frame.copy()

                if resultados[0].keypoints is not None and len(resultados[0].keypoints.xy) > 0:
                    cajas = resultados[0].boxes.xyxy.tolist()
                    areas = [(c[2] - c[0]) * (c[3] - c[1]) for c in cajas]
                    persona_idx = int(np.argmax(areas))
                    res_aislado = resultados[0][persona_idx]
                    
                    fotograma_esqueleto = res_aislado.plot(img=frame.copy(), boxes=False, labels=False)
                    puntos = res_aislado.keypoints.xy[0]
                    conf = res_aislado.keypoints.conf
                    conf = conf[0] if conf is not None else None

                    if conf is not None and len(puntos) > 14:
                        if modalidad == "Pull-ups (Dominadas)":
                            ok = lambda idx: float(conf[idx]) > 0.4
                            P = lambda idx: np.array([float(puntos[idx][0]), float(puntos[idx][1])])
                            if ok(5) and ok(6) and ok(11) and ok(12):
                                hombro_mid = (P(5) + P(6)) / 2
                                cadera_mid = (P(11) + P(12)) / 2
                                eje = hombro_mid - cadera_mid
                                torso = float(np.linalg.norm(eje))
                                if torso > 20:
                                    u = eje / torso
                                    s = lambda p: float(np.dot(p, u))
                                    s_hombro = s(hombro_mid)
                                    munecas = [P(i) for i in (9, 10) if ok(i) and s(P(i)) > s_hombro - 0.2 * torso]
                                    if len(munecas) > 0:
                                        s_barra = sum(s(m) for m in munecas) / len(munecas)
                                        h_bruto = (s_barra - s(hombro_mid)) / torso
                                        historial_h.append(h_bruto)
                                        h = sum(historial_h) / len(historial_h)
                                        if fase == "ABAJO" and h < 0.2:
                                            fase = "SUBIENDO"
                                        elif fase == "SUBIENDO" and h > 0.4 and (contador_frames - ultimo_frame_rep) >= int(0.5 * fps):
                                            repeticiones += 1
                                            ultimo_frame_rep = contador_frames
                                            fase = "BAJANDO"
                                        elif fase == "BAJANDO" and h < 0.2:
                                            fase = "ABAJO"

                        elif modalidad == "Squat Snatch":
                            y_muneca = float(puntos[9][1]) if float(conf[9]) > 0.3 else None
                            y_hombro = float(puntos[5][1]) if float(conf[5]) > 0.3 else None
                            y_cadera = float(puntos[11][1]) if float(conf[11]) > 0.3 else None
                            y_rodilla = float(puntos[13][1]) if float(conf[13]) > 0.3 else None

                            if None not in (y_muneca, y_hombro, y_cadera, y_rodilla):
                                barra_arriba = y_muneca < (y_hombro + 20)
                                en_squat = y_cadera >= (y_rodilla - 25)
                                de_pie = y_cadera < (y_rodilla - 40)

                                if fase == "ABAJO" and barra_arriba and en_squat:
                                    fase = "SQUAT"
                                elif fase == "SQUAT" and barra_arriba and de_pie and (contador_frames - ultimo_frame_rep) >= int(0.8 * fps):
                                    repeticiones += 1
                                    ultimo_frame_rep = contador_frames
                                    fase = "ABAJO"

                for img in (fotograma_limpio, fotograma_esqueleto):
                    cv2.rectangle(img, (10, 10), (180, 60), (0, 0, 0), -1)
                    cv2.putText(img, f"Reps: {repeticiones}", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
                
                writer_clean.write(fotograma_limpio)
                writer_skel.write(fotograma_esqueleto)

            cap.release()
            writer_clean.release()
            writer_skel.release()

            st.success("¡Conteno finalizado!")
            st.metric("Total Repeticiones", repeticiones)
            st.subheader("Vídeo Limpio")
            st.video(out_clean)
            with st.expander("Ver esqueleto digital"):
                st.video(out_skeleton)