import streamlit as st
import cv2
import numpy as np
import tempfile
import yt_dlp
from ultralytics import YOLO
from collections import deque

# Configuración de la página web
st.set_page_config(page_title="CrossFit Rep Counter", page_icon="🏋️‍♂️", layout="centered")

st.title("🏋️‍♂️ Contador de Repeticiones CrossFit")
st.write("Sube tu vídeo o pega un enlace de YouTube para analizar tus repeticiones.")

# Selección de modalidad
modalidad = st.selectbox("Selecciona el ejercicio:", ["Pull-ups (Dominadas)", "Squat Snatch"])

# Selección de fuente de vídeo
fuente_video = st.radio("¿Cómo quieres analizar el vídeo?", ("Subir archivo", "Enlace de YouTube"))

video_path = None

if fuente_video == "Subir archivo":
    video_file = st.file_uploader("Sube tu archivo de vídeo (.mp4, .mov)", type=["mp4", "mov", "avi"])
    if video_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(video_file.read())
        video_path = tfile.name
else:
    youtube_url = st.text_input("Pega el enlace de YouTube aquí:")
    if youtube_url:
        with st.spinner("Descargando vídeo de YouTube..."):
            try:
                # Opciones para descargar el video en formato mp4 compatible
                ydl_opts = {
                    'format': 'best[ext=mp4]', 
                    'outtmpl': tempfile.mktemp(suffix='.mp4'), 
                    'quiet': True
                }
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(youtube_url, download=True)
                    video_path = ydl.prepare_filename(info)
                st.success("Vídeo descargado correctamente. ¡Listo para analizar!")
            except Exception as e:
                st.error(f"Error al descargar el vídeo: {e}")

@st.cache_resource
def cargar_modelo():
    return YOLO('yolov8n-pose.pt')

if video_path and st.button("🚀 Analizar Vídeo"):
    with st.spinner("Procesando vídeo con IA... Esto puede tardar unos segundos."):
        
        model = cargar_modelo()
        cap = cv2.VideoCapture(video_path)
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Archivo temporal de salida
        output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
        # mp4v suele dar mejor compatibilidad en navegadores modernos al mostrarlo en Streamlit
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        repeticiones = 0
        contador_frames = 0
        
        # Variables de lógica para Pull-ups (Simplificadas sin cálculo de velocidad)
        fase = "ABAJO"
        h_min = 0.0
        historial_h = deque(maxlen=3)
        ventana_h = deque(maxlen=int(4.0 * fps))
        ultimo_frame_rep = -1000

        progress_bar = st.progress(0)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            contador_frames += 1
            progress_bar.progress(min(contador_frames / total_frames, 1.0))

            resultados = model(frame, verbose=False, imgsz=320)
            fotograma_dibujado = frame.copy()

            if resultados[0].keypoints is not None and len(resultados[0].keypoints.xy) > 0:
                cajas = resultados[0].boxes.xyxy.tolist()
                areas = [(c[2] - c[0]) * (c[3] - c[1]) for c in cajas]
                persona_idx = int(np.argmax(areas))
                
                res_aislado = resultados[0][persona_idx]
                fotograma_dibujado = res_aislado.plot(img=frame.copy(), boxes=False, labels=False)
                puntos = res_aislado.keypoints.xy[0]
                conf = res_aislado.keypoints.conf
                conf = conf[0] if conf is not None else None

                if modalidad == "Pull-ups (Dominadas)" and len(puntos) > 12 and conf is not None:
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
                                orejas = [(float(conf[i]), P(i)) for i in (3, 4) if float(conf[i]) > 0.2]
                                
                                if len(orejas) > 0:
                                    punto_cabeza = max(orejas, key=lambda x: x[0])[1]
                                    barbilla = punto_cabeza - u * (0.20 * torso)
                                else:
                                    barbilla = hombro_mid + u * (0.25 * torso)

                                h_bruto = (s_barra - s(barbilla)) / torso - 0.15
                                historial_h.append(h_bruto)
                                h = sum(historial_h) / len(historial_h)
                                ventana_h.append(h)

                                hang_ref = float(np.percentile(ventana_h, 80)) if len(ventana_h) >= 15 else 1.0
                                recorrido = hang_ref - h

                                # Lógica de conteo (Solo tracking espacial)
                                if fase == "ABAJO":
                                    if recorrido >= 0.15:
                                        fase = "SUBIENDO"
                                        h_min = h
                                elif fase == "SUBIENDO":
                                    h_min = min(h_min, h)
                                    if (h - h_min) > 0.08:  # Detecta inicio de bajada
                                        drop_pico = hang_ref - h_min
                                        if drop_pico >= 0.30 and (contador_frames - ultimo_frame_rep) >= int(0.6 * fps):
                                            repeticiones += 1
                                            ultimo_frame_rep = contador_frames
                                        fase = "BAJANDO"
                                elif fase == "BAJANDO":
                                    if h >= h_min + 0.5 * (hang_ref - h_min):
                                        fase = "ABAJO"

            # Pintar repeticiones en el vídeo
            cv2.rectangle(fotograma_dibujado, (10, 10), (160, 60), (0, 0, 0), -1)
            cv2.putText(fotograma_dibujado, f"Reps: {repeticiones}", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
            out.write(fotograma_dibujado)

        cap.release()
        out.release()

        st.success("¡Análisis completado con éxito!")
        st.metric(label="Total de Repeticiones Válidas", value=repeticiones)

        st.subheader("Vídeo Analizado:")
        st.video(output_path)