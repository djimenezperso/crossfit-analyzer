import streamlit as st
import cv2
import math
import os
import numpy as np
from ultralytics import YOLO
from collections import deque
import tempfile

# Configuración de la página web
st.set_page_config(page_title="CrossFit Analyzer App", page_icon="🏋️‍♂️️", layout="centered")

st.title("🏋️‍♂️ Analizador de CrossFit (Squat Snatch & Pull-ups)")
st.write("Sube tu vídeo, selecciona el movimiento y obtén el análisis de repeticiones, validez y potencia.")

# Selección de modalidad
modalidad = st.selectbox("Selecciona el ejercicio:", ["Pull-ups (Dominadas)", "Squat Snatch"])

# Datos del atleta
col1, col2 = st.columns(2)
with col1:
    peso_atleta = st.number_input("Peso del atleta (kg):", value=50.0, step=1.0)
with col2:
    if modalidad == "Squat Snatch":
        peso_barra = st.number_input("Peso de la barra (kg):", value=40.0, step=1.0)
        altura_atleta = st.number_input("Altura (m):", value=1.75, step=0.01)
    else:
        peso_barra = 0.0
        altura_atleta = st.number_input("Altura (m):", value=1.60, step=0.01)

# Subir archivo de vídeo
video_file = st.file_uploader("Sube tu archivo de vídeo (.mp4)", type=["mp4", "mov", "avi"])

@st.cache_resource
def cargar_modelo():
    # Carga el modelo (en producción web se suele usar el .pt estándar o un export compatible)
    return YOLO('yolov8n-pose.pt')

if video_file is not None:
    # Guardar el vídeo subido temporalmente en el servidor
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(video_file.read())
    video_path = tfile.name

    if st.button("🚀 Analizar Vídeo"):
        with st.spinner("Procesando vídeo con IA... Esto puede tardar unos segundos."):
            
            model = cargar_modelo()
            cap = cv2.VideoCapture(video_path)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Archivo temporal de salida para el vídeo procesado
            output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.avi').name
            out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'MJPG'), fps, (width, height))

            repeticiones = 0
            detalles_reps = []
            contador_frames = 0
            
            # Variables de lógica para Pull-ups
            fase = "ABAJO"
            h_min = 0.0
            hang_start = 0.0
            historial_h = deque(maxlen=3)
            ventana_h = deque(maxlen=int(4.0 * fps))
            h_anterior = None
            tiempo_anterior = None
            max_vel_subida_rep = 0.0
            ultimo_frame_rep = -10**9

            progress_bar = st.progress(0)

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                contador_frames += 1
                tiempo_s = contador_frames / fps
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

                                    # Cálculo de velocidad
                                    metros_por_pixel = altura_atleta / (torso * 2.2)
                                    if h_anterior is not None and tiempo_anterior is not None:
                                        dt = tiempo_s - tiempo_anterior
                                        if dt > 0:
                                            dh = h_anterior - h
                                            vel = max(0.0, (dh * torso * metros_por_pixel) / dt)
                                    h_anterior = h
                                    tiempo_anterior = tiempo_s

                                    hang_ref = float(np.percentile(ventana_h, 80)) if len(ventana_h) >= 15 else 1.0
                                    recorrido = hang_ref - h

                                    if fase == "ABAJO":
                                        max_vel_subida_rep = 0.0
                                        if recorrido >= 0.15:
                                            fase = "SUBIENDO"
                                            h_min = h
                                    elif fase == "SUBIENDO":
                                        h_min = min(h_min, h)
                                        if 'vel' in locals() and vel > max_vel_subida_rep:
                                            max_vel_subida_rep = vel
                                        if (h - h_min) > 0.08:  # Detecta bajada
                                            drop_pico = hang_ref - h_min
                                            if drop_pico >= 0.30 and (contador_frames - ultimo_frame_rep) >= int(0.6 * fps):
                                                repeticiones += 1
                                                ultimo_frame_rep = contador_frames
                                                fuerza_N = peso_atleta * 9.81
                                                potencia_W = fuerza_N * max(0.4, max_vel_subida_rep)
                                                detalles_reps.append({"rep": repeticiones, "vel": round(max(0.4, max_vel_subida_rep), 2), "pot": int(potencia_W)})
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

            # Mostrar tabla de potencia
            if len(detalles_reps) > 0:
                st.subheader("📊 Desglose de Potencia y Velocidad por Repetición")
                st.table(detalles_reps)

            # Mostrar vídeo resultante
            st.subheader("Vídeo Analizado:")
            st.video(output_path)
