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

REF_PATH = "referencia_snatch.mp4"  # vidéo modèle optionnelle, à mettre à la racine du dépôt

st.set_page_config(page_title="Comparateur de Snatch", page_icon="🏋️", layout="wide")
st.title("🏋️ Comparateur de Squat Snatch")
st.write("Compare la technique d'un élève avec une vidéo modèle : angles articulaires, phases et vidéo synchronisée.")

with st.sidebar:
    st.header("Réglages")
    modele_nom = st.selectbox("Modèle de pose", ["yolov8s-pose.pt", "yolov8n-pose.pt", "yolov8m-pose.pt"],
                              help="n = le plus rapide, m = le plus précis.")
    imgsz = st.select_slider("Résolution d'analyse", [320, 480, 640, 960], value=480)
    saut = st.select_slider("Analyser 1 image sur…", [1, 2, 3], value=1,
                            help="2 = deux fois plus rapide, mais moins précis. Le reste est interpolé.")
    seuil = st.slider("Seuil d'écart (degrés)", 5, 30, 15)
    st.caption("Plus de résolution et un modèle plus gros = plus précis mais plus lent.")

st.info("📹 **Comment filmer :** vue **de profil** (90° par rapport au plan du mouvement), caméra fixe à hauteur "
        "de hanche, à 3-4 m, corps entier visible et du même côté que la vidéo modèle.")


# ------------------------------------------------------------------ fonctions
@st.cache_resource
def charger_modele(nom):
    return YOLO(nom)


def sauver_upload(f):
    cle = f"upload_{f.name}_{f.size}"
    if cle not in st.session_state:
        t = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        t.write(f.getvalue())
        t.close()
        st.session_state[cle] = t.name
    return st.session_state[cle]


def extraire_keypoints(path, modele, imgsz, saut, barre):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
    w, h = int(cap.get(3)), int(cap.get(4))
    kps, i = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        k = np.full((17, 3), np.nan, dtype=np.float32)
        if i % saut == 0:
            r = modele(frame, verbose=False, imgsz=imgsz)[0]
            if r.keypoints is not None and len(r.boxes) > 0:
                boites = r.boxes.xyxy.cpu().numpy()
                aires = (boites[:, 2] - boites[:, 0]) * (boites[:, 3] - boites[:, 1])
                idx = int(np.argmax(aires))  # la plus grande personne = l'athlète principal
                k[:, :2] = r.keypoints.xy[idx].cpu().numpy()
                k[:, 2] = r.keypoints.conf[idx].cpu().numpy() if r.keypoints.conf is not None else 1.0
        kps.append(k)
        i += 1
        barre.progress(min(i / total, 1.0))
    cap.release()
    kps = np.array(kps)
    if saut > 1:  # interpolation des images sautées
        for p in range(17):
            for c in range(3):
                col = kps[:, p, c]
                ok = ~np.isnan(col)
                if ok.sum() > 1:
                    kps[:, p, c] = np.interp(np.arange(len(col)), np.where(ok)[0], col[ok])
    return dict(kps=kps, fps=fps, size=(w, h))


def lire_image(path, idx):
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, f = cap.read()
    cap.release()
    return cv2.cvtColor(f, cv2.COLOR_BGR2RGB) if ok else None


def vers_h264(path):
    """Réencode en H.264 (si ffmpeg est disponible) pour que le navigateur le lise."""
    if shutil.which("ffmpeg") is None:
        return path
    out = path.replace(".mp4", "_h264.mp4")
    r = subprocess.run(["ffmpeg", "-y", "-i", path, "-vcodec", "libx264", "-pix_fmt", "yuv420p",
                        "-loglevel", "error", out])
    return out if r.returncode == 0 and os.path.exists(out) else path


# -------------------------------------------------------------------- entrée
col1, col2 = st.columns(2)
with col1:
    st.subheader("Vidéo modèle")
    if os.path.exists(REF_PATH):
        path_ref = REF_PATH
        st.success(f"Utilisation de `{REF_PATH}` du dépôt")
    else:
        f = st.file_uploader("Téléverse la vidéo modèle", type=["mp4", "mov", "avi"], key="ref")
        path_ref = sauver_upload(f) if f else None
with col2:
    st.subheader("Vidéo de l'élève")
    f = st.file_uploader("Téléverse la vidéo de l'élève", type=["mp4", "mov", "avi"], key="al")
    path_al = sauver_upload(f) if f else None

if not (path_ref and path_al):
    st.stop()

if st.button("🚀 Analyser les vidéos (extraire les poses)", type="primary"):
    modele = charger_modele(modele_nom)
    for role, path, nom in (("ref", path_ref, "modèle"), ("al", path_al, "élève")):
        st.write(f"Analyse de la vidéo {nom}...")
        barre = st.progress(0)
        d = extraire_keypoints(path, modele, imgsz, saut, barre)
        d["path"] = path
        st.session_state[f"datos_{role}"] = d

R, A = st.session_state.get("datos_ref"), st.session_state.get("datos_al")
if not (R and A and R["path"] == path_ref and A["path"] == path_al):
    st.warning("Clique sur **Analyser les vidéos** pour extraire les poses.")
    st.stop()

# ---------------------------------------------------------- métriques et alertes
lado_r, lado_a = su.lado_visible(R["kps"]), su.lado_visible(A["kps"])
df_r, df_a = su.calcular_metricas(R["kps"], lado_r), su.calcular_metricas(A["kps"], lado_a)
nom_cote = {"izquierdo": "gauche", "derecho": "droit"}

for nom, d, lado in (("modèle", R, lado_r), ("élève", A, lado_a)):
    rat, cob = su.ratio_lateral(d["kps"], lado), su.cobertura(d["kps"], lado)
    if rat is not None and rat > 0.45:
        st.warning(f"⚠️ La vidéo de l'**{nom}** ne semble pas filmée de profil (ratio épaules/buste = {rat:.2f}). "
                   "Les angles 2D seront déformés : comparaison seulement indicative.")
    if cob < 0.7:
        st.warning(f"⚠️ Dans la vidéo de l'**{nom}**, le corps n'est bien détecté que dans {cob:.0%} des images "
                   "(barre qui masque le corps, corps coupé ?).")
st.caption(f"Côté analysé → modèle : {nom_cote[lado_r]} · élève : {nom_cote[lado_a]}")

# ------------------------------------------------------------------- découpage
st.subheader("1️⃣ Découpe la tentative")
st.write("Garde seulement le moment entre le départ et la fin du mouvement, debout avec la barre en haut "
         "(enlève les célébrations, la barre lâchée, etc.).")
c1, c2 = st.columns(2)
segs = {}
for col, role, d, nom in ((c1, "ref", R, "Modèle"), (c2, "al", A, "Élève")):
    with col:
        n, fps = len(d["kps"]), d["fps"]
        dur = n / fps
        t0, t1 = st.slider(f"{nom} (s)", 0.0, float(round(dur, 1)), (0.0, float(round(dur, 1))), 0.1, key=f"s_{role}")
        i0, i1 = int(t0 * fps), min(n - 1, int(t1 * fps))
        segs[role] = (i0, i1)
        a, b = st.columns(2)
        a.image(lire_image(d["path"], i0), caption="début", use_container_width=True)
        b.image(lire_image(d["path"], i1), caption="fin", use_container_width=True)

(r0, r1), (a0, a1) = segs["ref"], segs["al"]
if r1 - r0 < 15 or a1 - a0 < 15:
    st.error("Le segment est trop court.")
    st.stop()
if (r1 - r0) > 1200 or (a1 - a0) > 1200:
    st.error("Segment trop long pour l'alignement (max. ~1200 images). Raccourcis-le.")
    st.stop()

sr, sa = df_r.iloc[r0:r1 + 1].reset_index(drop=True), df_a.iloc[a0:a1 + 1].reset_index(drop=True)

# ---------------------------------------------------------------- alignement
with st.spinner("Alignement des mouvements (DTW)..."):
    path = su.alinear_dtw(sr, sa)
    mapa = su.mapa_alumno_a_ref(path, len(sa))
cr, ca = su.detectar_recepcion(sr), su.detectar_recepcion(sa)

st.subheader("2️⃣ Résultats")
m1, m2, m3 = st.columns(3)
m1.metric("Réception du modèle", f"{cr / len(sr):.0%} du mouvement")
m2.metric("Réception de l'élève", f"{ca / len(sa):.0%} du mouvement")
m3.metric("Écart de timing", f"{(ca / len(sa) - cr / len(sr)) * 100:+.0f} pts")
st.caption("Le temps est comparé en proportion (pas en secondes) car le modèle peut être au ralenti.")

# ---- courbes alignées
dfp = pd.DataFrame(path, columns=["i", "j"]).groupby("i")["j"].mean()
x = np.arange(len(sr))
courbes = st.columns(2)
for n, ang in enumerate(su.ANGULOS):
    al = np.interp(dfp.reindex(x).interpolate().bfill().ffill().values, np.arange(len(sa)), sa[ang].values)
    with courbes[n % 2]:
        st.markdown(f"**{ang}** (°)")
        st.line_chart(pd.DataFrame({"Modèle": sr[ang].values, "Élève": al}, index=x / (len(sr) - 1) * 100),
                      height=200)

# ---- tableau des moments clés
moments = {"Position de départ": (0, 0), "Réception": (cr, ca), "Fin": (len(sr) - 1, len(sa) - 1)}
lignes, indices = [], []
for mom, (i, j) in moments.items():
    for ang in su.ANGULOS:
        vr, va = su.valor_en(sr, ang, i), su.valor_en(sa, ang, j)
        d = va - vr
        drap = abs(d) > seuil
        lignes.append({"Moment": mom, "Angle": ang, "Modèle (°)": round(vr), "Élève (°)": round(va),
                       "Écart (°)": round(d), "": "⚠️" if drap else "✅"})
        if drap:
            indices.append(f"**{mom} · {ang} ({d:+.0f}°) :** {su.pista(ang, mom, d)}")
st.markdown("#### Moments clés")
st.dataframe(pd.DataFrame(lignes), use_container_width=True, hide_index=True)

# ---- par phases (chemin DTW)
st.markdown("#### Écart moyen par phase")
phases = {"Montée et réception": [(i, j) for i, j in path if i <= cr],
          "Remontée": [(i, j) for i, j in path if i > cr]}
lignes_p = []
for phase, paires in phases.items():
    if not paires:
        continue
    ii, jj = np.array([p[0] for p in paires]), np.array([p[1] for p in paires])
    ligne = {"Phase": phase}
    for ang in su.ANGULOS:
        d = sa[ang].values[jj] - sr[ang].values[ii]
        ligne[f"{ang} (moy. ±)"] = f"{d.mean():+.0f}° (|{np.abs(d).mean():.0f}|)"
    lignes_p.append(ligne)
st.dataframe(pd.DataFrame(lignes_p), use_container_width=True, hide_index=True)
st.caption("Signe : élève − modèle. Entre parenthèses, l'erreur absolue moyenne.")

st.markdown("#### 🧠 Pistes pour l'entraîneur")
if indices:
    for p in indices:
        st.markdown(f"- {p}")
else:
    st.success(f"Aucun écart ne dépasse {seuil}°.")
st.caption("Ce sont des écarts par rapport au modèle, pas des erreurs confirmées : les proportions et la mobilité "
           "changent d'un athlète à l'autre, et l'analyse 2D dépend de l'angle de la caméra.")

# --------------------------------------------------------------------- vidéo
st.subheader("3️⃣ Vidéo comparative")
cle_seg = (r0, r1, a0, a1, seuil)
if st.button("🎬 Générer la vidéo côte à côte"):
    with st.spinner("Rendu en cours..."):
        out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        barre = st.progress(0)
        su.render_comparacion(R["path"], R["kps"], df_r, lado_r, (r0, r1),
                              A["path"], A["kps"], df_a, lado_a, (a0, a1),
                              mapa, seuil, out, progreso=barre.progress)
        st.session_state["video_cmp"] = (cle_seg, vers_h264(out))
v = st.session_state.get("video_cmp")
if v and v[0] == cle_seg:
    st.video(v[1])
    st.caption("En rouge : segments du corps dont l'angle s'écarte de plus que le seuil à ce moment.")
