import base64
import hashlib
import os
import pickle
import shutil
import subprocess
import tempfile
import time
import uuid

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from ultralytics import YOLO

import snatch_utils as su

# ------------------------------------------------------------------ textes (fr / es)
TR = {
 "fr": {
  "titre": "🏋️ Comparateur de technique haltérophile",
  "intro": "Compare la technique d'un élève avec une vidéo modèle : angles articulaires, phases et vidéo synchronisée.",
  "ex_label": "Exercice", "ex_sn": "Squat Snatch", "ex_cj": "Épaulé-jeté (Clean & Jerk)",
  "reglages": "Réglages", "p_fast": "⚡ Rapide", "p_bal": "⚖️ Équilibré", "p_prec": "🎯 Précis",
  "preset_label": "Vitesse / précision",
  "preset_help": "Rapide : modèle léger, 1 image sur 2. Précis : plus lent mais plus fiable.",
  "seuil": "Seuil d'écart (degrés)",
  "cache_cap": "Une vidéo déjà analysée avec le même réglage est réutilisée instantanément.",
  "priv_cap": "🔒 Tes vidéos sont conservées {h} h sur le serveur pour reprendre ta session (garde le même lien), puis supprimées.",
  "clear": "🗑️ Effacer toutes mes vidéos",
  "filmer": "📹 **Comment filmer :** vue **de profil** (90° par rapport au plan du mouvement), caméra fixe à hauteur de hanche, à 3-4 m, corps entier visible et du même côté que la vidéo modèle.",
  "toggle": "📲 Compresser sur le téléphone avant l'envoi (bêta)",
  "toggle_help": "Réduit la vidéo dans le navigateur avant de l'envoyer : envoi bien plus rapide. Si ça ne marche pas, désactive l'option.",
  "retrouvee": "✅ Vidéo retrouvée : {nom}", "video_def": "vidéo", "suppr": "🗑️ Supprimer cette vidéo",
  "lourde": "Vidéo lourde ({mo:.0f} Mo) : le téléversement est long. Coupe-la à la seule tentative (10-15 s) et filme en 1080p à 30 images/s.",
  "v_modele": "Vidéo modèle", "own": "Utiliser ma propre vidéo modèle",
  "own_help": "Décoche pour utiliser la vidéo modèle intégrée à l'application.",
  "using": "Utilisation de `{p}` du dépôt", "up_model": "Téléverse la vidéo modèle",
  "v_eleve": "Vidéo de l'élève", "up_eleve": "Téléverse la vidéo de l'élève",
  "analyser": "🚀 Analyser les vidéos (extraire les poses)",
  "w_ref": "modèle", "w_al": "élève", "de_ref": "du modèle", "de_al": "de l'élève",
  "cached": "Vidéo {nom} : analyse déjà en mémoire ✅", "analysing": "Analyse de la vidéo {nom}...",
  "press": "Clique sur **Analyser les vidéos** pour extraire les poses.",
  "w_side": "⚠️ La vidéo {nom_de} ne semble pas filmée de profil (ratio épaules/buste = {rat:.2f}). Les angles 2D seront déformés : comparaison seulement indicative.",
  "w_cov": "⚠️ Dans la vidéo {nom_de}, le corps n'est bien détecté que dans {cob:.0%} des images (barre qui masque le corps, corps coupé ?).",
  "side_left": "gauche", "side_right": "droit",
  "side_cap": "Côté analysé → modèle : {m} · élève : {a}",
  "cut_title": "1️⃣ Découpe la tentative",
  "cut_text": "Garde seulement le moment entre le départ et la fin du mouvement, debout avec la barre en haut (enlève les célébrations, la barre lâchée, etc.).",
  "n_ref": "Modèle", "n_al": "Élève", "debut": "début", "fin": "fin",
  "too_short": "Le segment est trop court.",
  "too_long": "Segment trop long pour l'alignement (max. ~1200 images). Raccourcis-le.",
  "dtw": "Alignement des mouvements (DTW)...", "res_title": "2️⃣ Résultats",
  "undetected": "non détecté", "pts": "pts", "m_model": "modèle",
  "w_cj": "⚠️ L'épaulé ou le jeté n'a pas été repéré dans l'une des vidéos (barre jamais au-dessus de la tête, ou épaulé non détecté). Vérifie le découpage : la comparaison par moments clés sera partielle.",
  "cap_cj": "Position de chaque moment dans le mouvement, élève vs modèle (en %). Vu de profil, un seul côté du corps est visible : pour le jeté en fente, l'angle du genou dépend de la jambe visible (avant ou arrière).",
  "ph_pull": "Tirage et épaulé", "ph_dipjerk": "Dip et jeté", "ph_final": "Remontée finale",
  "ph_full": "Mouvement complet", "ph_up": "Montée et réception", "ph_rec": "Remontée",
  "m_rec_model": "Réception du modèle", "m_rec_al": "Réception de l'élève", "m_of_mov": "{x} du mouvement",
  "m_timing": "Écart de timing",
  "cap_time": "Le temps est comparé en proportion (pas en secondes) car le modèle peut être au ralenti.",
  "c_model": "Modèle", "c_al": "Élève", "h_key": "Moments clés",
  "col_moment": "Moment", "col_angle": "Angle", "col_model": "Modèle (°)", "col_al": "Élève (°)", "col_diff": "Écart (°)",
  "h_phase": "Écart moyen par phase", "col_phase": "Phase", "col_avg": "moy. ±",
  "cap_sign": "Signe : élève − modèle. Entre parenthèses, l'erreur absolue moyenne.",
  "h_tips": "#### 🧠 Pistes pour l'entraîneur", "no_diff": "Aucun écart ne dépasse {seuil}°.",
  "disclaimer": "Ce sont des écarts par rapport au modèle, pas des erreurs confirmées : les proportions et la mobilité changent d'un athlète à l'autre, et l'analyse 2D dépend de l'angle de la caméra.",
  "video_title": "3️⃣ Vidéo comparative", "gen_video": "🎬 Générer la vidéo côte à côte",
  "rendering": "Rendu en cours...",
  "red_cap": "En rouge : segments du corps dont l'angle s'écarte de plus que le seuil à ce moment.",
 },
 "es": {
  "titre": "🏋️ Comparador de técnica de halterofilia",
  "intro": "Compara la técnica de un alumno con un vídeo modelo: ángulos articulares, fases y vídeo sincronizado.",
  "ex_label": "Ejercicio", "ex_sn": "Arrancada (squat snatch)", "ex_cj": "Cargada y envión (Clean & Jerk)",
  "reglages": "Ajustes", "p_fast": "⚡ Rápido", "p_bal": "⚖️ Equilibrado", "p_prec": "🎯 Preciso",
  "preset_label": "Velocidad / precisión",
  "preset_help": "Rápido: modelo ligero, 1 imagen de cada 2. Preciso: más lento pero más fiable.",
  "seuil": "Umbral de diferencia (grados)",
  "cache_cap": "Un vídeo ya analizado con el mismo ajuste se reutiliza al instante.",
  "priv_cap": "🔒 Tus vídeos se conservan {h} h en el servidor para retomar tu sesión (guarda el mismo enlace) y después se borran.",
  "clear": "🗑️ Borrar todos mis vídeos",
  "filmer": "📹 **Cómo grabar:** vista **de perfil** (90° respecto al plano del movimiento), cámara fija a la altura de la cadera, a 3-4 m, cuerpo entero visible y del mismo lado que el vídeo modelo.",
  "toggle": "📲 Comprimir en el móvil antes de enviar (beta)",
  "toggle_help": "Reduce el vídeo en el navegador antes de enviarlo: el envío es mucho más rápido. Si no funciona, desactiva la opción.",
  "retrouvee": "✅ Vídeo recuperado: {nom}", "video_def": "vídeo", "suppr": "🗑️ Eliminar este vídeo",
  "lourde": "Vídeo pesado ({mo:.0f} MB): la subida es lenta. Recórtalo a la tentativa (10-15 s) y graba en 1080p a 30 imágenes/s.",
  "v_modele": "Vídeo modelo", "own": "Usar mi propio vídeo modelo",
  "own_help": "Desmarca para usar el vídeo modelo integrado en la aplicación.",
  "using": "Usando `{p}` del repositorio", "up_model": "Sube el vídeo modelo",
  "v_eleve": "Vídeo del alumno", "up_eleve": "Sube el vídeo del alumno",
  "analyser": "🚀 Analizar los vídeos (extraer las poses)",
  "w_ref": "modelo", "w_al": "alumno", "de_ref": "del modelo", "de_al": "del alumno",
  "cached": "Vídeo {nom}: análisis ya en memoria ✅", "analysing": "Analizando el vídeo {nom}...",
  "press": "Pulsa **Analizar los vídeos** para extraer las poses.",
  "w_side": "⚠️ El vídeo {nom_de} no parece grabado de perfil (ratio hombros/torso = {rat:.2f}). Los ángulos 2D estarán distorsionados: comparación solo orientativa.",
  "w_cov": "⚠️ En el vídeo {nom_de}, el cuerpo solo se detecta bien en el {cob:.0%} de los fotogramas (¿la barra tapa el cuerpo o está cortado?).",
  "side_left": "izquierdo", "side_right": "derecho",
  "side_cap": "Lado analizado → modelo: {m} · alumno: {a}",
  "cut_title": "1️⃣ Recorta el intento",
  "cut_text": "Deja solo el momento entre la salida y el final del movimiento, de pie con la barra arriba (quita celebraciones, la barra soltada, etc.).",
  "n_ref": "Modelo", "n_al": "Alumno", "debut": "inicio", "fin": "fin",
  "too_short": "El segmento es demasiado corto.",
  "too_long": "Segmento demasiado largo para alinear (máx. ~1200 fotogramas). Recórtalo más.",
  "dtw": "Alineando los movimientos (DTW)...", "res_title": "2️⃣ Resultados",
  "undetected": "no detectado", "pts": "pts", "m_model": "modelo",
  "w_cj": "⚠️ No se ha detectado la cargada o el envión en uno de los vídeos (la barra nunca sube sobre la cabeza, o no se detecta la cargada). Revisa el recorte: la comparación por momentos clave será parcial.",
  "cap_cj": "Posición de cada momento dentro del movimiento, alumno vs modelo (en %). De perfil solo se ve un lado del cuerpo: en el envión en tijera, el ángulo de la rodilla depende de la pierna visible (delantera o trasera).",
  "ph_pull": "Tirón y cargada", "ph_dipjerk": "Dip y envión", "ph_final": "Recuperación final",
  "ph_full": "Movimiento completo", "ph_up": "Subida y recepción", "ph_rec": "Levantada",
  "m_rec_model": "Recepción del modelo", "m_rec_al": "Recepción del alumno", "m_of_mov": "{x} del movimiento",
  "m_timing": "Diferencia de timing",
  "cap_time": "El tiempo se compara en proporción (no en segundos) porque el modelo puede estar en cámara lenta.",
  "c_model": "Modelo", "c_al": "Alumno", "h_key": "Momentos clave",
  "col_moment": "Momento", "col_angle": "Ángulo", "col_model": "Modelo (°)", "col_al": "Alumno (°)", "col_diff": "Dif. (°)",
  "h_phase": "Diferencia media por fase", "col_phase": "Fase", "col_avg": "media ±",
  "cap_sign": "Signo: alumno − modelo. Entre paréntesis, el error absoluto medio.",
  "h_tips": "#### 🧠 Pistas para el entrenador", "no_diff": "Ninguna diferencia supera {seuil}°.",
  "disclaimer": "Son diferencias respecto al modelo, no errores confirmados: las proporciones y la movilidad cambian de un atleta a otro, y el análisis 2D depende del ángulo de la cámara.",
  "video_title": "3️⃣ Vídeo comparativo", "gen_video": "🎬 Generar el vídeo lado a lado",
  "rendering": "Renderizando...",
  "red_cap": "En rojo: segmentos del cuerpo cuyo ángulo se desvía más del umbral en ese momento.",
 },
}


REF_SNATCH = "referencia_snatch.mp4"  # vidéos modèle optionnelles, à mettre à la racine du dépôt
REF_CJ = "referencia_cj.mp4"

st.set_page_config(page_title="Comparateur / Comparador · Snatch & Clean-Jerk", page_icon="🏋️", layout="wide")

LANGS = {"fr": "🇫🇷 Français", "es": "🇪🇸 Español"}
_l = str(st.query_params.get("lang", "fr"))
_l = _l if _l in LANGS else "fr"
LANG = st.radio("Langue / Idioma", list(LANGS), index=list(LANGS).index(_l), format_func=LANGS.get,
                horizontal=True, key="lang_ui", label_visibility="collapsed")
if st.query_params.get("lang") != LANG:
    st.query_params["lang"] = LANG


def t(cle, **kw):
    txt = TR[LANG][cle]
    return txt.format(**kw) if kw else txt


st.title(t("titre"))
st.write(t("intro"))

# ------------------------------------------------- stockage temporaire (reprise de session)
BASE = os.path.join(tempfile.gettempdir(), "comparateur_snatch")
SESS_ROOT, CACHE_DIR = os.path.join(BASE, "sessions"), os.path.join(BASE, "poses")
os.makedirs(SESS_ROOT, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
DUREE_H = 24  # les vidéos et analyses sont supprimées du serveur après 24 h


def nettoyer():
    limite = time.time() - DUREE_H * 3600
    for racine in (SESS_ROOT, CACHE_DIR):
        for nom in os.listdir(racine):
            p = os.path.join(racine, nom)
            try:
                if os.path.getmtime(p) < limite:
                    shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)
            except OSError:
                pass


nettoyer()
sid = "".join(c for c in str(st.query_params.get("s", "")) if c.isalnum())[:16]
if not sid:
    sid = uuid.uuid4().hex[:10]
    st.query_params["s"] = sid  # le lien garde l'identifiant : en le rouvrant, on retrouve ses vidéos
SESS = os.path.join(SESS_ROOT, sid)
os.makedirs(SESS, exist_ok=True)

exercice = st.radio(t("ex_label"), ["sn", "cj"], format_func=lambda k: t("ex_" + k), horizontal=True)
cj = exercice == "cj"

with st.sidebar:
    st.header(t("reglages"))
    PRESETS = {"fast": ("yolov8n-pose.pt", 320, 2),
               "bal": ("yolov8s-pose.pt", 480, 1),
               "prec": ("yolov8m-pose.pt", 640, 1)}
    preset = st.selectbox(t("preset_label"), list(PRESETS), index=0, format_func=lambda k: t("p_" + k),
                          help=t("preset_help"))
    modele_nom, imgsz, saut = PRESETS[preset]
    seuil = st.slider(t("seuil"), 5, 30, 15)
    st.caption(t("cache_cap"))
    st.caption(t("priv_cap", h=DUREE_H))
    if st.button(t("clear")):
        shutil.rmtree(SESS, ignore_errors=True)
        os.makedirs(SESS, exist_ok=True)
        for k in list(st.session_state.keys()):
            if k.startswith("datos_") or k == "video_cmp":
                del st.session_state[k]
        st.rerun()

st.info(t("filmer"))


# ------------------------------------------------------------------ fonctions
@st.cache_resource
def charger_modele(nom):
    return YOLO(nom)


_DOSSIER_COMP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "compresseur")
try:
    if os.path.isfile(os.path.join(_DOSSIER_COMP, "index.html")):
        _compresseur = components.declare_component("compresseur_video", path=_DOSSIER_COMP)
    else:
        _compresseur = None  # dossier absent : l'option de compression est simplement masquée
except Exception:
    _compresseur = None


def normaliser(src, dest):
    """Remet la vidéo compressée par le navigateur en MP4 H.264 à 30 images/s constantes."""
    if shutil.which("ffmpeg") is not None:
        r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-r", "30",
                            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-c:v", "libx264",
                            "-pix_fmt", "yuv420p", "-an", dest])
        if r.returncode == 0 and os.path.exists(dest):
            return True
    shutil.copy(src, dest)
    return False


def recevoir_compresse(titre, role, cle_widget):
    """Vidéo compressée dans le navigateur du téléphone, puis reçue ici (bêta)."""
    val = _compresseur(label=titre, max_side=720, bitrate=2500000, lang=LANG, key=f"comp_{cle_widget}", default=None)
    if not val or not val.get("data"):
        return None
    dest = os.path.join(SESS, f"{role}.mp4")
    meta = dest + ".meta"
    marque = f"{val['name']}|compresse{val['size']}"
    if not (os.path.exists(dest) and os.path.exists(meta) and open(meta, encoding="utf-8").read() == marque):
        brut = dest + ".brut"
        with open(brut, "wb") as out:
            out.write(base64.b64decode(val["data"]))
        normaliser(brut, dest)
        os.remove(brut)
        with open(meta, "w", encoding="utf-8") as m:
            m.write(marque)
    return dest


def sauver_upload(f, role):
    dest = os.path.join(SESS, f"{role}.mp4")
    meta = dest + ".meta"
    marque = f"{f.name}|{f.size}"
    deja = os.path.exists(dest) and os.path.exists(meta) and open(meta, encoding="utf-8").read() == marque
    if not deja:
        with open(dest, "wb") as out:
            out.write(f.getbuffer())
        with open(meta, "w", encoding="utf-8") as m:
            m.write(marque)
    return dest


def supprimer_video(role):
    for suf in ("", ".meta"):
        p = os.path.join(SESS, f"{role}.mp4{suf}")
        if os.path.exists(p):
            os.remove(p)


def zone_upload(titre, role, cle_widget):
    """Téléversement + reprise : une vidéo déjà envoyée est retrouvée si on revient avec le même lien."""
    auto = False
    if _compresseur is not None:
        auto = st.toggle(t("toggle"), key=f"auto_{cle_widget}", help=t("toggle_help"))
    path = None
    if auto:
        path = recevoir_compresse(titre, role, cle_widget)
    else:
        f = st.file_uploader(titre, type=["mp4", "mov", "avi"], key=cle_widget)
        if f:
            path = sauver_upload(f, role)
    if path is None:
        path = os.path.join(SESS, f"{role}.mp4")
        if not os.path.exists(path):
            return None
        meta = path + ".meta"
        nom = open(meta, encoding="utf-8").read().split("|")[0] if os.path.exists(meta) else t("video_def")
        st.success(t("retrouvee", nom=nom))
        if st.button(t("suppr"), key=f"del_{role}"):
            supprimer_video(role)
            st.rerun()
    mo = os.path.getsize(path) / 1e6
    if mo > 80:
        st.warning(t("lourde", mo=mo))
    return path


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


def chemin_cache(cle):
    return os.path.join(CACHE_DIR, hashlib.md5(repr(cle).encode()).hexdigest() + ".pkl")


def charger_poses(cle):
    p = chemin_cache(cle)
    if os.path.exists(p):
        try:
            with open(p, "rb") as fh:
                return pickle.load(fh)
        except Exception:
            return None
    return None


def sauver_poses(cle, d):
    with open(chemin_cache(cle), "wb") as fh:
        pickle.dump({k: v for k, v in d.items() if k not in ("path", "cle")}, fh)


def empreinte(path):
    """Identifiant rapide d'un fichier (taille + début + fin) pour réutiliser une analyse déjà faite."""
    h = hashlib.md5()
    size = os.path.getsize(path)
    h.update(str(size).encode())
    with open(path, "rb") as fh:
        h.update(fh.read(1 << 20))
        fh.seek(max(size - (1 << 20), 0))
        h.update(fh.read(1 << 20))
    return h.hexdigest()


@st.cache_data(show_spinner=False)
def lire_image(path, idx, cle):
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
    st.subheader(t("v_modele"))
    REF_PATH = REF_CJ if cj else REF_SNATCH
    existe = os.path.exists(REF_PATH)
    perso = st.checkbox(t("own"), value=not existe, disabled=not existe, help=t("own_help"))
    if existe and not perso:
        path_ref = REF_PATH
        st.success(t("using", p=REF_PATH))
    else:
        path_ref = zone_upload(t("up_model"), "ref_cj" if cj else "ref_sn", f"ref_{int(cj)}")
with col2:
    st.subheader(t("v_eleve"))
    path_al = zone_upload(t("up_eleve"), "al", "al")

if not (path_ref and path_al):
    st.stop()

CL = {role: (empreinte(path), modele_nom, imgsz, saut) for role, path in (("ref", path_ref), ("al", path_al))}


def reprendre_depuis_cache():
    """Recharge les poses déjà calculées (même vidéo, mêmes réglages), sans les recalculer."""
    for role, path in (("ref", path_ref), ("al", path_al)):
        d = st.session_state.get(f"datos_{role}")
        if d and d.get("cle") == CL[role]:
            continue
        c = charger_poses(CL[role])
        if c is None:
            return False
        c["path"], c["cle"] = path, CL[role]
        st.session_state[f"datos_{role}"] = c
    return True


reprendre_depuis_cache()

if st.button(t("analyser"), type="primary"):
    modele = None
    for role, path, nom in (("ref", path_ref, t("w_ref")), ("al", path_al, t("w_al"))):
        cle = CL[role]
        d = charger_poses(cle)
        if d is not None:
            st.write(t("cached", nom=nom))
        else:
            if modele is None:
                modele = charger_modele(modele_nom)
            st.write(t("analysing", nom=nom))
            barre = st.progress(0)
            d = extraire_keypoints(path, modele, imgsz, saut, barre)
            sauver_poses(cle, d)
        d["path"], d["cle"] = path, cle
        st.session_state[f"datos_{role}"] = d

R, A = st.session_state.get("datos_ref"), st.session_state.get("datos_al")
if not (R and A and R["path"] == path_ref and A["path"] == path_al
        and R.get("cle") == CL["ref"] and A.get("cle") == CL["al"]):
    st.warning(t("press"))
    st.stop()

# ---------------------------------------------------------- métriques et alertes
lado_r, lado_a = su.lado_visible(R["kps"]), su.lado_visible(A["kps"])
df_r, df_a = su.calcular_metricas(R["kps"], lado_r), su.calcular_metricas(A["kps"], lado_a)
nom_cote = {"izquierdo": t("side_left"), "derecho": t("side_right")}

for nom_de, d, lado in ((t("de_ref"), R, lado_r), (t("de_al"), A, lado_a)):
    rat, cob = su.ratio_lateral(d["kps"], lado), su.cobertura(d["kps"], lado)
    if rat is not None and rat > 0.45:
        st.warning(t("w_side", nom_de=nom_de, rat=rat))
    if cob < 0.7:
        st.warning(t("w_cov", nom_de=nom_de, cob=cob))
st.caption(t("side_cap", m=nom_cote[lado_r], a=nom_cote[lado_a]))

# ------------------------------------------------------------------- découpage
st.subheader(t("cut_title"))
st.write(t("cut_text"))
c1, c2 = st.columns(2)
segs = {}
for col, role, d, nom in ((c1, "ref", R, t("n_ref")), (c2, "al", A, t("n_al"))):
    with col:
        n, fps = len(d["kps"]), d["fps"]
        dur = n / fps
        t0, t1 = st.slider(f"{nom} (s)", 0.0, float(round(dur, 1)), (0.0, float(round(dur, 1))), 0.1, key=f"s_{role}")
        i0, i1 = int(t0 * fps), min(n - 1, int(t1 * fps))
        segs[role] = (i0, i1)
        a, b = st.columns(2)
        a.image(lire_image(d["path"], i0, empreinte(d["path"])), caption=t("debut"), use_container_width=True)
        b.image(lire_image(d["path"], i1, empreinte(d["path"])), caption=t("fin"), use_container_width=True)

(r0, r1), (a0, a1) = segs["ref"], segs["al"]
if r1 - r0 < 15 or a1 - a0 < 15:
    st.error(t("too_short"))
    st.stop()
if (r1 - r0) > 1200 or (a1 - a0) > 1200:
    st.error(t("too_long"))
    st.stop()

sr, sa = df_r.iloc[r0:r1 + 1].reset_index(drop=True), df_a.iloc[a0:a1 + 1].reset_index(drop=True)

# ---------------------------------------------------------------- alignement
with st.spinner(t("dtw")):
    path = su.alinear_dtw(sr, sa, cj)
    mapa = su.mapa_alumno_a_ref(path, len(sa))
st.subheader(t("res_title"))
if cj:
    ev_r, ev_a = su.detectar_epaule_jete(sr), su.detectar_epaule_jete(sa)
    pts = [("cc", "Réception épaulé"), ("dip", "Dip"), ("jc", "Réception jeté")]  # clés internes (fr)
    cols = st.columns(3)
    moments = {"Position de départ": (0, 0)}
    for col, (k, nom) in zip(cols, pts):
        if ev_r[k] is not None and ev_a[k] is not None:
            pr, pa = ev_r[k] / len(sr), ev_a[k] / len(sa)
            col.metric(su.nom_moment(nom, LANG), f"{pa:.0%} ({t('m_model')} {pr:.0%})",
                       f"{(pa - pr) * 100:+.0f} {t('pts')}", delta_color="off")
            moments[nom] = (ev_r[k], ev_a[k])
        else:
            col.metric(su.nom_moment(nom, LANG), t("undetected"))
    moments["Fin"] = (len(sr) - 1, len(sa) - 1)
    if ev_r["cc"] is None or ev_a["cc"] is None or ev_r["jc"] is None or ev_a["jc"] is None:
        st.warning(t("w_cj"))
    st.caption(t("cap_cj"))
    coupes = [v for v in (ev_r["cc"], ev_r["jc"]) if v is not None]
    if len(coupes) == 2:
        phases = {t("ph_pull"): [(i, j) for i, j in path if i <= coupes[0]],
                  t("ph_dipjerk"): [(i, j) for i, j in path if coupes[0] < i <= coupes[1]],
                  t("ph_final"): [(i, j) for i, j in path if i > coupes[1]]}
    else:
        phases = {t("ph_full"): list(path)}
else:
    cr, ca = su.detectar_recepcion(sr), su.detectar_recepcion(sa)
    m1, m2, m3 = st.columns(3)
    m1.metric(t("m_rec_model"), t("m_of_mov", x=f"{cr / len(sr):.0%}"))
    m2.metric(t("m_rec_al"), t("m_of_mov", x=f"{ca / len(sa):.0%}"))
    m3.metric(t("m_timing"), f"{(ca / len(sa) - cr / len(sr)) * 100:+.0f} {t('pts')}")
    st.caption(t("cap_time"))
    moments = {"Position de départ": (0, 0), "Réception": (cr, ca), "Fin": (len(sr) - 1, len(sa) - 1)}
    phases = {t("ph_up"): [(i, j) for i, j in path if i <= cr],
              t("ph_rec"): [(i, j) for i, j in path if i > cr]}

# ---- courbes alignées
dfp = pd.DataFrame(path, columns=["i", "j"]).groupby("i")["j"].mean()
x = np.arange(len(sr))
courbes = st.columns(2)
for n, ang in enumerate(su.ANGULOS):
    al = np.interp(dfp.reindex(x).interpolate().bfill().ffill().values, np.arange(len(sa)), sa[ang].values)
    with courbes[n % 2]:
        st.markdown(f"**{su.nom_angle(ang, LANG)}** (°)")
        st.line_chart(pd.DataFrame({t("c_model"): sr[ang].values, t("c_al"): al}, index=x / (len(sr) - 1) * 100),
                      height=200)

# ---- tableau des moments clés
lignes, indices = [], []
for mom, (i, j) in moments.items():
    for ang in su.ANGULOS:
        vr, va = su.valor_en(sr, ang, i), su.valor_en(sa, ang, j)
        d = va - vr
        drap = abs(d) > seuil
        lignes.append({t("col_moment"): su.nom_moment(mom, LANG), t("col_angle"): su.nom_angle(ang, LANG),
                       t("col_model"): round(vr), t("col_al"): round(va),
                       t("col_diff"): round(d), "": "⚠️" if drap else "✅"})
        if drap:
            sep = " : " if LANG == "fr" else ": "
            indices.append(f"**{su.nom_moment(mom, LANG)} · {su.nom_angle(ang, LANG)} ({d:+.0f}°){sep}**"
                           f"{su.pista(ang, mom, d, LANG)}")
st.markdown("#### " + t("h_key"))
st.dataframe(pd.DataFrame(lignes), use_container_width=True, hide_index=True)

# ---- par phases (chemin DTW)
st.markdown("#### " + t("h_phase"))
lignes_p = []
for phase, paires in phases.items():
    if not paires:
        continue
    ii, jj = np.array([p[0] for p in paires]), np.array([p[1] for p in paires])
    ligne = {t("col_phase"): phase}
    for ang in su.ANGULOS:
        d = sa[ang].values[jj] - sr[ang].values[ii]
        ligne[f"{su.nom_angle(ang, LANG)} ({t('col_avg')})"] = f"{d.mean():+.0f}° (|{np.abs(d).mean():.0f}|)"
    lignes_p.append(ligne)
st.dataframe(pd.DataFrame(lignes_p), use_container_width=True, hide_index=True)
st.caption(t("cap_sign"))

st.markdown(t("h_tips"))
if indices:
    for p in indices:
        st.markdown(f"- {p}")
else:
    st.success(t("no_diff", seuil=seuil))
st.caption(t("disclaimer"))

# --------------------------------------------------------------------- vidéo
st.subheader(t("video_title"))
cle_seg = (r0, r1, a0, a1, seuil, LANG)
if st.button(t("gen_video")):
    with st.spinner(t("rendering")):
        out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        barre = st.progress(0)
        su.render_comparacion(R["path"], R["kps"], df_r, lado_r, (r0, r1),
                              A["path"], A["kps"], df_a, lado_a, (a0, a1),
                              mapa, seuil, out, progreso=barre.progress, lang=LANG)
        st.session_state["video_cmp"] = (cle_seg, vers_h264(out))
v = st.session_state.get("video_cmp")
if v and v[0] == cle_seg:
    st.video(v[1])
    st.caption(t("red_cap"))
