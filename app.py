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

REF_SNATCH = "referencia_snatch.mp4"  # vidéos modèle optionnelles, à mettre à la racine du dépôt
REF_CJ = "referencia_cj.mp4"

st.set_page_config(page_title="Comparateur Snatch & Épaulé-jeté", page_icon="🏋️", layout="wide")
st.title("🏋️ Comparateur de technique haltérophile")
st.write("Compare la technique d'un élève avec une vidéo modèle : angles articulaires, phases et vidéo synchronisée.")

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



exercice = st.radio("Exercice", ["Squat Snatch", "Épaulé-jeté (Clean & Jerk)"], horizontal=True)
cj = exercice.startswith("Épaulé")

with st.sidebar:
    st.header("Réglages")
    PRESETS = {"⚡ Rapide": ("yolov8n-pose.pt", 320, 2),
               "⚖️ Équilibré": ("yolov8s-pose.pt", 480, 1),
               "🎯 Précis": ("yolov8m-pose.pt", 640, 1)}
    preset = st.selectbox("Vitesse / précision", list(PRESETS), index=0,
                          help="Rapide : modèle léger, 1 image sur 2. Précis : plus lent mais plus fiable.")
    modele_nom, imgsz, saut = PRESETS[preset]
    seuil = st.slider("Seuil d'écart (degrés)", 5, 30, 15)
    st.caption("Une vidéo déjà analysée avec le même réglage est réutilisée instantanément.")
    st.caption(f"🔒 Tes vidéos sont conservées {DUREE_H} h sur le serveur pour reprendre ta session "
               "(garde le même lien), puis supprimées.")
    if st.button("🗑️ Effacer toutes mes vidéos"):
        shutil.rmtree(SESS, ignore_errors=True)
        os.makedirs(SESS, exist_ok=True)
        for k in list(st.session_state.keys()):
            if k.startswith("datos_") or k == "video_cmp":
                del st.session_state[k]
        st.rerun()

st.info("📹 **Comment filmer :** vue **de profil** (90° par rapport au plan du mouvement), caméra fixe à hauteur "
        "de hanche, à 3-4 m, corps entier visible et du même côté que la vidéo modèle.")


# ------------------------------------------------------------------ fonctions
@st.cache_resource
def charger_modele(nom):
    return YOLO(nom)


_compresseur = components.declare_component(
    "compresseur_video", path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "compresseur"))


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
    val = _compresseur(label=titre, max_side=720, bitrate=2500000, key=f"comp_{cle_widget}", default=None)
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
    auto = st.toggle("📲 Compresser sur le téléphone avant l'envoi (bêta)", key=f"auto_{cle_widget}",
                     help="Réduit la vidéo dans le navigateur avant de l'envoyer : envoi bien plus rapide. "
                          "Si ça ne marche pas, désactive l'option.")
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
        nom = open(meta, encoding="utf-8").read().split("|")[0] if os.path.exists(meta) else "vidéo"
        st.success(f"✅ Vidéo retrouvée : {nom}")
        if st.button("🗑️ Supprimer cette vidéo", key=f"del_{role}"):
            supprimer_video(role)
            st.rerun()
    mo = os.path.getsize(path) / 1e6
    if mo > 80:
        st.warning(f"Vidéo lourde ({mo:.0f} Mo) : le téléversement est long. Coupe-la à la seule tentative "
                   "(10-15 s) et filme en 1080p à 30 images/s.")
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
    st.subheader("Vidéo modèle")
    REF_PATH = REF_CJ if cj else REF_SNATCH
    existe = os.path.exists(REF_PATH)
    perso = st.checkbox("Utiliser ma propre vidéo modèle", value=not existe, disabled=not existe,
                        help="Décoche pour utiliser la vidéo modèle intégrée à l'application.")
    if existe and not perso:
        path_ref = REF_PATH
        st.success(f"Utilisation de `{REF_PATH}` du dépôt")
    else:
        path_ref = zone_upload("Téléverse la vidéo modèle", "ref_cj" if cj else "ref_sn", f"ref_{int(cj)}")
with col2:
    st.subheader("Vidéo de l'élève")
    path_al = zone_upload("Téléverse la vidéo de l'élève", "al", "al")

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

if st.button("🚀 Analyser les vidéos (extraire les poses)", type="primary"):
    modele = None
    for role, path, nom in (("ref", path_ref, "modèle"), ("al", path_al, "élève")):
        cle = CL[role]
        d = charger_poses(cle)
        if d is not None:
            st.write(f"Vidéo {nom} : analyse déjà en mémoire ✅")
        else:
            if modele is None:
                modele = charger_modele(modele_nom)
            st.write(f"Analyse de la vidéo {nom}...")
            barre = st.progress(0)
            d = extraire_keypoints(path, modele, imgsz, saut, barre)
            sauver_poses(cle, d)
        d["path"], d["cle"] = path, cle
        st.session_state[f"datos_{role}"] = d

R, A = st.session_state.get("datos_ref"), st.session_state.get("datos_al")
if not (R and A and R["path"] == path_ref and A["path"] == path_al
        and R.get("cle") == CL["ref"] and A.get("cle") == CL["al"]):
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
        a.image(lire_image(d["path"], i0, empreinte(d["path"])), caption="début", use_container_width=True)
        b.image(lire_image(d["path"], i1, empreinte(d["path"])), caption="fin", use_container_width=True)

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
    path = su.alinear_dtw(sr, sa, cj)
    mapa = su.mapa_alumno_a_ref(path, len(sa))
st.subheader("2️⃣ Résultats")
if cj:
    ev_r, ev_a = su.detectar_epaule_jete(sr), su.detectar_epaule_jete(sa)
    pts = [("cc", "Réception épaulé"), ("dip", "Dip"), ("jc", "Réception jeté")]
    cols = st.columns(3)
    moments = {"Position de départ": (0, 0)}
    for col, (k, nom) in zip(cols, pts):
        if ev_r[k] is not None and ev_a[k] is not None:
            pr, pa = ev_r[k] / len(sr), ev_a[k] / len(sa)
            col.metric(nom, f"{pa:.0%} (modèle {pr:.0%})", f"{(pa - pr) * 100:+.0f} pts", delta_color="off")
            moments[nom] = (ev_r[k], ev_a[k])
        else:
            col.metric(nom, "non détecté")
    moments["Fin"] = (len(sr) - 1, len(sa) - 1)
    if ev_r["cc"] is None or ev_a["cc"] is None or ev_r["jc"] is None or ev_a["jc"] is None:
        st.warning("⚠️ L'épaulé ou le jeté n'a pas été repéré dans l'une des vidéos (barre jamais au-dessus de la tête, "
                   "ou épaulé non détecté). Vérifie le découpage : la comparaison par moments clés sera partielle.")
    st.caption("Position de chaque moment dans le mouvement, élève vs modèle (en %). Vu de profil, un seul côté du "
               "corps est visible : pour le jeté en fente, l'angle du genou dépend de la jambe visible (avant ou arrière).")
    coupes = [v for v in (ev_r["cc"], ev_r["jc"]) if v is not None]
    if len(coupes) == 2:
        phases = {"Tirage et épaulé": [(i, j) for i, j in path if i <= coupes[0]],
                  "Dip et jeté": [(i, j) for i, j in path if coupes[0] < i <= coupes[1]],
                  "Remontée finale": [(i, j) for i, j in path if i > coupes[1]]}
    else:
        phases = {"Mouvement complet": list(path)}
else:
    cr, ca = su.detectar_recepcion(sr), su.detectar_recepcion(sa)
    m1, m2, m3 = st.columns(3)
    m1.metric("Réception du modèle", f"{cr / len(sr):.0%} du mouvement")
    m2.metric("Réception de l'élève", f"{ca / len(sa):.0%} du mouvement")
    m3.metric("Écart de timing", f"{(ca / len(sa) - cr / len(sr)) * 100:+.0f} pts")
    st.caption("Le temps est comparé en proportion (pas en secondes) car le modèle peut être au ralenti.")
    moments = {"Position de départ": (0, 0), "Réception": (cr, ca), "Fin": (len(sr) - 1, len(sa) - 1)}
    phases = {"Montée et réception": [(i, j) for i, j in path if i <= cr],
              "Remontée": [(i, j) for i, j in path if i > cr]}

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
