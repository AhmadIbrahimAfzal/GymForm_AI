import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import torch
import torch.nn as nn
import numpy as np
from collections import deque, Counter
import warnings
import threading
import time
from exercises import BicepCurl, Squat, LateralRaise, ShoulderPress, TricepFinisher

warnings.filterwarnings("ignore", category=UserWarning)

st.set_page_config(
    page_title="GymForm AI — Smart Workout Coach",
    page_icon="🏋️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
.stApp { font-family: 'Inter', sans-serif; }
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a0a0a 0%, #1a1a2e 50%, #16213e 100%);
    border-right: 1px solid rgba(57,255,20,0.2);
}
.main-header { text-align:center; padding:0.3rem 0; margin-bottom:0.5rem; }
.main-header h1 {
    background: linear-gradient(135deg,#39FF14,#00D2FF);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    font-size:2rem; font-weight:800; margin:0; }
.stat-card {
    background: linear-gradient(135deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02));
    border:1px solid rgba(57,255,20,0.3); border-radius:10px;
    padding:0.5rem 0.8rem; text-align:center; margin-bottom:0.4rem;
    backdrop-filter:blur(10px); transition:all .3s ease;
    min-height:70px; display:flex; flex-direction:column; justify-content:center; }
.stat-card:hover { border-color:rgba(57,255,20,0.6); box-shadow:0 0 20px rgba(57,255,20,0.1); }
.stat-label { color:rgba(255,255,255,0.45); font-size:0.6rem; font-weight:700;
    text-transform:uppercase; letter-spacing:1.5px; margin-bottom:0.1rem; }
.stat-value { font-size:1.3rem; font-weight:800;
    background:linear-gradient(135deg,#39FF14,#00D2FF);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
.stat-value.good { background:linear-gradient(135deg,#39FF14,#00FF88);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
.stat-value.bad { background:linear-gradient(135deg,#FF4444,#FF6B6B);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
.stat-value.neutral { background:linear-gradient(135deg,#FFD700,#FFA500);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
.section-hdr { color:rgba(255,255,255,0.4); font-size:0.68rem; font-weight:700;
    text-transform:uppercase; letter-spacing:2px; margin:1.2rem 0 0.6rem;
    padding-bottom:0.4rem; border-bottom:1px solid rgba(255,255,255,0.1); }
[data-testid="stAppViewBlockContainer"] { padding-top:0.2rem !important; padding-bottom:0 !important; }
.block-container { padding-top:0.2rem !important; padding-bottom:0 !important; }
[data-testid="column"] .stMarkdown { margin-bottom:0 !important; }
video { max-height:350px !important; object-fit:contain; }
#MainMenu {visibility:hidden;} footer {visibility:hidden;}
header {visibility:hidden;}
[data-testid="stStatusWidget"] {display:none;}
[data-testid="stBottom"] {display:none;}
</style>
""", unsafe_allow_html=True)

POSE_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),(9,10),
    (11,12),(11,13),(13,15),(15,17),(15,19),(15,21),(17,19),
    (12,14),(14,16),(16,18),(16,20),(16,22),(18,20),
    (11,23),(12,24),(23,24),
    (23,25),(24,26),(25,27),(26,28),(27,29),(28,30),
    (29,31),(30,32),(27,31),(28,32),
]
LABELS_MAP_REVERSE = {
    0:'Bad Curl',1:'Good Curl',2:'Bad Squat',3:'Good Squat',4:'Bad Raise',5:'Good Raise',
    6:'Bad Shoulder',7:'Good Shoulder',8:'Bad Tricep',9:'Good Tricep'
}
EXERCISES_MAP = {"Bicep Curl": BicepCurl, "Squat": Squat, "Lateral Raise": LateralRaise, "Shoulder Press": ShoulderPress, "Tricep Finisher": TricepFinisher}

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180 else angle

class GymModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(8, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 10),
        )
    def forward(self, x):
        return self.network(x)

@st.cache_resource
def load_model():
    m = GymModel()
    m.load_state_dict(torch.load('gym_model_fullbody.pt', weights_only=True))
    m.eval()
    return m

class GymCoachProcessor(VideoProcessorBase):
    def __init__(self):
        self._exercise_name = "Bicep Curl"
        self._active_exercise = BicepCurl()
        self.filtered_landmarks = []
        self.prediction_history = deque(maxlen=10)
        self.lock = threading.Lock()
        self.EMA_ALPHA = 0.5
        self.rep_count = 0
        self.stage = "down"
        self.form_text = "Waiting..."
        self.confidence = 0.0
        self.model = load_model()
        base_opts = mp_python.BaseOptions(model_asset_path='pose_landmarker_lite.task')
        opts = vision.PoseLandmarkerOptions(
            base_options=base_opts,
            running_mode=vision.RunningMode.IMAGE,
            output_segmentation_masks=False,
        )
        self.detector = vision.PoseLandmarker.create_from_options(opts)

    @property
    def exercise_name(self):
        return self._exercise_name

    @exercise_name.setter
    def exercise_name(self, value):
        if value != self._exercise_name:
            with self.lock:
                self._exercise_name = value
                self._active_exercise = EXERCISES_MAP[value]()
                self.rep_count = 0
                self.stage = "down"
                self.form_text = "Waiting..."
                self.confidence = 0.0
                self.prediction_history.clear()
                self.filtered_landmarks = []

    def _smoothed(self, pred):
        self.prediction_history.append(pred)
        return Counter(self.prediction_history).most_common(1)[0][0]

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        with self.lock:
            ex_name = self._exercise_name
            ex_obj  = self._active_exercise

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        try:
            result = self.detector.detect(mp_img)
        except Exception:
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        if not result.pose_landmarks:
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        lms = result.pose_landmarks[0]
        if not self.filtered_landmarks:
            self.filtered_landmarks = [[lm.x, lm.y, lm.z] for lm in lms]
        else:
            a = self.EMA_ALPHA
            for i, lm in enumerate(lms):
                self.filtered_landmarks[i][0] = a*lm.x + (1-a)*self.filtered_landmarks[i][0]
                self.filtered_landmarks[i][1] = a*lm.y + (1-a)*self.filtered_landmarks[i][1]
                self.filtered_landmarks[i][2] = a*lm.z + (1-a)*self.filtered_landmarks[i][2]

        fl = self.filtered_landmarks
        l_sh, l_el, l_wr = fl[11][:2], fl[13][:2], fl[15][:2]
        l_hi, l_kn, l_an = fl[23][:2], fl[25][:2], fl[27][:2]
        r_sh, r_el, r_wr = fl[12][:2], fl[14][:2], fl[16][:2]
        r_hi, r_kn, r_an = fl[24][:2], fl[26][:2], fl[28][:2]

        ang = {
            'l_elbow':    calculate_angle(l_sh, l_el, l_wr),
            'r_elbow':    calculate_angle(r_sh, r_el, r_wr),
            'l_shoulder': calculate_angle(l_hi, l_sh, l_el),
            'r_shoulder': calculate_angle(r_hi, r_sh, r_el),
            'l_hip':      calculate_angle(l_sh, l_hi, l_kn),
            'r_hip':      calculate_angle(r_sh, r_hi, r_kn),
            'l_knee':     calculate_angle(l_hi, l_kn, l_an),
            'r_knee':     calculate_angle(r_hi, r_kn, r_an),
        }

        # active arm detection
        l_el_vis = lms[13].visibility if len(lms) > 13 else 0
        r_el_vis = lms[14].visibility if len(lms) > 14 else 0
        ang['active_arm'] = 'left' if l_el_vis > r_el_vis else 'right'

        if ex_name in ("Bicep Curl", "Lateral Raise", "Shoulder Press", "Tricep Finisher"):
            ang['l_hip'] = ang['r_hip'] = ang['l_knee'] = ang['r_knee'] = 180.0
        elif ex_name == "Squat":
            ang['l_elbow'] = ang['r_elbow'] = ang['l_shoulder'] = ang['r_shoulder'] = 180.0

        tensor = torch.FloatTensor([[
            ang['l_elbow'], ang['r_elbow'], ang['l_shoulder'], ang['r_shoulder'],
            ang['l_hip'], ang['r_hip'], ang['l_knee'], ang['r_knee'],
        ]])
        with torch.no_grad():
            out = self.model(tensor)
            
            # mask irrelevant classes
            if ex_name == "Bicep Curl":
                out[0, 2:] = -float('inf')
            elif ex_name == "Squat":
                out[0, :2] = -float('inf')
                out[0, 4:] = -float('inf')
            elif ex_name == "Lateral Raise":
                out[0, :4] = -float('inf')
                out[0, 6:] = -float('inf')
            elif ex_name == "Shoulder Press":
                out[0, :6] = -float('inf')
                out[0, 8:] = -float('inf')
            elif ex_name == "Tricep Finisher":
                out[0, :8] = -float('inf')
                
            probs = torch.softmax(out, dim=1)[0]
            idx = torch.argmax(probs).item()
            conf = probs[idx].item() * 100

        predicted = LABELS_MAP_REVERSE[idx]
        smoothed  = self._smoothed(predicted)
        reps, stage = ex_obj.update(ang, smoothed)

        self.rep_count  = reps
        self.stage      = stage
        self.confidence = conf

        bad = set()
        if conf < 60:
            smoothed = "Tracking..."
            txt_c = (0, 255, 255); base_c = (0, 255, 255)
        elif 'Good' in smoothed:
            txt_c = (0, 255, 0); base_c = (0, 255, 0)
        else:
            txt_c = (0, 0, 255); base_c = (0, 255, 0)
            if ex_name == "Bicep Curl":
                if ang['l_shoulder'] > 40 or ang['r_shoulder'] > 40:
                    bad.update([(11,13),(12,14),(11,23),(12,24)])
                else:
                    bad.update([(11,13),(13,15),(12,14),(14,16)])
            elif ex_name == "Squat":
                if ang['l_hip'] < 70 or ang['r_hip'] < 70:
                    bad.update([(11,23),(12,24),(23,24)])
                elif stage == "down" and (ang['l_knee'] > 120 or ang['r_knee'] > 120):
                    bad.update([(23,25),(25,27),(24,26),(26,28)])
                else:
                    bad.update([(23,25),(25,27),(24,26),(26,28)])
            elif ex_name == "Lateral Raise":
                if ang['l_elbow'] < 140 or ang['r_elbow'] < 140:
                    bad.update([(11,13),(13,15),(12,14),(14,16)])
                elif ang['l_shoulder'] > 100 or ang['r_shoulder'] > 100:
                    bad.update([(11,13),(12,14),(11,12)])
                else:
                    bad.update([(11,13),(13,15),(12,14),(14,16)])
            elif ex_name == "Shoulder Press":
                if ang['l_hip'] < 160 or ang['r_hip'] < 160:
                    bad.update([(11,23),(12,24),(23,24)])
                else:
                    bad.update([(11,13),(13,15),(12,14),(14,16)])
            elif ex_name == "Tricep Finisher":
                if ang['l_shoulder'] > 45 and ang['r_shoulder'] > 45:
                    bad.update([(11,13),(12,14)])
                else:
                    bad.update([(11,13),(13,15),(12,14),(14,16)])

        self.form_text = smoothed



        h, w = img.shape[:2]
        for s, e in POSE_CONNECTIONS:
            if s >= len(fl) or e >= len(fl):
                continue
            if s <= 10 or e <= 10:
                continue
            if ex_name in ("Bicep Curl", "Lateral Raise", "Shoulder Press", "Tricep Finisher") and (s >= 25 or e >= 25):
                continue
            if ex_name == "Squat" and (13 <= s <= 22 or 13 <= e <= 22):
                continue
            p1 = (int(fl[s][0]*w), int(fl[s][1]*h))
            p2 = (int(fl[e][0]*w), int(fl[e][1]*h))
            is_bad = (s, e) in bad or (e, s) in bad
            cv2.line(img, p1, p2, (0,0,255) if is_bad else base_c,
                     6 if is_bad else 3, cv2.LINE_AA)

        for i, pt in enumerate(fl):
            if i <= 10:
                continue
            if ex_name in ("Bicep Curl", "Lateral Raise", "Shoulder Press", "Tricep Finisher") and i >= 25:
                continue
            if ex_name == "Squat" and 13 <= i <= 22:
                continue
            jbad = any(c[0]==i or c[1]==i for c in bad)
            cv2.circle(img, (int(pt[0]*w), int(pt[1]*h)), 5,
                       (0,0,255) if jbad else (255,255,255), -1)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


with st.sidebar:
    st.markdown('<div class="section-hdr">Exercise</div>', unsafe_allow_html=True)
    selected_exercise = st.radio(
        "Choose your workout",
        list(EXERCISES_MAP.keys()),
        index=0,
        label_visibility="collapsed",
    )

st.markdown("""
<div class="main-header">
    <h1>GymForm AI</h1>
</div>""", unsafe_allow_html=True)

_, vid_col, _ = st.columns([1, 2, 1])
with vid_col:
    ctx = webrtc_streamer(
        key="gym-coach",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=GymCoachProcessor,
        media_stream_constraints={"video": {"width": {"ideal": 480}, "height": {"ideal": 360}}, "audio": False},
        async_processing=True,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    )

stats_cols = st.columns(3)
reps_ph  = stats_cols[0].empty()
stage_ph = stats_cols[1].empty()
form_ph  = stats_cols[2].empty()

if ctx.video_processor:
    ctx.video_processor.exercise_name = selected_exercise

    while ctx.state.playing:
        p = ctx.video_processor 
        reps_ph.markdown(
            f'<div class="stat-card"><div class="stat-label">Reps</div>'
            f'<div class="stat-value">{p.rep_count}</div></div>',
            unsafe_allow_html=True,
        )
        stage_ph.markdown(
            f'<div class="stat-card"><div class="stat-label">Stage</div>'
            f'<div class="stat-value">{p.stage.upper()}</div></div>',
            unsafe_allow_html=True,
        )
        cls = "good" if "Good" in p.form_text else ("bad" if "Bad" in p.form_text else "neutral")
        form_ph.markdown(
            f'<div class="stat-card"><div class="stat-label">Form</div>'
            f'<div class="stat-value {cls}">{p.form_text} ({p.confidence:.0f}%)</div></div>',
            unsafe_allow_html=True,
        )
        time.sleep(0.2)
