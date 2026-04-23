import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import torch
import torch.nn as nn
import warnings
import numpy as np
from collections import deque, Counter
from exercises import BicepCurl, Squat, LateralRaise
import tkinter as tk

warnings.filterwarnings("ignore", category=UserWarning)

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

prediction_history = deque(maxlen=10)
def get_smoothed_prediction(new_prediction):
    prediction_history.append(new_prediction)
    return Counter(prediction_history).most_common(1)[0][0]

class GymModel(nn.Module):
    def __init__(self):
        super(GymModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(8, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 6)
        )
    def forward(self, x):
        return self.network(x)

model = GymModel()
model.load_state_dict(torch.load('gym_model_fullbody.pt', weights_only=True))
model.eval()

base_options = python.BaseOptions(model_asset_path='pose_landmarker_lite.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    output_segmentation_masks=False)
detector = vision.PoseLandmarker.create_from_options(options)

exercises_map = {
    "Bicep Curl": BicepCurl(),
    "Squat": Squat(),
    "Lateral Raise": LateralRaise()
}

selected_exercise = "Bicep Curl"

def set_exercise(name):
    global selected_exercise
    selected_exercise = name
    root.quit()

root = tk.Tk()
root.title("Gym Coach")
root.geometry("400x250")
root.configure(bg="#2E2E2E")
root.eval('tk::PlaceWindow . center')

tk.Label(root, text="Select Workout", font=("Helvetica", 16, "bold"), fg="white", bg="#2E2E2E").pack(pady=20)

for ex_name in exercises_map.keys():
    btn = tk.Button(root, text=ex_name, font=("Helvetica", 14, "bold"), bg="#39FF14", fg="black", 
                    activebackground="#32CD32", width=20, relief=tk.FLAT,
                    command=lambda name=ex_name: set_exercise(name))
    btn.pack(pady=10)

root.mainloop()
try:
    root.destroy()
except:
    pass

current_exercise_name = selected_exercise 
active_exercise = exercises_map[current_exercise_name]

cap = cv2.VideoCapture(0)

filtered_landmarks = []
EMA_ALPHA = 0.5 

POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19), 
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (24, 26), (25, 27), (26, 28), (27, 29), (28, 30),
    (29, 31), (30, 32), (27, 31), (28, 32)
]

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    detection_result = detector.detect(mp_image)

    if detection_result.pose_landmarks:
        landmarks = detection_result.pose_landmarks[0]

        if not filtered_landmarks:
            filtered_landmarks = [[lm.x, lm.y, lm.z] for lm in landmarks]
        else:
            for i, lm in enumerate(landmarks):
                filtered_landmarks[i][0] = EMA_ALPHA * lm.x + (1 - EMA_ALPHA) * filtered_landmarks[i][0]
                filtered_landmarks[i][1] = EMA_ALPHA * lm.y + (1 - EMA_ALPHA) * filtered_landmarks[i][1]
                filtered_landmarks[i][2] = EMA_ALPHA * lm.z + (1 - EMA_ALPHA) * filtered_landmarks[i][2]

        l_sh, l_el, l_wr = [filtered_landmarks[11][:2], filtered_landmarks[13][:2], filtered_landmarks[15][:2]]
        l_hi, l_kn, l_an = [filtered_landmarks[23][:2], filtered_landmarks[25][:2], filtered_landmarks[27][:2]]
        r_sh, r_el, r_wr = [filtered_landmarks[12][:2], filtered_landmarks[14][:2], filtered_landmarks[16][:2]]
        r_hi, r_kn, r_an = [filtered_landmarks[24][:2], filtered_landmarks[26][:2], filtered_landmarks[28][:2]]

        angles_dict = {
            'l_elbow': calculate_angle(l_sh, l_el, l_wr),
            'r_elbow': calculate_angle(r_sh, r_el, r_wr), 
            'l_shoulder': calculate_angle(l_hi, l_sh, l_el),
            'r_shoulder': calculate_angle(r_hi, r_sh, r_el),
            'l_hip': calculate_angle(l_sh, l_hi, l_kn),
            'r_hip': calculate_angle(r_sh, r_hi, r_kn), 
            'l_knee': calculate_angle(l_hi, l_kn, l_an),
            'r_knee': calculate_angle(r_hi, r_kn, r_an)
        }

        X_live_tensor = torch.FloatTensor([[
            angles_dict['l_elbow'], angles_dict['r_elbow'], angles_dict['l_shoulder'], 
            angles_dict['r_shoulder'], angles_dict['l_hip'], angles_dict['r_hip'], 
            angles_dict['l_knee'], angles_dict['r_knee']
        ]])

        with torch.no_grad():
            outputs = model(X_live_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            class_idx = torch.argmax(probabilities).item()
            confidence = probabilities[class_idx].item() * 100
        
        labels_map_reverse = {0: 'Bad Curl', 1: 'Good Curl', 2: 'Bad Squat', 3: 'Good Squat', 4: 'Bad Raise', 5: 'Good Raise'}
        predicted_class = labels_map_reverse[class_idx]
        smoothed_class = get_smoothed_prediction(predicted_class)

        rep_count, current_stage = active_exercise.update(angles_dict, smoothed_class)

        if confidence < 60.0:
            smoothed_class = "Tracking..."
            color = (255, 255, 0)
        elif 'Good' in smoothed_class:
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (600, 150), (0, 0, 0), -1) 
        alpha = 0.6
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        cv2.putText(frame, f"EXERCISE: {current_exercise_name}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"REPS: {rep_count}   STAGE: {current_stage.upper()}", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(frame, f"FORM: {smoothed_class} ({confidence:.1f}%)", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3, cv2.LINE_AA)

        for connection in POSE_CONNECTIONS:
            start_idx, end_idx = connection
            if start_idx < len(filtered_landmarks) and end_idx < len(filtered_landmarks):
                start_point = (int(filtered_landmarks[start_idx][0] * frame.shape[1]), 
                               int(filtered_landmarks[start_idx][1] * frame.shape[0]))
                end_point = (int(filtered_landmarks[end_idx][0] * frame.shape[1]), 
                             int(filtered_landmarks[end_idx][1] * frame.shape[0]))
                cv2.line(frame, start_point, end_point, color, 3, cv2.LINE_AA)

        for flm in filtered_landmarks:
            x = int(flm[0] * frame.shape[1])
            y = int(flm[1] * frame.shape[0])
            cv2.circle(frame, (x, y), 4, (255, 255, 255), -1)

    cv2.imshow('Gym Coach', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()