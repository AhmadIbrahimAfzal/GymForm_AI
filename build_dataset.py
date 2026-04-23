import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import csv
import os
import numpy as np

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
    return angle

csv_file = 'dataset_fullbody.csv'

def create_csv_header():
    with open(csv_file, mode='w', newline='') as f:
        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        headers = ['label', 'l_elbow', 'r_elbow', 'l_shoulder', 'r_shoulder', 'l_hip', 'r_hip', 'l_knee', 'r_knee']
        csv_writer.writerow(headers)

def process_video_to_csv(video_path, label_name):
    if not os.path.exists(video_path):
        print(f"Error: Could not find {video_path}")
        return

    base_options = python.BaseOptions(model_asset_path='pose_landmarker_lite.task')
    options = vision.PoseLandmarkerOptions(
        base_options=base_options, running_mode=vision.RunningMode.IMAGE)
    detector = vision.PoseLandmarker.create_from_options(options)
    
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    with open(csv_file, mode='a', newline='') as f:
        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break 
                
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            detection_result = detector.detect(mp_image)
            
            if detection_result.pose_landmarks:
                landmarks = detection_result.pose_landmarks[0]
                
                l_sh = [landmarks[11].x, landmarks[11].y]
                l_el = [landmarks[13].x, landmarks[13].y]
                l_wr = [landmarks[15].x, landmarks[15].y]
                l_hi = [landmarks[23].x, landmarks[23].y]
                l_kn = [landmarks[25].x, landmarks[25].y]
                l_an = [landmarks[27].x, landmarks[27].y]
                
                r_sh = [landmarks[12].x, landmarks[12].y]
                r_el = [landmarks[14].x, landmarks[14].y]
                r_wr = [landmarks[16].x, landmarks[16].y]
                r_hi = [landmarks[24].x, landmarks[24].y]
                r_kn = [landmarks[26].x, landmarks[26].y]
                r_an = [landmarks[28].x, landmarks[28].y]

                le_ang = calculate_angle(l_sh, l_el, l_wr)
                re_ang = calculate_angle(r_sh, r_el, r_wr)
                ls_ang = calculate_angle(l_hi, l_sh, l_el) 
                rs_ang = calculate_angle(r_hi, r_sh, r_el)
                lh_ang = calculate_angle(l_sh, l_hi, l_kn)
                rh_ang = calculate_angle(r_sh, r_hi, r_kn)
                lk_ang = calculate_angle(l_hi, l_kn, l_an)
                rk_ang = calculate_angle(r_hi, r_kn, r_an)

                row = [label_name, le_ang, re_ang, ls_ang, rs_ang, lh_ang, rh_ang, lk_ang, rk_ang]
                csv_writer.writerow(row)
                frame_count += 1

    cap.release()
    print(f"Processed {video_path} -> {frame_count} frames saved.")

print("Building Full Body Dataset...")
create_csv_header()

process_video_to_csv('good_curl.mp4', 'Good Curl')
process_video_to_csv('bad_curl.mp4', 'Bad Curl')
process_video_to_csv('good_squat.mp4', 'Good Squat')
process_video_to_csv('bad_squat.mp4', 'Bad Squat')
process_video_to_csv('good_lat.mp4', 'Good Raise')
process_video_to_csv('bad_lat.mp4', 'Bad Raise')

print(f"Done! Saved to {csv_file}")