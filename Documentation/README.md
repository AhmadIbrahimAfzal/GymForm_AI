# GymForm AI - Smart Workout Coach

GymForm AI is a real-time computer vision app built to track your reps and give you live feedback on your workout form. It uses your webcam to track your body mechanics, counts your reps automatically, and yells at you (visually) if you start cheating or using bad form.

## How It Works

We built this using **Streamlit** for the web interface, **MediaPipe** for the skeleton tracking, and a custom **PyTorch** neural network to classify the exercises. 

Here is what the pipeline actually does under the hood:
1. **Live Tracking:** Streamlit captures your webcam feed and passes it to MediaPipe, which plots 33 3D landmarks on your body.
2. **Angle Calculation:** We calculate 8 core joint angles (Left/Right Elbows, Shoulders, Hips, and Knees) to figure out exactly what your body is doing.
3. **Form Detection:** Those 8 angles are fed into our custom PyTorch neural network. The network was trained on our own dataset (`dataset_fullbody.csv`) to recognize what "Good" and "Bad" form looks like for each exercise.
4. **Rep Counting:** We use a state machine (`exercises.py`) to count reps. It's smart enough to know that both arms need to be moving in sync (so you can't cheat by flapping one arm), and if you use bad form at *any* point during the rep, that rep is invalidated.

## Supported Exercises

The model currently tracks 5 different exercises:
*   **Bicep Curls:** Tracks your elbows and watches out for swinging shoulders.
*   **Squats:** Tracks your hips and knees to make sure you are getting enough depth and not leaning too far forward.
*   **Lateral Raises:** Tracks shoulder and elbow angles. 
*   **Shoulder Press:** Tracks elbows and makes sure you aren't leaning your hips back dangerously to push the weight up.
*   **Tricep Finisher (Extensions/Kickbacks):** Dynamically tracks whichever arm is currently facing the camera to monitor tricep isolation.

## Project Structure

*   `app.py`: The main Streamlit web app. Handles the UI, webcam streaming, overlaying the skeleton on your feed, and hooking into the PyTorch model.
*   `exercises.py`: The logic for rep counting. This is where the state machines live that decide what counts as a full range-of-motion rep for each exercise.
*   `train_model.py`: The script we used to build and train the PyTorch Neural Network (`gym_model_fullbody.pt`) on our dataset.
*   `build_dataset.py`: The data generation script. It takes raw mp4 videos of good/bad form, runs them through MediaPipe, and spits out the raw joint angles into a CSV file for training.

## Setup & Running

You need Python installed. We recommend setting up a virtual environment.

```bash
pip install streamlit streamlit-webrtc mediapipe opencv-python torch torchvision pandas numpy
```

To run the app:
```bash
python -m streamlit run app.py
```
*(Note: Streamlit WebRTC needs to access your camera, so it works best locally or hosted on a secure HTTPS server).*
