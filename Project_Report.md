# GymForm AI - Project Report

## I. Problem Definition & Domain Research

**The "Why"**
Working out with poor form is one of the leading causes of gym injuries, and hiring a personal trainer is expensive. We wanted to build a real-time, accessible AI coach that watches your form and provides immediate visual feedback to prevent injury and ensure you're actually targeting the right muscles.

**Data Provenance**
Instead of scraping random internet videos where camera angles vary wildly, we built our own custom dataset from scratch. We recorded ourselves performing both "Good" and "Bad" versions of 5 different exercises (Bicep Curls, Squats, Lateral Raises, Shoulder Presses, and Tricep Extensions). 
*   **Features:** We didn't use raw video frames. Instead, we passed the videos through Google's MediaPipe to extract 33 3D skeletal landmarks, and calculated 8 specific joint angles (Left/Right Elbows, Shoulders, Hips, and Knees). These 8 angles are our actual features.
*   **Missing Values:** We didn't have traditional missing values because MediaPipe mathematically infers the position of occluded (hidden) joints. 
*   **Biases:** Because we built the dataset ourselves, it is heavily biased toward our specific body proportions and the camera sitting at roughly chest-height. 

## II. Methodology & Design Choices

**Preprocessing Logic**
The biggest design choice was converting raw `(x, y)` pixel coordinates into internal joint angles. We did this because angles are *scale and translation invariant*. If someone stands 10 feet away from the camera, their elbow pixel coordinates are drastically different than if they stood 2 feet away, but the *angle* of their elbow is exactly the same. We also implemented an Exponential Moving Average (EMA) filter on the incoming landmarks to smooth out camera jitter before calculating the angles.

**Architecture Selection**
We chose a lightweight Feedforward Neural Network in PyTorch (8 inputs → 64 → 32 → 10 outputs). 
*   *Why not a CNN?* CNNs are great for spatial pixel data, but our preprocessing already boiled the image down to 8 distinct numbers. A CNN would be massive overkill and computationally expensive.
*   *Why a Neural Network over Random Forest?* We needed sub-millisecond inference times to run inside a live video loop without lagging the webcam feed. A small PyTorch model allowed us to quickly compute probabilities and mathematically "mask" outputs on the fly.

## III. The "Failure Analysis"

**Failure 1: The "Cross-Prediction" Bug**
Initially, we just passed the 8 angles into the network and took the highest probability class. However, when testing the Bicep Curl, the model would sometimes randomly flash "Good Squat". Because the network evaluates all classes simultaneously, slight body twitches caused the "Squat" neurons to fire. 
*Fix:* We implemented a dynamic logit mask. If the user selects "Bicep Curl", we intercept the raw output tensor and set the probabilities for Squats, Raises, and Presses to negative infinity (`-inf`) *before* the softmax layer, mathematically forcing the model to only choose between Good/Bad Curl.

**Failure 2: Textbook Strictness vs. Real Life**
Originally, our state machine for counting reps required absolute textbook form. For example, a bicep curl only counted if the arm was perfectly straight (150+ degrees). During live testing, we realized this was physically impossible for some people without hyper-extending. The system felt broken because it wouldn't count valid reps. We had to drastically relax the angle thresholds (down to 120 degrees for a curl extension) and build a dynamic "active arm" detector for tricep extensions because the camera often couldn't see both arms at the same time.

## IV. Results & Hyperparameter Tuning

*   **Hyperparameters:** We used the Adam optimizer with a learning rate of `0.005` and trained for `150` epochs. We added a Dropout layer (`0.2`) between the hidden layers to prevent the model from instantly memorizing our small dataset.
*   **Temporal Tuning:** Instead of just relying on single-frame accuracy (which hovered around 55-60% during training), we tuned the live app to use a rolling prediction history. We store the last 10 frame predictions in a queue and use the `most_common` result. This drastically improved the perceived F1-score in real-time, because isolated "Bad Form" false-positives are smoothed out by the surrounding "Good Form" frames.

## V. Ethical Reflection & Limitations

**Real-World Failures**
This model will likely struggle in a crowded commercial gym. MediaPipe can get horribly confused by mirrors (seeing multiple skeletons) or by loose, baggy gym clothing (pump covers) that obscure the actual joint hinges. 

**Ethical Implications**
The biggest ethical issue is that the AI enforces a "one size fits all" standard for biomechanics. People have different femur lengths, shoulder mobilities, and physical disabilities. A deep squat might be physically unsafe for someone with bad knees, but our AI will tell them they have "Bad Form" if they don't go low enough. In its current state, the model lacks the nuance to adapt to individual physiological limitations, which could inadvertently encourage users to push past their safe range of motion to satisfy the algorithm.
