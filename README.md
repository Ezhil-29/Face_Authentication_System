# Face Recognition and Authetication System with Liveness Detection

This project implements a secure, real-time user authentication system using facial recognition combined with liveness detection (through blink detection) and a fallback password-based login. It is built using Python, OpenCV, Dlib, and Tkinter.

---

## Features

-  Register up to two users with their name , face and a backup password. (Can be implemented for multiple users also.)
-  128-dimensional face encoding using Dlib's ResNet model.
-  Eye-blink based liveness detection to prevent spoofing using photos or videos.
-  Fallback password authentication after multiple failed recognition attempts.
-  Simple GUI dialogs for name and password input using Tkinter.
-  Stores data in a JSON file; no external database required.

---

## Technologies Used

1. Python 3
2. OpenCV for image and video processing
3. Dlib for face detection, landmark detection, and facial encoding
4. Tkinter for GUI input prompts
5. NumPy for numerical operations
6. JSON for data storage

---

## Project Structure
Project structure should be somehow like this:
face-authentication-system/
 - data/             # Folder to store face encodings
     -  encodings.json          # Stores usernames, face encodings, and passwords
-  register.py                  # Script for registering new users
-  ecognize.py                  # Script for recognizing and authenticating users
-  README.md                    # This documentation file
-  shape_predictor_68_face_landmarks.dat        # Dlib model for facial landmarks
-  dlib_face_recognition_resnet_model_v1.dat    #  Dlib model for face recognition

---

## How the Project Works

### Registration

- User enters their name and sets a backup password via GUI.
- Webcam activates; user aligns their face and presses 's' to capture.
- Face encoding is generated and stored with the password.
- The system checks for duplicates before saving.

### Recognition

- Webcam prompts the user to blink within a few seconds to confirm liveness.
- If blinking is detected and the face matches a registered user, access is granted.
- If recognition fails after multiple attempts, the user is prompted to enter their password.

### After running recognize.py:

- Look into the webcam and blink when prompted.
- If your face is recognized, access is granted.
- Otherwise, you can log in using your backup password.

## To Be Noted
- This project is intended for local demonstration and learning purposes.
- By default, only two users can be registered (can be modified in the code).
- Passwords are stored in plain text inside encodings.json. In production,they should be hashed and securely stored.
- Requires a working webcam and decent lighting for accurate detection.

---
