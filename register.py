# REGISTERATION
import cv2, dlib, json, os
import numpy as np
import tkinter as tk
from tkinter import simpledialog, messagebox


PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
RECOGNITION_MODEL_PATH = "dlib_face_recognition_resnet_model_v1.dat"
SAVE_PATH = "data/encodings.json"    # Encodings sstored in JSON 
TOLERANCE = 0.6  # Similarity threshold for duplicate face check

#Output directory for storing encodings?
os.makedirs("data", exist_ok=True)

#GUI prompt for password input using Tkinter
def prompt_password_gui():
    root = tk.Tk()
    root.withdraw()
    password = simpledialog.askstring("Backup Password", "Set your backup password:", show='*')
    root.destroy()
    return password

#128-D embedding from the input face image
def compute_face_encoding(image, face, predictor, face_encoder):
    shape = predictor(image, face)
    return np.array(face_encoder.compute_face_descriptor(image, shape))

#MAIN FUNCTION
def register_user():
    # Confirm required model files exist
    if not os.path.exists(PREDICTOR_PATH) or not os.path.exists(RECOGNITION_MODEL_PATH):
        print("❌ Required Dlib models missing.")
        return

    # Loading existing encoding and password data 
    data = {}
    if os.path.exists(SAVE_PATH):
        with open(SAVE_PATH, "r") as f:
            data = json.load(f)

    # Limiting registration to 2 users
    if len(data) >= 2:
        print("❌ Only 2 users allowed.")
        return

    #User name input
    name = input("Enter your name: ").strip()
    if not name or name in data:
        print("❌ Invalid or duplicate name.")
        return

    # Prompt for password input
    password = prompt_password_gui()
    if not password:
        print("❌ Password required.")
        return

    # Initialize models
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(PREDICTOR_PATH)
    face_encoder = dlib.face_recognition_model_v1(RECOGNITION_MODEL_PATH)

    cam = cv2.VideoCapture(0)
    print("📸 Press 's' to capture your face...")

    encoding = None
    while True:
        ret, frame = cam.read()
        if not ret:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = detector(rgb)

        #Facial landmarks for user alignment assistance
        for face in faces:
            shape = predictor(rgb, face)
            for i in range(68):
                pt = shape.part(i)
                cv2.circle(frame, (pt.x, pt.y), 2, (255, 0, 0), -1)

        cv2.putText(frame, "Align your face. Press 's' to capture.", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow("Register", frame)

        # Wait for user to press 's' to take the face snapshot
        if cv2.waitKey(1) & 0xFF == ord('s'):
            if faces:
                encoding = compute_face_encoding(rgb, faces[0], predictor, face_encoder)
                break
            else:
                print("⚠️ No face detected.")

    cam.release()
    cv2.destroyAllWindows()

    #Face successfully captured?
    if encoding is None:
        print("❌ Face not captured.")
        return

    # Verification of present face not being  already registered
    for user in data.values():
        stored_vec = np.array(user["encoding"])
        dist = np.linalg.norm(encoding - stored_vec)
        if dist < TOLERANCE:
            print("⚠️ This face is already registered.")
            return

    #New user details
    data[name] = {
        "encoding": encoding.tolist(),
        "password": password
    }

    with open(SAVE_PATH, "w") as f:
        json.dump(data, f, indent=2)

    print(f"✅ '{name}' registered successfully.")

if __name__ == "__main__":
    register_user()
