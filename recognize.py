#RECOGNITION
import cv2, dlib, json, numpy as np, os, time
import tkinter as tk
from tkinter import simpledialog, messagebox

#CONSTANTS
PREDICTOR_PATH         = "shape_predictor_68_face_landmarks.dat"  # Model to detect facial landmarks
RECOGNITION_MODEL_PATH = "dlib_face_recognition_resnet_model_v1.dat" # Model to extract 128D face embeddings
SAVE_PATH              = "data/encodings.json"  # File that stores face encodings and passwords
TOLERANCE              = 0.60   # Threshold for facial recognition match; lower is stricter
MAX_ATTEMPTS           = 3      # Maximum face or password attempts allowed
BLINK_THRESHOLD        = 0.21   # Threshold below which an eye is considered closed
CONSEC_FRAMES          = 2      # Minimum consecutive frames with eyes closed to count as a blink
BLINK_TIMEOUT          = 5      # Seconds to detect a blink before retry
RECOGNITION_DELAY      = 3      # Wait after recognition for confirmation


#Calculates Eye Aspect Ratio (EAR) -> quantify eye openness.
def eye_aspect_ratio(eye: np.ndarray) -> float:
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)

#Returns coordinates of left and right eyes from the 68 facial landmarks.
def extract_eye_points(shape):
    left  = np.array([(shape.part(i).x, shape.part(i).y) for i in range(36, 42)])
    right = np.array([(shape.part(i).x, shape.part(i).y) for i in range(42, 48)])
    return left, right

#GUI dialog for username and password entry.
def prompt_password_gui(usernames):
    root = tk.Tk(); root.withdraw()
    name = simpledialog.askstring("Authentication", "Enter your name:")
    if name not in usernames:
        messagebox.showerror("Error", "User not found."); root.destroy(); return None, None
    pwd  = simpledialog.askstring("Authentication", "Enter your password:", show="*")
    root.destroy(); return name, pwd


#128D encoding for the given face using dlib.
def compute_face_encoding(img, face, predictor, encoder):
    shape = predictor(img, face)
    return np.array(encoder.compute_face_descriptor(img, shape))

#MAIN fUNCTION

def recognize_user():
    if not all(os.path.exists(p) for p in (PREDICTOR_PATH, RECOGNITION_MODEL_PATH, SAVE_PATH)):
        print("❌ Required files missing."); return

    #Loading Face encodings and Passwords
    with open(SAVE_PATH, "r") as f:
        data = json.load(f)
    known_encodings = {n: np.array(v["encoding"]) for n, v in data.items()}
    passwords       = {n: v["password"]        for n, v in data.items()}

    #Initialising Models and Camera
    detector  = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(PREDICTOR_PATH)
    encoder   = dlib.face_recognition_model_v1(RECOGNITION_MODEL_PATH)
    cam       = cv2.VideoCapture(0)

    print("🔎 Look at the camera → blink when asked so we know you're alive!")

    face_attempts = 0
    while face_attempts < MAX_ATTEMPTS:
        print(f"🕒 Attempt {face_attempts+1}/{MAX_ATTEMPTS}: blink within {BLINK_TIMEOUT}s …")
        blink_detected = False
        blink_counter  = 0
        start = time.time()

        # Blink detection
        while time.time() - start < BLINK_TIMEOUT:
            ok, frame = cam.read()
            if not ok: continue
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = detector(rgb)

            if faces:
                shape = predictor(rgb, faces[0])
                left_eye, right_eye = extract_eye_points(shape)
                ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2

                if ear < BLINK_THRESHOLD:
                    blink_counter += 1
                else:
                    if blink_counter >= CONSEC_FRAMES:
                        blink_detected = True; break
                    blink_counter = 0

            cv2.putText(frame, "Please BLINK", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(0,255,255),2)
            cv2.imshow("Liveness", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cam.release(); cv2.destroyAllWindows(); return

        cv2.destroyWindow("Liveness")

        # If blink is not detected in time
        if not blink_detected:
            print("❌ Blink not detected → retrying …")
            face_attempts += 1
            continue

        # Next-> facial recognition
        print("✅ Blink detected → capturing face …")
        ok, frame = cam.read()
        if not ok:
            print("⚠ Camera error."); face_attempts += 1; continue
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = detector(rgb)
        if not faces:
            print("⚠ Lost face after blink → retry."); face_attempts += 1; continue

        enc = compute_face_encoding(rgb, faces[0], predictor, encoder)
        matched = None
        for name, vec in known_encodings.items():
            if np.linalg.norm(enc - vec) < TOLERANCE:
                matched = name; break

        if matched:
            print(f"✅ Face matched: {matched}. Waiting {RECOGNITION_DELAY}s for final confirmation …")
            time.sleep(RECOGNITION_DELAY)
            print(f"🔓 Access GRANTED to {matched}.")
            cam.release(); cv2.destroyAllWindows(); return
        else:
            print("❌ Face NOT recognized.")
            face_attempts += 1

    # After failed attempts of facial recognition → password authentication
    cam.release(); cv2.destroyAllWindows()
    print("🔐 Switching to PASSWORD mode …")
    for i in range(MAX_ATTEMPTS):
        name, pwd = prompt_password_gui(passwords.keys())
        if name and passwords.get(name) == pwd:
            print(f"🔓 Access GRANTED (password) → {name}"); return
        messagebox.showerror("Access Denied", f"❌ Wrong password. Attempts left: {MAX_ATTEMPTS-i-1}")
        print(f"⛔ Wrong password. Attempts left: {MAX_ATTEMPTS-i-1}")
    print("⛔ Too many wrong passwords. Access DENIED.")

if __name__ == "__main__":
    recognize_user()
