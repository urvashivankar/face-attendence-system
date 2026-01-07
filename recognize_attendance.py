import cv2
import numpy as np
import pandas as pd
import pickle
import os
from datetime import datetime
from liveness_detection import detect_eyes, track_movement, get_face_center

# Constants
MOVEMENT_THRESHOLD = 40
BLINK_CONSEC_FRAMES = 2

def mark_attendance(name):
    filename = "attendance.csv"
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    
    if not os.path.exists(filename):
        df = pd.DataFrame(columns=["Name", "Date", "Time"])
        df.to_csv(filename, index=False)
    
    df = pd.read_csv(filename)
    if not ((df['Name'] == name) & (df['Date'] == date_str)).any():
        new_entry = pd.DataFrame([[name, date_str, time_str]], columns=["Name", "Date", "Time"])
        df = pd.concat([df, new_entry], ignore_index=True)
        df.to_csv(filename, index=False)
        print(f"Attendance marked for {name}")

def main():
    if not os.path.exists("encodings/lbph_model.yml"):
        print("Model not found! Run register_face.py first.")
        return

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("encodings/lbph_model.yml")
    with open("encodings/names.pkl", "rb") as f:
        names_map = pickle.load(f)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    video_capture = cv2.VideoCapture(0)
    
    # Liveness state
    eyes_closed_counter = 0
    blink_detected = False
    movement_detected = False
    start_center = None
    is_live = False

    while True:
        ret, frame = video_capture.read()
        if not ret: break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_roi_gray = gray[y:y+h, x:x+w]
            face_roi_color = frame[y:y+h, x:x+w]
            
            # 1. Recognition
            label_id, confidence = recognizer.predict(cv2.resize(face_roi_gray, (200, 200)))
            name = "Unknown"
            if confidence < 85: # Threshold for LBPH
                name = names_map.get(label_id, "Unknown")

            # 2. Liveness - Blink Detection (Simplification: Absence of eyes)
            eyes_present = detect_eyes(face_roi_color)
            if not eyes_present:
                eyes_closed_counter += 1
            else:
                if eyes_closed_counter >= BLINK_CONSEC_FRAMES:
                    blink_detected = True
                eyes_closed_counter = 0

            # 3. Liveness - Movement Tracking
            current_center = get_face_center((x, y, w, h))
            if start_center is None:
                start_center = current_center
            
            if not movement_detected:
                movement_detected = track_movement(current_center, start_center, MOVEMENT_THRESHOLD)

            # 4. Final Verification
            if blink_detected and movement_detected:
                is_live = True
                if name != "Unknown":
                    mark_attendance(name)

            # UI
            color = (0, 255, 0) if is_live else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, f"{name}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Dashboard
        cv2.putText(frame, f"Blink: {'OK' if blink_detected else 'Required'}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f"Move: {'OK' if movement_detected else 'Required'}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        status_text = "Verified" if is_live else "Verifying Liveness..."
        cv2.putText(frame, status_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow("Face Attendance - Universal Compatibility Edition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
