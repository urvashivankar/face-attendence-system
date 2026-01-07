import cv2
import numpy as np
import os
import pickle

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def register_user():
    name = input("Enter the name of the person: ").strip()
    if not name:
        print("Name cannot be empty!")
        return

    user_dir = os.path.join("dataset", name)
    if not os.path.exists(user_dir):
        os.makedirs(user_dir)

    video_capture = cv2.VideoCapture(0)
    print(f"Capturing face for {name}. Press 'c' to capture, or 'q' to quit.")

    img_counter = 0
    while True:
        ret, frame = video_capture.read()
        if not ret: break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.imshow("Register Face - Press 'c' to capture", frame)

        key = cv2.waitKey(1)
        if key % 256 == ord('q'):
            break
        elif key % 256 == ord('c'):
            if len(faces) > 0:
                img_name = os.path.join(user_dir, f"{name}_{img_counter}.jpg")
                cv2.imwrite(img_name, frame)
                print(f"{img_name} saved!")
                img_counter += 1
                if img_counter >= 10:
                    print("Captured 10 images.")
                    break
            else:
                print("No face detected! Try again.")

    video_capture.release()
    cv2.destroyAllWindows()

    if img_counter > 0:
        print("Training recognition model...")
        train_model()

def train_model():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    faces_list = []
    labels = []
    names = {}
    label_id = 0

    dataset_path = "dataset"
    if not os.path.exists(dataset_path): return

    for user_name in os.listdir(dataset_path):
        user_dir = os.path.join(dataset_path, user_name)
        if not os.path.isdir(user_dir): continue
        
        names[label_id] = user_name
        for img_name in os.listdir(user_dir):
            img_path = os.path.join(user_dir, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            detected_faces = face_cascade.detectMultiScale(img, 1.3, 5)
            for (x, y, w, h) in detected_faces:
                face_roi = img[y:y+h, x:x+w]
                faces_list.append(cv2.resize(face_roi, (200, 200)))
                labels.append(label_id)
        label_id += 1

    if faces_list:
        recognizer.train(faces_list, np.array(labels))
        if not os.path.exists("encodings"): os.makedirs("encodings")
        recognizer.save("encodings/lbph_model.yml")
        with open("encodings/names.pkl", "wb") as f:
            pickle.dump(names, f)
        print("Model trained and saved to encodings/lbph_model.yml")
    else:
        print("No faces found to train.")

if __name__ == "__main__":
    register_user()
