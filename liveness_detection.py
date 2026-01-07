import cv2
import numpy as np

# Load Haar Cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

def detect_eyes(face_roi):
    """
    Detect eyes within a face region. 
    Returns True if eyes are detected.
    """
    eyes = eye_cascade.detectMultiScale(face_roi, 1.3, 5)
    return len(eyes) >= 1

def track_movement(current_center, start_center, threshold=30):
    """
    Track centroid movement to detect head movement.
    """
    if start_center is None:
        return False
    
    dist = np.sqrt((current_center[0] - start_center[0])**2 + (current_center[1] - start_center[1])**2)
    return dist > threshold

def get_face_center(bbox):
    """
    Calculate the center of a bounding box.
    """
    x, y, w, h = bbox
    return (x + w//2, y + h//2)
