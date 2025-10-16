import cv2
import numpy as np
import mediapipe as mp
import datetime
import itertools

# Initialize mediapipe face mesh and utilities
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, 
                                  min_detection_confidence=0.5, min_tracking_confidence=0.3)

def get_head_top_position(image, face_landmarks):
    # For a crown, top of the head is the highest point in the FACE_OVAL landmarks
    image_height, image_width, _ = image.shape
    landmarks = []
    for idx in mp_face_mesh.FACEMESH_FACE_OVAL:
        for point in idx:
            x = int(face_landmarks.landmark[point].x * image_width)
            y = int(face_landmarks.landmark[point].y * image_height)
            landmarks.append((x, y))
    # Get the minimum y value (topmost point)
    top_landmark = min(landmarks, key=lambda lm: lm[1])
    return top_landmark

def overlay_crown(image, crown_img, face_landmarks):
    annotated_image = image.copy()
    # Use the topmost point as reference to position the crown
    top_x, top_y = get_head_top_position(image, face_landmarks)
    crown_h, crown_w, _ = crown_img.shape
    # Position crown slightly above the head
    x1 = top_x - crown_w // 2
    y1 = top_y - crown_h // 2

    # Make sure overlay is within image bounds
    if x1 < 0: x1 = 0
    if y1 < 0: y1 = 0
    x2 = x1 + crown_w
    y2 = y1 + crown_h
    if x2 > image.shape[1]: x2 = image.shape[1]
    if y2 > image.shape[0]: y2 = image.shape[0]

    # Handling transparent PNG crowns
    alpha_crown = crown_img[:,:,3] / 255.0 if crown_img.shape[2] == 4 else np.ones((crown_h, crown_w))
    for c in range(0, 3):
        annotated_image[y1:y2, x1:x2, c] = (
            alpha_crown * crown_img[:y2-y1,:x2-x1,c] +
            (1-alpha_crown) * annotated_image[y1:y2, x1:x2, c]
        )
    return annotated_image

def apply_crown_filter():
    camera_video = cv2.VideoCapture(0)
    camera_video.set(3, 1280)
    camera_video.set(4, 960)
    cv2.namedWindow('Crown Filter', cv2.WINDOW_NORMAL)
    crown = cv2.imread('media/crown.png', cv2.IMREAD_UNCHANGED)  # Load with alpha

    while camera_video.isOpened():
        ok, frame = camera_video.read()
        if not ok:
            continue
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                frame = overlay_crown(frame, crown, face_landmarks)
        cv2.imshow('Crown Filter', frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('c'):
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"crown_image_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Image captured and saved as {filename}")
        if k == ord('q'):  # Press 'q' to exit
            break
    camera_video.release()
    cv2.destroyAllWindows()

# To run the filter, just call:
# apply_crown_filter()
