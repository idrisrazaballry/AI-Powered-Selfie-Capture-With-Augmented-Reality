import cv2
import mediapipe as mp
import numpy as np
import speech_recognition as sr
import threading
import queue
import os
import datetime

# Load AR filters
filters = {
    "1": cv2.imread("filters/glasses.png", cv2.IMREAD_UNCHANGED),
    "2": cv2.imread("filters/mustache.png", cv2.IMREAD_UNCHANGED),
    "3": cv2.imread("filters/beard.png", cv2.IMREAD_UNCHANGED),
    "4": cv2.imread("filters/cowboy_hat.png", cv2.IMREAD_UNCHANGED),
    "5": cv2.imread("filters/dog_face_filter.png", cv2.IMREAD_UNCHANGED),
    "6": cv2.imread("filters/dog_filter.png", cv2.IMREAD_UNCHANGED),
}

recognizer = sr.Recognizer()

def overlay_filter(img, filter_img, landmarks, point1, point2):
    h, w, _ = img.shape
    x1, y1 = int(landmarks.landmark[point1].x * w), int(landmarks.landmark[point1].y * h)
    x2, y2 = int(landmarks.landmark[point2].x * w), int(landmarks.landmark[point2].y * h)

    filter_width = int(1.5 * abs(x2 - x1))
    filter_height = int(filter_width * filter_img.shape[0] / filter_img.shape[1])
    x_offset = max(x1 - filter_width // 2, 0)
    y_offset = max(y1 - filter_height // 2, 0)

    filter_resized = cv2.resize(filter_img, (filter_width, filter_height), interpolation=cv2.INTER_AREA)

    y1_range = slice(y_offset, min(y_offset + filter_height, h))
    x1_range = slice(x_offset, min(x_offset + filter_width, w))
    fh = y1_range.stop - y1_range.start
    fw = x1_range.stop - x1_range.start
    filter_crop = filter_resized[0:fh, 0:fw]

    if filter_crop.shape[2] == 4:
        alpha_filter = filter_crop[..., 3:4] / 255.0
        alpha_img = 1.0 - alpha_filter
        img[y1_range, x1_range] = (alpha_img * img[y1_range, x1_range] + alpha_filter * filter_crop[..., :3]).astype(np.uint8)

    return img

def voice_listener(result_queue, stop_event):
    with sr.Microphone() as source:
        while not stop_event.is_set():
            try:
                audio = recognizer.listen(source, timeout=1, phrase_time_limit=2)
                text = recognizer.recognize_google(audio).lower()
                if "hey sefi" in text or "take selfie" in text:
                    result_queue.put(True)
                    return
            except sr.WaitTimeoutError:
                continue
            except (sr.UnknownValueError, sr.RequestError):
                continue
    result_queue.put(False)

def capture_selfie(mode):
    cap = cv2.VideoCapture(0)
    captured = False
    selfie = None

    with mp.solutions.face_mesh.FaceMesh(refine_landmarks=True) as face_mesh:
        if mode == "3":
            result_queue = queue.Queue()
            stop_event = threading.Event()
            listener_thread = threading.Thread(target=voice_listener, args=(result_queue, stop_event))
            listener_thread.start()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if mode == "1":  # Smile Mode
                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        upper_lip_y = face_landmarks.landmark[13].y
                        lower_lip_y = face_landmarks.landmark[14].y
                        lip_dist = abs(upper_lip_y - lower_lip_y)
                        if lip_dist > 0.035:
                            selfie = frame.copy()
                            captured = True
                            break
                cv2.putText(frame, "Smile to capture", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            elif mode == "2":  # Blink Mode
                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        left_eye_top = face_landmarks.landmark[159].y
                        left_eye_bottom = face_landmarks.landmark[145].y
                        left_eye_ratio = abs(left_eye_top - left_eye_bottom)

                        right_eye_top = face_landmarks.landmark[386].y
                        right_eye_bottom = face_landmarks.landmark[374].y
                        right_eye_ratio = abs(right_eye_top - right_eye_bottom)

                        if left_eye_ratio < 0.015 and right_eye_ratio < 0.015:
                            selfie = frame.copy()
                            captured = True
                            break
                cv2.putText(frame, "Blink to capture", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            elif mode == "3":  # Voice Mode
                cv2.putText(frame, "Say 'Hey Sefi' or 'Take selfie'", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                try:
                    if not result_queue.empty() and result_queue.get():
                        selfie = frame.copy()
                        captured = True
                        stop_event.set()
                        listener_thread.join()
                        break
                except Exception:
                    pass

            cv2.imshow("Camera", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                if mode == "3":
                    stop_event.set()
                    listener_thread.join()
                break

            if captured:
                break

    cap.release()
    cv2.destroyAllWindows()
    return selfie

def apply_ar_filter(selfie):
    print("\nSelect AR Filter to Apply:")
    print("1 - Glasses\n2 - Mustache\n3 - Beard\n4 - Cowboy_hat\n5 - Dog_face_filter\n6 - Dog_filter")
    choice = input("Enter filter number: ").strip()

    if choice not in filters or filters[choice] is None:
        print(" Filter image not found! Check filters folder.")
        return

    filter_img = filters[choice]

    with mp.solutions.face_mesh.FaceMesh(refine_landmarks=True) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(selfie, cv2.COLOR_BGR2RGB))
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                if choice == "1":  # Glasses
                    selfie = overlay_filter(selfie, filter_img, face_landmarks, 33, 263)
                elif choice == "2":  # Mustache
                    selfie = overlay_filter(selfie, filter_img, face_landmarks, 61, 291)
                elif choice == "3":  # Beard
                    selfie = overlay_filter(selfie, filter_img, face_landmarks, 152, 378)
                elif choice == "4" or choice == "5":  # Wig/Crown
                    selfie = overlay_filter(selfie, filter_img, face_landmarks, 10, 338)
                elif choice == "6":  # Emoji
                    selfie = overlay_filter(selfie, filter_img, face_landmarks, 234, 454)

    # Ensure folder exists
    save_folder = "saved_selfies"
    os.makedirs(save_folder, exist_ok=True)

    # Unique filename based on timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(save_folder, f"ar_selfie_{timestamp}.jpg")

    # Save the image
    cv2.imwrite(filename, selfie)
    print(f"\n AR Selfie saved as: {filename}")

    cv2.imshow("AR Selfie", selfie)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    print("Select Capture Mode:")
    print("1 - Smile Capture\n2 - Blink Capture\n3 - Voice Command Capture")
    mode = input("Enter mode (1/2/3): ").strip()

    selfie = capture_selfie(mode)
    if selfie is not None:
        apply_ar_filter(selfie)
    else:
        print("No selfie captured.")

if __name__ == "__main__":
    main()
