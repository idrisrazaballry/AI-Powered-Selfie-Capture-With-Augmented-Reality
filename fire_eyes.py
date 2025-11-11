import os, cv2, mediapipe as mp, numpy as np
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MEDIA_DIR = os.path.join(BASE_DIR, "media")

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def apply_fire_eyes(frame):
    """Applies the fire eyes filter with robust blending and clipping."""
    try:
        left_path = os.path.join(MEDIA_DIR, "fire_left.png")
        right_path = os.path.join(MEDIA_DIR, "fire_right.png")
        left_fire = cv2.imread(left_path, cv2.IMREAD_UNCHANGED)
        right_fire = cv2.imread(right_path, cv2.IMREAD_UNCHANGED)
        
        # Check for existence and 4 channels
        if left_fire is None or right_fire is None or left_fire.shape[2] < 4 or right_fire.shape[2] < 4:
            print(f"❌ Missing or invalid fire images in {MEDIA_DIR}")
            return frame

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)
        if not res.multi_face_landmarks:
            return frame

        h, w = frame.shape[:2]
        lm = res.multi_face_landmarks[0].landmark
        
        # Landmarks for eye centers
        left_eye_center, right_eye_center = lm[263], lm[33]

        # Calculate a stable size based on inter-eye distance
        eye_dist_norm = abs(right_eye_center.x - left_eye_center.x)
        # Target width for one eye is slightly larger than the distance between eye corners
        target_eye_w = int(eye_dist_norm * w * 0.9) 
        
        # Determine height based on aspect ratio
        left_h_w_ratio = left_fire.shape[0] / left_fire.shape[1]
        right_h_w_ratio = right_fire.shape[0] / right_fire.shape[1]
        
        target_left_h = int(target_eye_w * left_h_w_ratio)
        target_right_h = int(target_eye_w * right_h_w_ratio)

        for img, eye_lm, target_h in [
            (left_fire, left_eye_center, target_left_h), 
            (right_fire, right_eye_center, target_right_h)
        ]:
            if target_eye_w <= 0 or target_h <= 0: continue

            cx, cy = int(eye_lm.x * w), int(eye_lm.y * h)
            
            # Define the target bounding box
            x1_target = cx - target_eye_w // 2
            y1_target = cy - target_h // 2
            
            # --- Safe Boundary Clipping ---
            x1_clip = max(x1_target, 0)
            y1_clip = max(y1_target, 0)
            x2_clip = min(x1_target + target_eye_w, w)
            y2_clip = min(y1_target + target_h, h)

            if x1_clip >= x2_clip or y1_clip >= y2_clip: continue

            # Resize the fire image to the *original* target size
            resized_fire = cv2.resize(img, (target_eye_w, target_h), interpolation=cv2.INTER_AREA)

            # Get the corresponding slice of the *filter* image (for clipping)
            x1_offset = x1_clip - x1_target
            y1_offset = y1_clip - y1_target
            x2_offset = x1_offset + (x2_clip - x1_clip)
            y2_offset = y1_offset + (y2_clip - y1_clip)
            
            overlay_clipped = resized_fire[y1_offset:y2_offset, x1_offset:x2_offset]
            roi = frame[y1_clip:y2_clip, x1_clip:x2_clip].copy()

            # --- Alpha Blending ---
            overlay_bgr = overlay_clipped[:, :, :3].astype(np.float32)
            alpha = overlay_clipped[:, :, 3].astype(np.float32) / 255.0
            
            alpha_mask = cv2.merge([alpha, alpha, alpha])
            
            blended = (overlay_bgr * alpha_mask) + (roi.astype(np.float32) * (1.0 - alpha_mask))
            
            frame[y1_clip:y2_clip, x1_clip:x2_clip] = blended.astype(np.uint8)
            
        return frame
    except Exception as e:
        # print("🔥 Fire Eyes Filter Error:", e)
        return frame