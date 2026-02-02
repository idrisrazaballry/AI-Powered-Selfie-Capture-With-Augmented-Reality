import cv2
import numpy as np

def adjust_brightness(frame, brightness):
    """Helper function to adjust brightness using HSV colorspace."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # Adjust the value (brightness) channel and clip to 0-255
    v = cv2.add(v, brightness)
    v = np.clip(v, 0, 255).astype(v.dtype)
    
    hsv = cv2.merge((h, s, v))
    adjusted_frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    return adjusted_frame

def apply_brighten(frame):
    """
    Applies the brighten filter to a single input frame.
    """
    # Use 50 as the default brightness value
    return adjust_brightness(frame, 50)