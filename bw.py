import cv2

def apply_bw(frame):
    """
    Applies the black and white filter to a single input frame.
    Converts the resulting grayscale image back to 3 channels.
    """
    # 1. Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 2. Convert grayscale (1 channel) back to BGR (3 channels) for Tkinter compatibility
    filtered_frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    return filtered_frame