import cv2

def apply_blur(frame):
    """
    Applies the Gaussian blur filter to a single input frame.
    """
    # Apply Gaussian blur to the frame with a 15x15 kernel
    blurred_frame = cv2.GaussianBlur(frame, (15, 15), 0)
    

    return blurred_frame
