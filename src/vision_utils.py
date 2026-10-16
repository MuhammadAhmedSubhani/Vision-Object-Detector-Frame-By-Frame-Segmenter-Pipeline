"""OpenCV Gaussian blur and adaptive threshold diagnostic."""
import cv2

def make_segmentation_view(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    block = 21 if min(frame.shape[:2]) >= 21 else 11
    return cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, 5
    )
