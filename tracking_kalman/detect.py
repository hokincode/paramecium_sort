# @title Write a Detector function
# Import python libraries
import cv2
import copy
import numpy as np

g_blurSize = 3
g_minContourSize = 80
g_maxContourSize = 2000
g_thresholdValue = 10 #100 for conditioning.mov

class Detectors(object):
    """Detectors class to detect objects in video frame
    Attributes:
        None
    """
    def __init__(self):
        """Initialize variables used by Detectors class
        Args:
            None
        Return:
            None
        """
        self.fgbg = cv2.createBackgroundSubtractorMOG2()

    def Detect(self, frame):
        """Detect objects in video frame using following pipeline
            - Convert captured frame from BGR to GRAY
            - Perform Background Subtraction
            - Detect edges using Canny Edge Detection
              http://docs.opencv.org/trunk/da/d22/tutorial_py_canny.html
            - Retain only edges within the threshold
            - Find contours
            - Find centroids for each valid contours
        Args:
            frame: single video frame
        Return:
            centers: vector of object centroids in a frame
        """

        # Convert BGR to GRAY
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (g_blurSize, g_blurSize), 0)

        _, g_thresholded = cv2.threshold(blurred, g_thresholdValue, 255, cv2.THRESH_BINARY)

        g_thresholded = cv2.GaussianBlur(g_thresholded, (7, 7), 0)

        _, g_thresholded = cv2.threshold(g_thresholded, 10, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(g_thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        centers = []  # vector of object centroids in a frame
        contourColor = (64,163,241)  # Orange color (BGR format)
        centroidColor = (0, 0, 255)  # Red color (BGR format)

        # Find centroid for each valid contours
        for cnt in contours:
            try:
                # Calculate and draw circle
                (x, y), radius = cv2.minEnclosingCircle(cnt)
                area = cv2.contourArea(cnt)
                centeroid = (int(x), int(y))
                if g_minContourSize < area < g_maxContourSize:
                    cv2.drawContours(frame, [cnt], -1, contourColor, 2)
                    cv2.circle(frame, centeroid, 3, centroidColor, cv2.FILLED)
                    b = np.array([[x], [y]])
                    centers.append(np.round(b))
            except ZeroDivisionError:
                pass

        return centers