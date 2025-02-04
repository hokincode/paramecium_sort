# @title Write a Detector function
# Import python libraries
from __future__ import print_function, unicode_literals, absolute_import, division
import sys
import numpy as np
import matplotlib
matplotlib.rcParams["image.interpolation"] = 'none'
import matplotlib.pyplot as plt

from glob import glob
from tifffile import imread
from csbdeep.utils import Path, normalize
from csbdeep.io import save_tiff_imagej_compatible

from stardist import random_label_cmap, _draw_polygons, export_imagej_rois
from stardist.models import StarDist2D

import cv2
import numpy as np
import pandas as pd

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

    def Detect(self, frame, model):
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

        contourColor = (64,163,241)  # Orange color (BGR format)
        centroidColor = (0, 0, 255)  # Red color (BGR format)
        centroidColor2 = (255, 0, 0)  # Red color (BGR format)
        boundingBoxColor = (0, 255, 0)

        # frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #change to gray scale

        n_channel = 1 if img.ndim == 2 else img.shape[-1]
        axis_norm = (0,1)   # normalize channels independently

        img = normalize(img, 1,99.8, axis=axis_norm)
        labels, details = model.predict_instances(img, predict_kwargs=dict(verbose=False))

        # Convert the 'labels' image to uint8 to use it with cv2.findContours()
        labels_uint8 = labels.astype(np.uint8)
        # Find contours in the 'labels' image
        contours, _ = cv2.findContours(labels_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        finalout = [];
        id_counter = 0;
        results = []
        shape_data = pd.DataFrame(columns=['ID', 'x', 'y'])
        centroid_data = pd.DataFrame(columns=['ID', 'x', 'y'])
        for contour in contours:
            # Calculate the bounding box for each contour
            x, y, w, h = cv2.boundingRect(contour)

            # Calculate the centroid of the bounding box
            centroid_x = x + w // 2
            centroid_y = y + h // 2

            # Draw the centroid
            centerA = (centroid_x, centroid_y)
            cv2.circle(frame, centerA, 2, centroidColor, cv2.FILLED)

            # Draw the bounding box
            cv2.drawContours(frame, [contour], 0, (0, 90, 170), 2)
            cv2.drawContours(frame, [contour], 0, (45, 150, 230), 1)

            # Extract and add the contour coordinates to the DataFrame
            for point in contour:
                xC, yC = point[0]
                shape_data = pd.concat([shape_data, pd.DataFrame({"ID": [id_counter], "x": [xC], "y": [yC]})], ignore_index=True)   

            centroid_data = pd.concat([centroid_data, pd.DataFrame({"ID": [id_counter], "x": [centroid_x], "y": [centroid_y]})], ignore_index=True)            
            
            # Extract the bounding box coordinates and assign a unique ID for each object
            xmin, ymin, xmax, ymax = int(x), int(y), int(x + w), int(y + h)
            class_id = id_counter
            id_counter += 1

            # Append the centroid coordinates to 'finalout'
            finalout.append(np.array([[centroid_x], [centroid_y]]))
            results.append([[xmin, ymin,  int(w), int(h)], 1, class_id])

        centers = details['points']
        centers = centers[:, ::-1]
        centers = centers.astype(int)
        for center in centers:
            b = np.array([[center[0]], [center[1]]])
            center = tuple(center)
            finalout.append(np.round(b))

        return (centroid_data, shape_data, results)