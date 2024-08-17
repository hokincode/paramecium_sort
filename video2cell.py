import os
import cv2
import copy
import numpy as np
import pandas as pd
import sys
from tracking_kalman.detect_stardist import Detectors
from datetime import timedelta
from stardist import random_label_cmap, _draw_polygons, export_imagej_rois
from stardist.models import StarDist2D
import math
import json

class Frame:
    def __init__(self, frame, ID, x, y):
        self.frame = frame
        self.ID = ID
        self.x = x
        self.y = y

    def __repr__(self):
        return (f"Frame(frame={self.frame}, ID={self.ID}, x={self.x}, "
                f"y={self.y}")

    def to_dict(self):
        def convert(value):
            if isinstance(value, (np.integer, np.floating)):
                return value.item()  # Convert numpy data types to native Python types
            return value

        return {
            'frame': convert(self.frame),
            'ID': convert(self.ID),
            'x': convert(self.x),
            'y': convert(self.y),
        }

class Cell:
    def __init__(self, initial_frame_info):
        """
        Initialize the Cell with the first frame information.
        initial_frame_info should be a dictionary containing the cell's information in the first frame.
        """
        self.frames = [Frame(**initial_frame_info)]

    def __repr__(self):
        """
        Represent the Cell with all its frame information.
        """
        return f"Cell(frames={self.frames})"

    def to_dict(self):
        """
        Convert the Cell to a dictionary with all its frame information.
        """
        return {'frames': [frame.to_dict() for frame in self.frames]}

    def add_frame_info(self, frame_info):
        """
        Add information for a new frame to the Cell.
        frame_info should be a dictionary containing the cell's information in the new frame.
        """
        self.frames.append(Frame(**frame_info))

    def distance_to(self, other):
        """
        Calculate the distance to another cell based on the specified frame index.
        Defaults to the latest frame if frame_index is not provided.
        :rtype: object
        """
        x1, y1 = self.frames[-1].x, self.frames[-1].y
        x2, y2 = other['x'], other['y']
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def find_nearest_json(self, group, frame_index=-1):
        """
        Find the nearest cell in the next frame based on the specified frame index.
        Defaults to the latest frame if frame_index is not provided.
        """
        nearest_box = None
        min_distance = float('inf')
        for _, row in group.iterrows():
            box_info = {
                'frame': row['frame'],
                'ID': row['ID'],
                'x': row['x'],
                'y': row['y'],
            }
            distance = self.distance_to(box_info)
            if distance < min_distance:
                min_distance = distance
                nearest_box = box_info
        return nearest_box

    def save(self, path):
        """
        Save the Cell's data to a JSON file at the specified path.
        """
        with open(path, 'w') as file:
            json.dump(self.to_dict(), file, indent=4)

    @classmethod
    def read_json(cls, path):
        """
        Read the Cell's data from a JSON file at the specified path.
        """
        with open(path, 'r') as file:
            data = json.load(file)
        # Create a new Cell instance
        frames = data['frames']
        cell = cls(frames[0])
        # Add the rest of the frames
        for frame_info in frames[1:]:
            cell.add_frame_info(frame_info)
        return cell


"""
Video Processing Script

This script processes a video to detect and track objects frame by frame, 
and outputs both a processed video and a CSV file containing the centroid 
data of the detected objects.

Usage:
    python your_script.py input_video.mp4 output_video.mp4 centroid_data.csv

Arguments:
    video_path: Path to the input video file.
    output_video_path: Path to save the processed output video file.
    centroid_data_path: Path to save the centroid data as a CSV file.

Dependencies:
    - OpenCV (cv2)
    - Pandas
    - StarDist2D model from stardist
"""

def main(video_path, output_video_path, centroid_data_path):

    centroid_data = pd.DataFrame(columns=['frame', 'ID', 'x', 'y'])
    cap = cv2.VideoCapture(video_path)
    detector = Detectors()

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create video writer object to save the output
    output_video = cv2.VideoWriter(output_video_path,
                                   cv2.VideoWriter_fourcc(*'mp4v'),
                                   fps, (width, height))

    # Read the first frame
    ret, prev_frame = cap.read()
    gray_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    # Convert the grayscale frame back to BGR
    prev_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
    model = StarDist2D(None, name='grayscale_paramecium', basedir='models')
    centroid, contours, centers = detector.Detect(prev_frame, model)
    print('Centroid detected', centroid, '\n')
    print('Contours detected', contours, '\n')
    print(type(contours))
    print(contours.iloc[2])
    contours = contours.groupby('ID')
    print('Contours detected', contours.count())
    first_object = contours.get_group(0)
    print(first_object)
    print('Centers detected', centers, '\n')
    print(type(centers), '\n')
    centroid["frame"] = 0
    centroid_data = pd.concat([centroid_data, centroid], ignore_index=True)
    frame_count = 1

    # Initialize with centroids in the first frame
    print('Centroids Detected in the First Frame', centroid)
    list_of_cells = []
    for _, row in centroid.iterrows():
        box_info = {
            'frame': row['frame'],
            'ID': row['ID'],
            'x': row['x'],
            'y': row['y'],
        }
        cell = Cell(box_info)
        list_of_cells.append(cell)
    print('Cell List instantiated from First Frame Centroid')

    # Infinite loop to process video frames
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break
        # Make copy of original frame
        orig_frame = copy.copy(frame)
        gray_frame = cv2.cvtColor(orig_frame, cv2.COLOR_BGR2GRAY)
        # Convert the grayscale frame back to BGR
        orig_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
        # Detect and return centroids of the objects in the frame
        centroid, contours, centers = detector.Detect(orig_frame, model)
        contours["frame"] = frame_count
        centroid["frame"] = frame_count
        centroid_data = pd.concat([centroid_data, centroid], ignore_index=True)
        print("Currently processing:", timedelta(seconds=(frame_count / 30.0)),   end="\r", flush=True)
        # Write the processed frame to the output video
        output_video.write(orig_frame)
        frame_count = frame_count + 1

    # Release the video capture and writer objects
    cap.release()
    output_video.release()

    print("Frames Per Seconds (fps):", fps)
    # Save track_data as CSV
    centroid_data.to_csv(centroid_data_path, mode='a', index=False, header=False)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Wrong Usage: System parameters must equal to 3")
        print(sys.argv)
        print(len(sys.argv))
        print(sys.argv[0])
        print(sys.argv[1])
        print(sys.argv[2])
        print(sys.argv[3])
    else:
        video_path = sys.argv[1]
        output_video_path = sys.argv[2]
        centroid_data_path = sys.argv[3]
        main(video_path, output_video_path, centroid_data_path)