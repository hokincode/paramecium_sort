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
    def __init__(self, frame, ID, x, y, center, contour):
        self.frame = frame
        self.ID = ID
        self.x = x
        self.y = y
        self.center = center
        self.contour = contour

    def __repr__(self):
        return (f"Frame(frame={self.frame}, ID={self.ID}, x={self.x}, "
                f"y={self.y}, center={self.center}, contour_shape={self.contour.shape})")

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
            'center': [convert(c) for c in self.center],  # Convert center list elements
            'contour': self.contour.to_dict(orient='list')  # Convert DataFrame to dict
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

    def find_nearest(self, group, frame_index=-1):
        """
        Find the nearest cell in the next frame based on the specified frame index.
        Defaults to the latest frame if frame_index is not provided.
        """
        nearest_box = None
        min_distance = float('inf')

        for _, cache in group.iterrows():
            box_info = {
                'frame': cache['frame'],
                'ID': cache['ID'],
                'x': cache['x'],
                'y': cache['y'],
                'center': cache['center'],
                'contour':cache['contour']
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

This script processes a video to detect and track cells frame by frame, 
and outputs both a processed video and a folder containing the centroid 
data of the detected cells.

Usage:
    python your_script.py example_experiment.mp4 example_experiment
"""

def main(video_path, output_video_path, experiment_name):\

    # create a background subtractor
    # background_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)

    # Define the path for the experiment by appending a forward slash to the experiment name.
    # Create a directory with the path.
    path = experiment_name + '/'
    os.makedirs(path, exist_ok=True)
    centroid_data_path = os.path.join(path, 'raw_centroid_data.csv')

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
    # prev_frame = background_subtractor.apply(prev_frame)
    model = StarDist2D(None, name='grayscale_paramecium', basedir='models')
    centroid, contours, centers = detector.Detect(prev_frame, model)
    contours = contours.groupby('ID')
    centroid["frame"] = 0
    centroid_data = pd.concat([centroid_data, centroid], ignore_index=True)
    frame_count = 1

    # Initialize with centroids in the first frame
    print('Centroids Detected in the First Frame', centroid)
    list_of_cells = []
    for center_info in centers:
        coords = center_info[0]  # [x, y, width, height]
        frame = center_info[1]  # frame value
        ID = center_info[2]  # ID value
        box_info = {
            'frame': frame,
            'ID': ID,
            'x': coords[0],  # x coordinate
            'y': coords[1],  # y coordinate
            'center': center_info,  # Assuming you want to keep the full center info
            'contour': contours.get_group(ID)  # Assuming contours is a DataFrameGroupBy object
        }
        cell = Cell(box_info)
        list_of_cells.append(cell)
    print('Cell List instantiated from First Frame Centroid')

    # Infinite loop to process video frames
    while True:
        group = pd.DataFrame(columns=['frame', 'ID', 'x', 'y', 'center', 'contour'])
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break
        # Make copy of original frame
        orig_frame = copy.copy(frame)
        gray_frame = cv2.cvtColor(orig_frame, cv2.COLOR_BGR2GRAY)
        # Convert the grayscale frame back to BGR
        orig_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
        # orig_frame = background_subtractor.apply(orig_frame)
        # Detect and return centroids of the objects in the frame
        centroid, contours, centers = detector.Detect(orig_frame, model)
        contours["frame"] = frame_count
        centroid["frame"] = frame_count
        contours = contours.groupby('ID')

        # and add to group object
        for center_info in centers:
            coords = center_info[0]  # [x, y, width, height]
            frame = center_info[1]  # frame value
            ID = center_info[2]  # ID value
            box_info = {
                'frame': frame,
                'ID': ID,
                'x': coords[0],  # x coordinate
                'y': coords[1],  # y coordinate
                'center': center_info,  # Assuming you want to keep the full center info
                'contour': contours.get_group(ID)  # Assuming contours is a DataFrameGroupBy object
            }
            group = pd.concat([group, pd.DataFrame([box_info])], ignore_index=True)

        # Having a reference of the total data
        centroid_data = pd.concat([centroid_data, centroid], ignore_index=True)
        print("Currently processing:", timedelta(seconds=(frame_count / 30.0)),   end="\r", flush=True)

        ## iterate over cells for finding the nearest neighbour
        for i in range(len(list_of_cells)):
            cell = list_of_cells[i]
            next_frame = cell.find_nearest(group)
            cell.add_frame_info(next_frame)
            list_of_cells[i] = cell
            cv2.putText(orig_frame, str(i), (next_frame['x'], next_frame['y']), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 2)

        # Write the processed frame to the output video
        output_video.write(orig_frame)
        frame_count = frame_count + 1

    # Release the video capture and writer objects
    cap.release()
    output_video.release()

    print("Frames Per Seconds (fps):", fps)
    # Save track_data as CSV
    centroid_data.to_csv(centroid_data_path, mode='a', index=False, header=False)
    for i in range(len(list_of_cells)):
        cell = list_of_cells[i]
        cell.save(os.path.join(path, f'cell_{i}.json'))

def remove_edge(video_path, area_threshold=1000):

    cap = cv2.VideoCapture(video_path)

    # Read the first frame
    ret, image = cap.read()
    # Apply Canny edge detection
    edges = cv2.Canny(image, 100, 200)
    # Find contours in the edges image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    print('Contours found:', len(contours))
    # Iterate over contours to remove those that meet the criteria
    for i, contour in enumerate(contours):
        if area_threshold is not None:
            area = cv2.contourArea(contour)
            if area < area_threshold:
                # Draw over the contour to remove it (fill with black)
                cv2.drawContours(image, [contour], -1, (0, 0, 255), thickness=3)

    output_path = 'output_image.png'  # Ensure the path ends with a valid image extension
    # Save the modified image
    cv2.imwrite(output_path, image)
    return image


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
        experiment_name = sys.argv[3]
        remove_edge(video_path)
        # main(video_path, output_video_path, experiment_name)