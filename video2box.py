import os
import cv2
import copy
import numpy as np
import pandas as pd
import sys
from tracking_kalman.detect_stardist import Detectors
from stardist.models import StarDist2D
from datetime import timedelta

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

    contours["frame"] = 0
    centroid["frame"] = 0
    centroid_data = pd.concat([centroid_data, centroid], ignore_index=True)

    centroid_data.to_csv(centroid_data_path, index=False)
    frame_count = 1

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

        centroid.to_csv(centroid_data_path, mode='a', index=False, header=False)
        contours.to_csv(shape_data_path, mode='a', index=False, header=False)

        print("Currently processing:", timedelta(seconds=(frame_count / 30.0)),   end="\r", flush=True)

        # Write the processed frame to the output video
        output_video.write(orig_frame)

    # Release the video capture and writer objects
    cap.release()
    output_video.release()

    print("Frames Per Seconds (fps):", fps)

    # Save track_data as CSV
    centroid_data.to_csv(centroid_data_path, mode='a', index=False, header=False)

if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Wrong Usage: System parameters must equal to 6")
        print(sys.argv)
        print(len(sys.argv))
        print(sys.argv[1])
        print(sys.argv[2])
        print(sys.argv[3])
    else:
        video_path = sys.argv[1]
        output_video_path = sys.argv[2]
        track_data_path = sys.argv[3]
        centroid_data_path = sys.argv[4]
        shape_data_path = sys.argv[5]
        main(video_path, output_video_path, track_data_path, centroid_data_path, shape_data_path)