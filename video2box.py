import os
import cv2
import copy
import numpy as np
import pandas as pd
import sys
from tracking_kalman.detect_stardist import Detectors
from tracking_kalman.tracking import Tracker
from deep_sort_realtime.deepsort_tracker import DeepSort
import haiku as hk
import jax
import tree
from stardist.models import StarDist2D
from datetime import timedelta


boundingBoxColor = (200, 0, 0)
areaThresh = 550

def frame_to_time(frame_number, frames_per_second=30):
    minutes, seconds = divmod(frame_number, frames_per_second)
    return f"{minutes:02d}:{seconds:02d}"
current_dir = os.getcwd()
print(current_dir)

def build_model(frames, query_points):
  """Compute point tracks and occlusions given frames and query points."""
  model = tapir_model.TAPIR()
  outputs = model(
      video=frames,
      is_training=False,
      query_points=query_points,
      query_chunk_size=64,
  )
  return outputs

model = hk.transform_with_state(build_model)
model_apply = jax.jit(model.apply)

def preprocess_frames(frames):
  """Preprocess frames to model inputs.

  Args:
    frames: [num_frames, height, width, 3], [0, 255], np.uint8

  Returns:
    frames: [num_frames, height, width, 3], [-1, 1], np.float32
  """
  frames = frames.astype(np.float32)
  frames = frames / 255 * 2 - 1
  return frames

def postprocess_occlusions(occlusions, expected_dist):
  """Postprocess occlusions to boolean visible flag.

  Args:
    occlusions: [num_points, num_frames], [-inf, inf], np.float32
    expected_dist: [num_points, num_frames], [-inf, inf], np.float32

  Returns:
    visibles: [num_points, num_frames], bool
  """
  # visibles = occlusions < 0
  visibles = (1 - jax.nn.sigmoid(occlusions)) * (1 - jax.nn.sigmoid(expected_dist)) > 0.5
  return visibles

def inference(frames, query_points):
    # Preprocess video to match model inputs format
    frames = preprocess_frames(frames)
    num_frames, height, width = frames.shape[0:3]
    query_points = query_points.astype(np.float32)
    frames, query_points = frames[None], query_points[None]  # Add batch dimension
    
    # Model inference
    rng = jax.random.PRNGKey(42)
    outputs, _ = model_apply(params, state, rng, frames, query_points)
    outputs = tree.map_structure(lambda x: np.array(x[0]), outputs)
    tracks, occlusions, expected_dist = outputs['tracks'], outputs['occlusion'], outputs['expected_dist']
    
    # Binarize occlusions
    visibles = postprocess_occlusions(occlusions, expected_dist)
    return tracks, visibles

def sample_random_points(frame_max_idx, height, width, num_points):
  """Sample random points with (time, height, width) order."""
  y = np.random.randint(0, height, (num_points, 1))
  x = np.random.randint(0, width, (num_points, 1))
  t = np.random.randint(0, frame_max_idx + 1, (num_points, 1))
  points = np.concatenate((t, y, x), axis=-1).astype(np.int32)  # [num_points, 3]
  return points

def main(video_path, output_video_path, track_data_path, centroid_data_path, shape_data_path):
    #track_data = pd.DataFrame(columns=['Time', 'ID', 'X', 'Y'])

    track_data = pd.DataFrame(columns=['frame', 'ID', 'xmin', 'ymin', 'xmax', 'ymax'])
    track_data.to_csv(track_data_path, index=False)

    centroid_data = pd.DataFrame(columns=['frame', 'ID', 'x', 'y'])
    shape_data = pd.DataFrame(columns=['frame', 'ID', 'x', 'y'])

    cap = cv2.VideoCapture(video_path)
    detector = Detectors()
    tracker = Tracker(10, 30, 8, 120)
    tracker = DeepSort(max_age=50)

    # Variables initialization
    track_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                    (0, 255, 255), (255, 0, 255), (255, 127, 255),
                    (127, 0, 255), (127, 0, 127)]
    pause = False

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Variables initialization
    skip_frame_count = 0

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
    shape_data = pd.concat([shape_data, contours], ignore_index=True)
    centroid_data = pd.concat([centroid_data, centroid], ignore_index=True)

    centroid_data.to_csv(centroid_data_path, index=False)
    shape_data.to_csv(shape_data_path, index=False)

    # update the tracker with the new detections
    tracks = tracker.update_tracks(centers, frame=prev_frame)
    mytracks = []
    # loop over the tracks
    for track in tracks:
        # if the track is not confirmed, ignore it
        if not track.is_confirmed():
            continue

        # get the track id and the bounding box
        track_id = track.track_id
        ltrb = track.to_ltrb()

        xmin, ymin, xmax, ymax = int(ltrb[0]), int(
            ltrb[1]), int(ltrb[2]), int(ltrb[3])

        # Calculate the area of the bounding box
        area = (xmax - xmin) * (ymax - ymin)
        if area < areaThresh:
          # Check if there are enough tracks in mytracks
          while len(mytracks) <= track_id:
            mytracks.append([])  # Initialize new tracks as empty lists
            
          if len(mytracks) <= int(track_id):
            mytracks[int(track_id)].append((xmin, ymin))
            # draw the bounding box and the track id
            clr = int(track_id) % 9
            cv2.rectangle(prev_frame, (xmin, ymin), (xmax, ymax), track_colors[clr], 1)
            cv2.rectangle(prev_frame, (xmin, ymin - 20), (xmin + 20, ymin), track_colors[clr], -1)
            cv2.putText(prev_frame, str(track_id), (xmin + 3, ymin - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)

            for j in range(len(mytracks[int(track_id)]) - 1):
                if j < len(mytracks[int(track_id)]) - 1:
                    x1 = mytracks[int(track_id)][j][0][0]
                    y1 = mytracks[int(track_id)][j][1][0]
                    x2 = mytracks[int(track_id)][j + 1][0][0]
                    y2 = mytracks[int(track_id)][j + 1][1][0]

                    cv2.line(prev_frame, (int(x1), int(y1)), (int(x2), int(y2)),
                            track_colors[clr], 1)      

    output_video.write(prev_frame)

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

        ######################################
        # TRACKING
        ######################################

        # update the tracker with the new detections
        tracks = tracker.update_tracks(centers, frame=frame)

        # loop over the tracks
        for track in tracks:
            # if the track is not confirmed, ignore it
            if not track.is_confirmed():
                continue

            # get the track id and the bounding box
            track_id = track.track_id
            ltrb = track.to_ltrb()

            xmin, ymin, xmax, ymax = int(ltrb[0]), int(
                ltrb[1]), int(ltrb[2]), int(ltrb[3])

            # Calculate the area of the bounding box
            area = (xmax - xmin) * (ymax - ymin)
            if area < areaThresh:
              while len(mytracks) <= int(track_id):
                mytracks.append([])  # Initialize new tracks as empty lists

              mytracks[int(track_id)].append((xmin, ymin))
              clr = int(track_id) % 9
              cv2.rectangle(orig_frame, (xmin, ymin), (xmax, ymax), track_colors[clr], 1)
              cv2.rectangle(orig_frame, (xmin, ymin - 20), (xmin + 20, ymin), track_colors[clr], -1)
              cv2.putText(orig_frame, str(track_id), (xmin + 3, ymin - 8),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)
              track_data_tmp = pd.DataFrame({'frame': [frame_count], 'ID': [track_id], 'xmin': [xmin], 'ymin': [ymin], 'xmax': [xmax], 'ymax': [ymax]})
              track_data_tmp.to_csv(track_data_path, mode='a', index=False, header=False)


        frame_count += 1

        # Write the processed frame to the output video
        output_video.write(orig_frame)

    # Release the video capture and writer objects
    cap.release()
    output_video.release()

    print("Frames Per Seconds (fps):", fps)

    # Save track_data as CSV
    track_data.to_csv(track_data_path, mode='a', index=False, header=False)
    centroid_data.to_csv(centroid_data_path, mode='a', index=False, header=False)
    shape_data.to_csv(shape_data_path, mode='a', index=False, header=False)

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
