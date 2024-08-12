# @title Imports {form-width: "25%"}

import haiku as hk
import jax
import mediapy as media
import numpy as np
import tree

from tapnet import tapir_model
from tapnet.utils import transforms
from tapnet.utils import viz_utils

import cv2

# @title Load Checkpoint {form-width: "25%"}

checkpoint_path = 'tapnet/checkpoints/causal_tapir_checkpoint.npy' #tapir_checkpoint.npy
ckpt_state = np.load(checkpoint_path, allow_pickle=True).item()
params, state = ckpt_state['params'], ckpt_state['state']

# @title Build Model {form-width: "25%"}

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

# @title Utility Functions {form-width: "25%"}

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

# @title Load an Exemplar Video {form-width: "25%"}

video = media.read_video('./video/tapir_test.avi') #'tapnet/examplar_videos/horsejump-high.mp4'
media.show_video(video, fps=10)

# @title Predict Sparse Point Tracks {form-width: "25%"}

resize_height = 256  # @param {type: "integer"}
resize_width = 256  # @param {type: "integer"}
num_points = 18  # @param {type: "integer"}

height, width = video.shape[1:3]
frames = media.resize_video(video, (resize_height, resize_width))
#query_points = sample_random_points(0, frames.shape[1], frames.shape[2], num_points)

# Original pixel coordinates from a 640x480 image
original_coords = np.array([[0, 16, 498],
                           [0, 76, 502],
                           [0, 93, 502],
                           [0, 298, 544],
                           [0, 311, 415],
                           [0, 335, 385],
                           [0, 312, 320],
                           [0, 299, 149],
                           [0, 287, 112],
                           [0, 260, 102],
                           [0, 159, 5],
                           [0, 143, 40],
                           [0, 89, 99],
                           [0, 98, 263],
                           [0, 76, 286],
                           [0, 91, 366],
                           [0, 135, 383],
                           [0, 155, 389],
                           [0, 170, 400],
                           [25, 25, 445],
                           [25, 80, 428],
                           [25, 105, 459],
                           [25, 112, 466],
                           [25, 187, 468],
                           [25, 191, 478],
                           [25, 272, 455],
                           [25, 309, 566],
                           [25, 371, 428],
                           [25, 198, 308],
                           [25, 123, 265],
                           [25, 179, 174],
                           [25, 147, 150],
                           [25, 138, 136],
                           [25, 180, 45],
                           [25, 240, 92],
                           [25, 302, 76],
                           [25, 407, 80],
                           [25, 152, 4],
                           [59, 117, 49],
                           [59, 169, 41],
                           [59, 192, 70],
                           [59, 160, 106],
                           [59, 46, 231],
                           [59, 303, 151],
                           [59, 340, 139],
                           [59, 247, 290],
                           [59, 322, 247],
                           [59, 327, 326],
                           [59, 270, 485],
                           [59, 294, 512],
                           [59, 257, 514],
                           [59, 192, 376],
                           [59, 152, 483],
                           [59, 58, 504],
                           [59, 40, 408],
                           [43, 171, 50],
                           [43, 113, 89],
                           [43, 218, 85],
                           [43, 144, 189],
                           [43, 106, 191],
                           [43, 252, 201],
                           [43, 336, 115],
                           [43, 323, 121],
                           [43, 280, 325],
                           [43, 182, 381],
                           [43, 173, 400],
                           [43, 208, 434],
                           [43, 215, 482],
                           [43, 230, 497],
                           [43, 344, 444],
                           [43, 51, 422],
                           [43, 29, 448],
                           [13, 208, 17],
                           [13, 148, 15],
                           [13, 99, 115],
                           [13, 267, 102],
                           [13, 296, 81],
                           [13, 238, 156],
                           [13, 117, 217],
                           [13, 418, 60],
                           [13, 53, 51],
                           [13, 158, 294],
                           [13, 129, 312],
                           [13, 105, 378],
                           [13, 170, 434],
                           [13, 340, 392],
                           [13, 363, 327],
                           [13, 307, 492],
                           [13, 299, 499],
                           [13, 186, 478],
                           [13, 70, 503],
                           [13, 39, 507],
                           [13, 14, 458],
                           [13, 90, 482],
                           [22, 26, 80],
                           [22, 150, 6],
                           [22, 188, 40],
                           [22, 122, 130],
                           [22, 136, 164],
                           [22, 246, 94],
                           [22, 192, 170],
                           [22, 304, 72],
                           [22, 414, 72],
                           [22, 366, 430],
                           [22, 282, 468],
                           [22, 310, 552],
                           [22, 186, 466],
                           [22, 186, 478],
                           [22, 100, 478],
                           [22, 90, 470],
                           [22, 84, 416],
                           [22, 30, 460],
                           [22, 124, 270],
                           [22, 188, 306]], dtype=np.int32)

# Scale down the coordinates to a 256x256 image
scaled_coords = original_coords * (1, 256 / 480, 256 / 640)

# Convert the coordinates to int32 data type
query_points = scaled_coords.astype(np.int32)
#query_points = np.insert(query_points, 0, 0, axis=1)


tracks, visibles = inference(frames, query_points)

# Visualize sparse point tracks
tracks = transforms.convert_grid_coordinates(tracks, (resize_width, resize_height), (width, height))
video_viz = viz_utils.paint_point_track(video, tracks, visibles)
#media.show_video(video_viz, fps=10)

# Assuming you have a NumPy array called 'frames' with shape (num_frames, height, width, 3) and dtype uint8

# Define the output video path and filename
output_path = 'output_video_cells.mp4'

# Obtain the video dimensions
num_frames, height, width, _ = video_viz.shape

# Create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Specify the codec (e.g., 'mp4v', 'avc1', 'XVID')
fps = 10  # Specify the frames per second (FPS)
video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Iterate through each frame and write it to the video file
for i in range(num_frames):
    frame = cv2.cvtColor(video_viz[i], cv2.COLOR_RGB2BGR) 
    video_writer.write(frame)

# Release the VideoWriter
video_writer.release()

print(f"Video saved as {output_path}")