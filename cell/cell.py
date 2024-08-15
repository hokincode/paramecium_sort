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
        return {
            'frame': self.frame,
            'ID': self.ID,
            'x': self.x,
            'y': self.y,
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
    def read(cls, path):
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