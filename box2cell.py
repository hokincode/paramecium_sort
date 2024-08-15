import csv
import re
import json
import tqdm
import os
import os.path as osp
import argparse
import glob
import logging
import time
import os
import re
import pandas as pd
from cell.cell import Cell as Cell_Object

def logger_setup():
    """ logger_setup

    Returns:
        : logger
    """
    # create logger
    logger = logging.getLogger('Split into each frame')
    logger.setLevel(logging.CRITICAL)
    # create file handler which logs even debug messages
    log_name = '../log/{}.log'.format(time.strftime('%Y-%m-%d-%H-%M'))
    fh = logging.FileHandler(log_name)
    fh.setLevel(logging.CRITICAL)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.disabled = True  # Disable the logger
    return logger

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sort', type=str, default='centroid_example', help='directory of to be sorted', dest='sort')
    parser.add_argument('--sorted', type=str, default='sorted_example', help='directory of to be sorted', dest='sorted')
    return parser.parse_args()

def sort(df, path):
    # Group by the 'frame' column

    grouped = df.groupby('frame')
    group_2 = grouped.get_group(2)
    list_of_cells = []

    for _, row in group_2.iterrows():
        box_info = {
            'frame': row['frame'],
            'ID': row['ID'],
            'x': row['x'],
            'y': row['y'],
        }
        print(type(Cell_Object))
        cell = Cell_Object(box_info)
        list_of_cells.append(cell)
    print('Cell list instantiated')

    # Iterate through each frame's data
    for frame_id, group in grouped:
        print('Working on frame {}'.format(frame_id))
        for i in range(len(list_of_cells)):
            cell = list_of_cells[i]
            next_frame = cell.find_nearest(group)
            cell.add_frame_info(next_frame)
            list_of_cells[i] = cell

    for i in range(len(list_of_cells)):
        cell = list_of_cells[i]
        cell.save(os.path.join(path, f'cell_{i}.json'))

if __name__ == "__main__":
    args = get_args()
    print(os.getcwd())
    out_path = args.sorted
    input_path = args.sort
    experiment_behavior_csv = args.sort
    df = pd.read_csv(experiment_behavior_csv)
    out_path = out_path + args.sort + '/'
    os.makedirs(out_path, exist_ok=True)
    sort(df, out_path)