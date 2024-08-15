## Paramecium Sorting Algorithm

Sort videos into segmentation boxes of paramecium cells.

```sh
python video2box.py './video/example.avi' 'out_example.mp4' 'track_example.csv' 'centroid_example.csv' 'shape_data.csv'
```

Sort boxes in the frames into individual cells (save in the "sorted_cell" folder)

```sh
python box2cell.py --sort centroid_example.csv --sorted sorted_cell
```

Analysis

```sh
jupyter notebook analysis.ipynb
```
