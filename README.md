# Chess Position Recognition using Computer Vision

This project implements the chessboard detection algorithm described in Section 3.1 of the paper "Development of an autonomous chess robot system using computer vision and deep learning" by Truong Duc Phuc and Bui Cao Son.

## Overview

The chessboard detection process consists of four main steps:
1. Edge detection using Canny algorithm
2. Line detection using Hough Line Transform
3. Intersection point detection via clustering
4. Flattening the board using Homography transform

## Requirements

- Python 3.x
- OpenCV (cv2)
- NumPy
- Matplotlib
- scikit-learn (for DBSCAN clustering)

## Usage

Run the main script with:

```bash
python chess_board_detection.py
```

The script will:
1. Load and resize the chessboard image
2. Apply the detection algorithm
3. Save the visualization results to the `output` directory

## Output

The algorithm produces the following outputs:
- Original and resized images
- Edge detection results
- Line detection visualization
- Intersection points
- Flattened chessboard view
- Extracted chess squares

## Literature Review

This implementation is part of a deep learning literature review project, focusing on the chess position recognition section of the paper. It demonstrates how computer vision techniques can be used to detect a chessboard from an image before applying deep learning for piece recognition. 