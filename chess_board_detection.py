#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chess Position Recognition: Chessboard Detection

This script implements the approach described in Section 3.1 of the paper
"Development of an autonomous chess robot system using computer vision and deep learning"

The chessboard detection process consists of four main steps:
1. Edge detection using Canny algorithm
2. Line detection using Hough Line Transform
3. Intersection point detection via clustering
4. Flattening the board using Homography transform
"""

# Import necessary libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import os

# Configure matplotlib for better visualization
plt.rcParams['figure.figsize'] = (12, 10)

def edge_detection(image):
    """Detect edges using the Canny algorithm as described in the paper"""
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur (5x5 kernel as mentioned in the paper)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    
    # Apply Canny edge detection
    # The paper uses a double threshold in the range [100, 200]
    edges = cv2.Canny(blurred_image, 100, 150)
    
    return gray_image, blurred_image, edges

def line_detection(edges, min_line_length=100, max_line_gap=10):
    """Detect lines using the Hough Line Transform"""
    # Apply probabilistic Hough Line Transform
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=100,
                         minLineLength=min_line_length, maxLineGap=max_line_gap)
    
    return lines

def draw_lines(image, lines):
    """Draw detected lines on the image"""
    line_image = image.copy()
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
    
    return line_image

def find_line_intersections(lines, image_shape):
    """Find the intersections of all lines"""
    height, width = image_shape[0:2]
    intersections = []
    
    # Convert detected lines to a more usable format
    processed_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Calculate line equation ax + by + c = 0
            if x2 - x1 == 0:  # Vertical line
                a, b, c = 1, 0, -x1
            else:
                a = (y2 - y1) / (x2 - x1)
                b = -1
                c = y1 - a * x1
            processed_lines.append((a, b, c))
    
    # Find intersections between all pairs of lines
    for i in range(len(processed_lines)):
        for j in range(i+1, len(processed_lines)):
            a1, b1, c1 = processed_lines[i]
            a2, b2, c2 = processed_lines[j]
            
            # Calculate determinant
            det = a1 * b2 - a2 * b1
            
            # Check if lines are parallel
            if abs(det) < 1e-10:
                continue
            
            # Calculate intersection point
            x = (b1 * c2 - b2 * c1) / det
            y = (a2 * c1 - a1 * c2) / det
            
            # Only include points that are within the image
            if 0 <= x < width and 0 <= y < height:
                intersections.append((x, y))
    
    return np.array(intersections)

def cluster_intersections(intersections, eps=20, min_samples=1):
    """Cluster intersection points using DBSCAN"""
    if len(intersections) == 0:
        return []
        
    # Apply DBSCAN clustering
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(intersections)
    
    # Get cluster labels and number of clusters
    labels = clustering.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    
    # Calculate cluster centers
    clustered_points = []
    for i in range(n_clusters):
        cluster_points = intersections[labels == i]
        center_x = np.mean(cluster_points[:, 0])
        center_y = np.mean(cluster_points[:, 1])
        clustered_points.append((center_x, center_y))
    
    return np.array(clustered_points)

def draw_intersections(image, intersections):
    """Draw intersection points on the image"""
    intersection_image = image.copy()
    
    for point in intersections:
        x, y = point
        cv2.circle(intersection_image, (int(x), int(y)), 5, (0, 255, 0), -1)
    
    return intersection_image

def create_flattened_chessboard(image, intersections, board_size=8):
    """Create a flattened view of the chessboard using Homography transform"""
    # Not enough intersections for a valid chessboard
    if len(intersections) < 4:
        print(f"Warning: Not enough intersections for a chessboard.")
        return None, None
    
    # We need to sort the intersections to form a grid
    # First, let's try to find corners of the board
    try:
        # Sort by y-coordinate (top to bottom)
        sorted_by_y = intersections[np.argsort(intersections[:, 1])]
        
        # Try to get points for each row
        rows = []
        current_y = sorted_by_y[0, 1]
        current_row = []
        
        # Group points by similar y-coordinate
        for point in sorted_by_y:
            if abs(point[1] - current_y) < 20:  # Same row if within 20 pixels
                current_row.append(point)
            else:
                if current_row:  # If we have points in the current row
                    # Sort current row by x-coordinate
                    current_row = sorted(current_row, key=lambda p: p[0])
                    rows.append(current_row)
                # Start a new row
                current_y = point[1]
                current_row = [point]
        
        # Add the last row if it's not empty
        if current_row:
            current_row = sorted(current_row, key=lambda p: p[0])
            rows.append(current_row)
            
        # Extract the four corners of the chessboard
        if len(rows) > 0 and len(rows[0]) > 0 and len(rows[-1]) > 0:
            top_left = rows[0][0]
            top_right = rows[0][-1]
            bottom_left = rows[-1][0]
            bottom_right = rows[-1][-1]
            
            # Define source points (detected corners)
            src_points = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)
            
            # Define destination points (rectangle)
            dst_size = 800  # Size of the output square
            dst_points = np.array([[0, 0], [dst_size, 0], [dst_size, dst_size], [0, dst_size]], dtype=np.float32)
            
            # Calculate homography matrix
            H, _ = cv2.findHomography(src_points, dst_points)
            
            # Apply perspective transformation
            flattened = cv2.warpPerspective(image, H, (dst_size, dst_size))
            
            return flattened, H
        else:
            print("Error: Could not find valid rows for chessboard corners.")
            return None, None
    except Exception as e:
        print(f"Error applying homography: {e}")
        return None, None

def extract_chessboard_squares(flattened_image, board_size=8):
    """Extract individual squares from the flattened chessboard"""
    if flattened_image is None:
        return None
    
    height, width = flattened_image.shape[:2]
    square_size = min(height, width) // board_size
    
    squares = []
    for row in range(board_size):
        for col in range(board_size):
            # Calculate square coordinates
            top = row * square_size
            left = col * square_size
            
            # Extract the square
            square = flattened_image[top:top+square_size, left:left+square_size]
            squares.append(((row, col), square))
    
    return squares

def resize_image(image, target_width=416):
    """Resize the image to the target width while maintaining aspect ratio"""
    height, width = image.shape[:2]
    aspect_ratio = height / width
    target_height = int(target_width * aspect_ratio)
    resized_image = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)
    return resized_image

def process_chessboard_image(image_path, target_width=416):
    """Process a chessboard image through all steps"""
    # Load the chess board image
    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"Error: Could not load image from {image_path}")
        return None
    
    # Resize the image to the target width while maintaining aspect ratio
    print(f"Original image size: {original_image.shape[1]}x{original_image.shape[0]}")
    resized_image = resize_image(original_image, target_width)
    print(f"Resized image to: {resized_image.shape[1]}x{resized_image.shape[0]}")
    
    # Convert to RGB for display purposes (OpenCV loads as BGR)
    rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
    
    # Step 1: Edge detection
    print("Step 1: Applying edge detection...")
    gray_image, blurred_image, edges = edge_detection(resized_image)
    
    # Step 2: Line detection
    print("Step 2: Detecting lines using Hough Transform...")
    lines = line_detection(edges)
    if lines is None:
        print("No lines detected in the image.")
        return None
    print(f"Number of lines detected: {len(lines)}")
    
    # Step 3: Find and cluster intersection points
    print("Step 3: Finding line intersections...")
    intersections = find_line_intersections(lines, resized_image.shape)
    print(f"Number of raw intersections: {len(intersections)}")
    
    clustered_intersections = cluster_intersections(intersections)
    print(f"Number of clustered intersections: {len(clustered_intersections)}")
    
    # Step 4: Apply homography to flatten the chessboard
    print("Step 4: Applying homography transform...")
    flattened_image, homography_matrix = create_flattened_chessboard(resized_image, clustered_intersections)
    
    if flattened_image is None:
        print("Failed to create flattened chessboard view.")
        return None
    
    # Optional: Extract individual squares
    print("Extracting individual chessboard squares...")
    squares = extract_chessboard_squares(flattened_image)
    
    # Return all processed results
    results = {
        'original_image': original_image,
        'resized_image': resized_image,
        'rgb_image': rgb_image,
        'gray_image': gray_image,
        'blurred_image': blurred_image,
        'edges': edges,
        'lines': lines,
        'intersections': intersections,
        'clustered_intersections': clustered_intersections,
        'flattened_image': flattened_image,
        'squares': squares
    }
    
    return results

def display_results(results, save_dir='output'):
    """Display the results of chessboard detection and save to files"""
    if results is None:
        print("No results to display.")
        return
    
    # Create output directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Created output directory: {save_dir}")
    
    # Save original image
    plt.figure(figsize=(16, 8))
    
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(results['original_image'], cv2.COLOR_BGR2RGB))
    plt.title('Original Chess Board Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(results['rgb_image'])
    plt.title(f'Resized Image ({results["resized_image"].shape[1]}x{results["resized_image"].shape[0]})')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/01_original_and_resized.png", bbox_inches='tight')
    plt.close()
    print(f"Saved original and resized images to {save_dir}/01_original_and_resized.png")
    
    # Save edge detection results
    plt.figure(figsize=(16, 4))
    
    plt.subplot(1, 3, 1)
    plt.imshow(results['gray_image'], cmap='gray')
    plt.title('Grayscale Image')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(results['blurred_image'], cmap='gray')
    plt.title('Blurred Image')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(results['edges'], cmap='gray')
    plt.title('Canny Edge Detection')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/02_edge_detection.png", bbox_inches='tight')
    plt.close()
    print(f"Saved edge detection results to {save_dir}/02_edge_detection.png")
    
    # Save individual edge detection images for clarity
    cv2.imwrite(f"{save_dir}/02a_grayscale.png", results['gray_image'])
    cv2.imwrite(f"{save_dir}/02b_blurred.png", results['blurred_image'])
    cv2.imwrite(f"{save_dir}/02c_edges.png", results['edges'])
    
    # Save line detection results
    line_image = draw_lines(results['rgb_image'], results['lines'])
    plt.figure(figsize=(12, 10))
    plt.imshow(line_image)
    plt.title('Detected Lines (Hough Transform)')
    plt.axis('off')
    plt.savefig(f"{save_dir}/03_line_detection.png", bbox_inches='tight')
    plt.close()
    print(f"Saved line detection results to {save_dir}/03_line_detection.png")
    
    # Also save as BGR for OpenCV
    cv2.imwrite(f"{save_dir}/03_line_detection_cv2.png", 
               cv2.cvtColor(line_image, cv2.COLOR_RGB2BGR))
    
    # Save intersection points
    intersection_image = draw_intersections(results['rgb_image'], results['clustered_intersections'])
    plt.figure(figsize=(12, 10))
    plt.imshow(intersection_image)
    plt.title('Detected Intersections')
    plt.axis('off')
    plt.savefig(f"{save_dir}/04_intersections.png", bbox_inches='tight')
    plt.close()
    print(f"Saved intersection detection results to {save_dir}/04_intersections.png")
    
    # Also save as BGR for OpenCV
    cv2.imwrite(f"{save_dir}/04_intersections_cv2.png", 
               cv2.cvtColor(intersection_image, cv2.COLOR_RGB2BGR))
    
    # Save flattened chessboard
    if results['flattened_image'] is not None:
        flattened_rgb = cv2.cvtColor(results['flattened_image'], cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(10, 10))
        plt.imshow(flattened_rgb)
        plt.title('Flattened Chessboard View')
        plt.axis('off')
        plt.savefig(f"{save_dir}/05_flattened_chessboard.png", bbox_inches='tight')
        plt.close()
        print(f"Saved flattened chessboard to {save_dir}/05_flattened_chessboard.png")
        
        # Also save the OpenCV BGR version
        cv2.imwrite(f"{save_dir}/05_flattened_chessboard_cv2.png", results['flattened_image'])
    
    # Save a subset of extracted squares
    if results['squares']:
        plt.figure(figsize=(15, 5))
        for i in range(min(8, len(results['squares']))):
            position, square = results['squares'][i]
            plt.subplot(1, 8, i+1)
            plt.imshow(cv2.cvtColor(square, cv2.COLOR_BGR2RGB))
            plt.title(f"({position[0]}, {position[1]})")
            plt.axis('off')
        plt.tight_layout()
        plt.savefig(f"{save_dir}/06_extracted_squares.png", bbox_inches='tight')
        plt.close()
        print(f"Saved extracted squares to {save_dir}/06_extracted_squares.png")
        
        # Save individual squares
        for i in range(min(16, len(results['squares']))):
            position, square = results['squares'][i]
            square_filename = f"{save_dir}/square_{position[0]}_{position[1]}.png"
            cv2.imwrite(square_filename, square)

if __name__ == "__main__":
    # Process the chessboard image
    image_path = 'chessboard3.png'
    results = process_chessboard_image(image_path)
    
    # Display the results
    if results:
        display_results(results)
    else:
        print("Processing failed. Check the error messages above.")
