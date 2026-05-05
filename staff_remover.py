import cv2
import numpy as np

def binarize(img):
    """Convert BGR or grayscale image to binary (0=black, 255=white)"""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    return binary

def find_staff_line_groups(binary, threshold=0.3):
    """
    Find groups of rows that are staff lines.
    A staff line row has more than threshold*width dark pixels.
    
    Returns list of groups, each group is a list of consecutive row indices.
    """
    height, width = binary.shape
    dark_counts = np.sum(binary == 0, axis=1)
    line_rows = np.where(dark_counts >= width * threshold)[0]

    if len(line_rows) == 0:
        return []

    groups = []
    current = [line_rows[0]]
    for i in range(1, len(line_rows)):
        if line_rows[i] - line_rows[i-1] <= 2:
            current.append(line_rows[i])
        else:
            groups.append(current)
            current = [line_rows[i]]
    groups.append(current)
    return groups

def remove_staff_lines(binary, check_radius=3):
    """
    Remove staff lines from binary image while preserving symbols.
    Only erases a dark pixel if there are no dark pixels 
    above or below it within check_radius rows.
    
    Returns cleaned binary image.
    """
    height, width = binary.shape
    groups = find_staff_line_groups(binary)
    result = binary.copy()

    for group in groups:
        y_start = max(1, group[0] - 1)
        y_end = min(height - 2, group[-1] + 1)

        above_start = max(0, y_start - check_radius)
        above_end = max(0, y_start - 1)
        below_start = min(height - 1, y_end + 1)
        below_end = min(height - 1, y_end + check_radius)

        for y in range(y_start, y_end + 1):
            dark_pixels = result[y, :] == 0
            if not np.any(dark_pixels):
                continue

            has_above = np.any(result[above_start:above_end+1, :] == 0, axis=0) \
                if above_start < above_end else np.zeros(width, dtype=bool)
            has_below = np.any(result[below_start:below_end+1, :] == 0, axis=0) \
                if below_start < below_end else np.zeros(width, dtype=bool)

            erase = dark_pixels & ~has_above & ~has_below
            result[y, erase] = 255

    return result

def clean_staff_image(img):
    """
    Full cleaning pipeline for a staff crop:
    1. Binarize
    2. Remove staff lines
    
    Returns cleaned binary image ready for symbol detection.
    """
    binary = binarize(img)
    cleaned = remove_staff_lines(binary)
    return cleaned