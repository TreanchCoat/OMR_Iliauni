import cv2
import numpy as np
from pathlib import Path


def order_points(pts):
    pts = np.array(pts, dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    rect = np.zeros((4, 2), dtype="float32")
    rect[0] = pts[np.argmin(s)]      # top-left
    rect[2] = pts[np.argmax(s)]      # bottom-right
    rect[1] = pts[np.argmin(diff)]   # top-right
    rect[3] = pts[np.argmax(diff)]   # bottom-left
    return rect


def perspective_transform(image):
    """
    Automatically tries to detect the paper boundary and rectify it.
    If it fails, returns the original image.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blur, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    page_contour = None

    for c in contours[:10]:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4:
            page_contour = approx.reshape(4, 2)
            break

    if page_contour is None:
        print("Page boundary not found. Using original image.")
        return image

    rect = order_points(page_contour)
    tl, tr, br, bl = rect

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = int(max(width_a, width_b))

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = int(max(height_a, height_b))

    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (max_width, max_height))

    return warped


def binarize_music_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Adaptive threshold works better for photographed paper
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        35,
        15
    )

    return binary


def detect_stafflines(binary, min_width_ratio=0.25):
    """
    Detect horizontal staffline candidates.

    Returns:
        lines: list of (x1, y1, x2, y2)
        staff_mask: binary image containing isolated staffline pixels
    """
    h, w = binary.shape

    # Horizontal kernel length should be large enough to preserve stafflines
    # but not too large, because some stafflines may be short.
    kernel_len = max(30, w // 25)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))

    staff_mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)

    # Connect small gaps caused by noteheads, noise, or scanning artifacts
    connect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(10, w // 100), 1))
    staff_mask = cv2.dilate(staff_mask, connect_kernel, iterations=1)

    contours, _ = cv2.findContours(staff_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []

    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)

        # Stafflines are long and thin
        if cw < min_width_ratio * w:
            continue
        if ch > 12:
            continue

        points = c.reshape(-1, 2)

        # Robust line fitting
        vx, vy, x0, y0 = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
        vx, vy, x0, y0 = vx[0], vy[0], x0[0], y0[0]

        # Reject strongly non-horizontal lines
        slope = vy / (vx + 1e-8)
        if abs(slope) > 0.05:
            continue

        x1 = x
        x2 = x + cw
        y1 = int(y0 + slope * (x1 - x0))
        y2 = int(y0 + slope * (x2 - x0))

        candidates.append((x1, y1, x2, y2, cw))

    # Sort by vertical position
    candidates = sorted(candidates, key=lambda line: (line[1] + line[3]) / 2)

    # Merge duplicate detections very close in y
    merged = []
    for line in candidates:
        x1, y1, x2, y2, cw = line
        y_mid = (y1 + y2) / 2

        if not merged:
            merged.append(line)
            continue

        px1, py1, px2, py2, pcw = merged[-1]
        py_mid = (py1 + py2) / 2

        if abs(y_mid - py_mid) <= 3:
            # Keep the longer one
            if cw > pcw:
                merged[-1] = line
        else:
            merged.append(line)

    lines = [(x1, y1, x2, y2) for x1, y1, x2, y2, _ in merged]
    return lines, staff_mask


def group_staffs(lines, tolerance_factor=1.6):
    """
    Groups detected lines into staffs.
    Standard staff has 5 lines.
    """
    if len(lines) < 5:
        return []

    y_values = np.array([(y1 + y2) / 2 for x1, y1, x2, y2 in lines])
    gaps = np.diff(y_values)

    # Estimate staffspace as the most common small gap
    small_gaps = gaps[gaps < np.percentile(gaps, 70)]
    if len(small_gaps) == 0:
        return []

    staffspace = np.median(small_gaps)
    max_gap_inside_staff = tolerance_factor * staffspace

    groups = []
    current = [lines[0]]

    for i in range(1, len(lines)):
        prev_y = y_values[i - 1]
        curr_y = y_values[i]

        if curr_y - prev_y <= max_gap_inside_staff:
            current.append(lines[i])
        else:
            if len(current) >= 4:
                groups.append(current)
            current = [lines[i]]

    if len(current) >= 4:
        groups.append(current)

    # Prefer exactly 5-line groups where possible
    final_groups = []
    for g in groups:
        if len(g) == 5:
            final_groups.append(g)
        elif len(g) > 5:
            # Split into chunks of 5
            for i in range(0, len(g), 5):
                chunk = g[i:i+5]
                if len(chunk) >= 4:
                    final_groups.append(chunk)
        else:
            final_groups.append(g)

    return final_groups


def draw_stafflines(image, staff_groups):
    output = image.copy()

    for staff_id, staff in enumerate(staff_groups):
        for line in staff:
            x1, y1, x2, y2 = line
            cv2.line(output, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Draw staff bounding box
        xs = []
        ys = []
        for x1, y1, x2, y2 in staff:
            xs.extend([x1, x2])
            ys.extend([y1, y2])

        cv2.rectangle(
            output,
            (min(xs), min(ys) - 5),
            (max(xs), max(ys) + 5),
            (0, 255, 0),
            2
        )

        cv2.putText(
            output,
            f"Staff {staff_id + 1}",
            (min(xs), min(ys) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2
        )

    return output


def process_music_page(input_path, output_dir="output"):
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    original = cv2.imread(str(input_path))
    if original is None:
        raise FileNotFoundError(f"Cannot load image: {input_path}")

    warped = perspective_transform(original)
    binary = binarize_music_image(warped)

    lines, staff_mask = detect_stafflines(binary)
    staff_groups = group_staffs(lines)

    result = draw_stafflines(warped, staff_groups)

    cv2.imwrite(str(output_dir / "01_original.png"), original)
    cv2.imwrite(str(output_dir / "02_homography_transformed.png"), warped)
    cv2.imwrite(str(output_dir / "03_binary.png"), binary)
    cv2.imwrite(str(output_dir / "04_staff_mask.png"), staff_mask)
    cv2.imwrite(str(output_dir / "05_stafflines_detected.png"), result)

    print(f"Detected raw staffline candidates: {len(lines)}")
    print(f"Detected staff groups: {len(staff_groups)}")
    print(f"Saved results to: {output_dir}")

    return {
        "original": original,
        "warped": warped,
        "binary": binary,
        "staff_mask": staff_mask,
        "lines": lines,
        "staff_groups": staff_groups,
        "result": result
    }


# Example usage
results = process_music_page("score.png")