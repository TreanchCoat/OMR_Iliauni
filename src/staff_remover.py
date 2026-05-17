import cv2
import numpy as np


# Default binarisation method.  See ``binarize`` below for the menu.
DEFAULT_BINARIZE_METHOD = 'flatten'

# Higher sensitivity = MORE pixels classified as ink (less aggressive
# removal).  Applied as a positive bias on top of the Otsu threshold;
# 0 = pure Otsu, 30 = noticeably more ink retained, 60 = quite forgiving.
# Range roughly 0..80.  Set this high if the binarised output is
# eating thin strokes.
DEFAULT_BINARIZE_SENSITIVITY = 45


def binarize(img,
             method: str = DEFAULT_BINARIZE_METHOD,
             sensitivity: int = DEFAULT_BINARIZE_SENSITIVITY):
    """
    Convert a BGR / grayscale image to a binary image (255=paper, 0=ink).

    Why background-flattening is the default
    ----------------------------------------
    Music scans (especially handwritten or photographed pages) almost
    always have uneven illumination — one side of the page is brighter
    than the other, scanner shading, paper aging, etc.  A single
    global threshold (the old ``cv2.threshold(gray, 128, …)`` or even
    plain Otsu) inevitably turns thin ink in the dimmer regions into
    broken dotted patterns because the same threshold can't fit both
    halves of the page.

    The default 'flatten' method neutralises illumination first:

        bg = GaussianBlur(gray, ksize≈page/14)        # local paper colour
        flat = gray / bg * 255                         # ink stands out
        Otsu(flat)                                     # global threshold OK

    A small morphological close fills 1-px holes that opacity gradients
    sometimes leave in thick strokes.

    Method menu
    -----------
    ``flatten``   background-flatten, then Otsu  (default — recommended)
    ``otsu``      plain Otsu on the original grayscale
    ``adaptive``  cv2.adaptiveThreshold(GAUSSIAN_C, block=31, C=10)
    ``fixed``     the legacy hard cutoff at 128 (kept for comparison)
    """
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy() if img.dtype == np.uint8 else img.astype(np.uint8)

    if method == 'fixed':
        _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
        return binary

    if method == 'otsu':
        # Compute Otsu's threshold then bias it UP by `sensitivity`.
        # Higher threshold (with THRESH_BINARY) classifies more pixels
        # as ink (gray <= threshold → 0), so the result keeps more
        # weak strokes.
        otsu_t, _ = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        t = max(0, min(255, int(otsu_t) + int(sensitivity)))
        _, binary = cv2.threshold(gray, t, 255, cv2.THRESH_BINARY)
        return _fill_ink_pinholes(binary)

    if method == 'adaptive':
        # blockSize tuned for ~9-px staff line spacing.  Must be odd.
        # ``C`` shifts the local threshold: smaller C keeps more pixels
        # as ink, so we map `sensitivity` onto C inversely.
        block = max(11, (gray.shape[0] // 25) | 1)
        C = max(-30, 10 - int(sensitivity) // 3)
        bin_ = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=block, C=C)
        return _fill_ink_pinholes(bin_)

    if method == 'flatten':
        # Background estimate: a wide Gaussian blur of the gray image.
        # ksize sized to ~1/14 of the page width — large enough that
        # staff lines and noteheads are blurred away, small enough that
        # real shading is captured.
        h, w = gray.shape
        ksize = max(31, ((w // 14) | 1))  # odd, ≥31
        bg = cv2.GaussianBlur(gray, (ksize, ksize), 0)
        bg = np.clip(bg, 1, 255).astype(np.uint8)
        # Normalise ink with respect to local background.  cv2.divide
        # with scale=255 maps the ratio back into 0..255.
        flat = cv2.divide(gray, bg, scale=255)
        # Otsu picks the optimum split, then we bias up by `sensitivity`
        # to keep more weak ink.
        otsu_t, _ = cv2.threshold(
            flat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        t = max(0, min(255, int(otsu_t) + int(sensitivity)))
        _, binary = cv2.threshold(flat, t, 255, cv2.THRESH_BINARY)
        return _fill_ink_pinholes(binary)

    raise ValueError(
        f"Unknown binarize method {method!r}.  Valid options: "
        "'flatten' (default), 'otsu', 'adaptive', 'fixed'.")


def _fill_ink_pinholes(binary, kernel_size: int = 2):
    """
    Fill 1-pixel pinholes that occasionally appear inside ink strokes
    after thresholding (caused by uneven ink density / scanner noise).

    On our paper=255 / ink=0 convention this is a MORPH_OPEN operation:
    OPEN = DILATE(ERODE(I)) — small islands of WHITE inside a dark ink
    region get eroded away, then the remaining ink expands back to its
    original boundary.  Net effect: tiny white dots embedded in ink
    are filled with ink; ink edges keep their shape.
    """
    if kernel_size <= 0:
        return binary
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (kernel_size, kernel_size))
    return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

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