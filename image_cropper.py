import cv2
import numpy as np
from staff_detector import detect_staves

def crop_staff(img, img_h, img_w, staff, padding_ratio=0.95):
    """
    Crop a single staff from the full image using full width.
    
    Args:
        img           - full page BGR image
        img_h, img_w  - image dimensions
        staff         - staff dict from detect_staves()
        padding_ratio - how much vertical padding to add 
                        relative to staff height (default 95%)
    
    Returns:
        crop  - cropped BGR image (full width, padded height)
        meta  - dict with:
                offset_y    - y position of crop top in original image
                staff       - original staff dict (coordinates in full image)
                img_w       - original image width
                img_h       - original image height
    """
    pad = staff['height'] * padding_ratio
    crop_y1 = max(0, int(staff['y1'] - pad))
    crop_y2 = min(img_h, int(staff['y2'] + pad))

    crop = img[crop_y1:crop_y2, 0:img_w]

    meta = {
        'offset_y': crop_y1,
        'offset_x': 0,
        'staff': staff,
        'img_w': img_w,
        'img_h': img_h
    }
    return crop, meta

def get_all_crops(img_path, padding_ratio=0.95):
    """
    Detect all staves in an image and return their crops with metadata.
    
    Returns list of (crop_img, meta) tuples, one per staff,
    sorted top to bottom.
    
    Example:
        crops = get_all_crops('page1.png')
        for crop, meta in crops:
            # crop is the staff image
            # meta['staff']['lines'] has the 5 line y positions
            # meta['offset_y'] lets you convert crop coords back to page coords
    """
    img, img_h, img_w, staves = detect_staves(img_path)

    if not staves:
        return []

    crops = []
    for idx, staff in enumerate(staves):
        crop, meta = crop_staff(img, img_h, img_w, staff, padding_ratio)
        meta['staff_idx'] = idx
        crops.append((crop, meta))

    return crops