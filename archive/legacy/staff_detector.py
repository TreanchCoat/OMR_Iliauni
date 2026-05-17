from ultralytics import YOLO
import cv2

_model = None

def load_model(model_path=r'S:\saved_models\deepscores_crops_v1.pt'):
    global _model
    if _model is None:
        _model = YOLO(model_path)
    return _model

def is_duplicate(new_staff, existing_staves, overlap_thresh=0.5):
    for s in existing_staves:
        y_overlap = min(new_staff['y2'], s['y2']) - max(new_staff['y1'], s['y1'])
        if y_overlap <= 0:
            continue
        if y_overlap / (new_staff['y2'] - new_staff['y1']) > overlap_thresh:
            return True
    return False

def detect_staves(img_path, conf=0.15, iou=0.3):
    """
    Detect all staff lines in an image.
    
    Returns:
        img       - loaded BGR image
        staves    - list of dicts, each with:
                    y1, y2, x1, x2  - bounding box in original image pixels
                    height          - y2 - y1
                    spacing         - distance between staff lines (height/4)
                    lines           - list of 5 y positions for each staff line
    """
    model = load_model()
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f'Could not read image: {img_path}')

    img_h, img_w = img.shape[:2]
    results = model(img_path, imgsz=640, conf=conf, iou=iou, verbose=False)[0]

    staves = []
    for box in results.boxes:
        if model.names[int(box.cls)] != 'staff':
            continue
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
        w = x2 - x1
        h = y2 - y1
        if h == 0 or (w / h) < 8:
            continue
        staff = {
            'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
            'height': h,
            'spacing': h / 4,
            'lines': [y1 + (h / 4) * i for i in range(5)]
        }
        if not is_duplicate(staff, staves):
            staves.append(staff)

    staves.sort(key=lambda s: s['y1'])
    return img, img_h, img_w, staves