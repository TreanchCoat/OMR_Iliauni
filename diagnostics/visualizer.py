import sys
sys.path.append(r'S:\omr')

from ultralytics import YOLO
import cv2
import os
import json
from pathlib import Path

MODEL_PATH = r'S:\saved_models\deepscores_crops_v1.pt'
_model = None

def get_model():
    global _model
    if _model is None:
        _model = YOLO(MODEL_PATH)
    return _model

def visualize_single(img_path, conf=0.2, iou=0.5):
    """
    Run detection on a single image.
    Returns annotated image and list of detections.
    
    Each detection is a dict with:
        class_id, class_name, conf, x1, y1, x2, y2, cx, cy
    """
    model = get_model()
    results = model(img_path, imgsz=640, conf=conf, iou=iou, verbose=False)[0]

    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f'Could not read: {img_path}')

    if len(img.shape) == 2 or img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        cls_id = int(box.cls)
        cls_name = model.names[cls_id]
        score = float(box.conf)

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.putText(img, f'{cls_name} {score:.2f}',
                    (x1, max(0, y1-3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

        detections.append({
            'class_id': cls_id,
            'class_name': cls_name,
            'conf': round(score, 4),
            'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
            'cx': (x1 + x2) // 2,
            'cy': (y1 + y2) // 2
        })

    # Sort detections left to right
    detections.sort(key=lambda d: d['cx'])

    return img, detections

def visualize_directory(in_dir, out_dir=None, suffix='_clean', conf=0.2):
    """
    Run detection on all cleaned images in a directory.
    Saves for each image:
        - annotated PNG with boxes drawn
        - JSON file with all detection data
        - TXT summary with class names and positions
    
    Returns dict mapping image name -> list of detections.
    """
    if out_dir is None:
        out_dir = os.path.join(in_dir, 'visualized')
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    if suffix:
        imgs = [f for f in os.listdir(in_dir)
                if f.endswith('.png') and suffix in f]
    else:
        imgs = [f for f in os.listdir(in_dir) if f.endswith('.png')]

    if not imgs:
        print(f'No images found with suffix "{suffix}"')
        return {}

    print(f'Visualizing {len(imgs)} images...')

    all_results = {}

    for img_name in imgs:
        img_path = os.path.join(in_dir, img_name)
        stem = Path(img_name).stem

        try:
            annotated, detections = visualize_single(img_path, conf=conf)

            # Save annotated image
            cv2.imwrite(os.path.join(out_dir, img_name), annotated)

            # Save JSON with full detection data
            json_path = os.path.join(out_dir, f'{stem}.json')
            with open(json_path, 'w') as f:
                json.dump({
                    'image': img_name,
                    'total_detections': len(detections),
                    'detections': detections
                }, f, indent=2)

            # Save human readable TXT summary
            txt_path = os.path.join(out_dir, f'{stem}.txt')
            with open(txt_path, 'w') as f:
                f.write(f'Image: {img_name}\n')
                f.write(f'Total detections: {len(detections)}\n')
                f.write('-' * 50 + '\n')
                for d in detections:
                    f.write(f'{d["class_name"]:30s} '
                            f'conf={d["conf"]:.2f}  '
                            f'cx={d["cx"]:4d}  cy={d["cy"]:4d}  '
                            f'box=({d["x1"]},{d["y1"]},{d["x2"]},{d["y2"]})\n')

            all_results[img_name] = detections
            print(f'  {img_name}: {len(detections)} detections')

        except Exception as e:
            print(f'  {img_name}: ERROR - {e}')

    # Save combined JSON for all images
    combined_path = os.path.join(out_dir, 'all_detections.json')
    with open(combined_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f'\nSaved to {out_dir}')
    print(f'Combined results: {combined_path}')
    return all_results


if __name__ == '__main__':
    results = visualize_directory(
        in_dir=r'S:\omr\preprocess_test',
        out_dir=r'S:\omr\preprocess_test\visualized',
        suffix='_clean'
    )

    # Print summary
    print('\nDetection summary:')
    for img_name, detections in results.items():
        classes = {}
        for d in detections:
            classes[d['class_name']] = classes.get(d['class_name'], 0) + 1
        print(f'\n{img_name}:')
        for cls, count in sorted(classes.items()):
            print(f'  {cls}: {count}')