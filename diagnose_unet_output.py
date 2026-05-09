# Save as S:\omr\diagnose_unet_output.py and run:
# python diagnose_unet_output.py input\score.png

import sys
import cv2
import numpy as np
import torch

sys.path.insert(0, '.')
from staff_detector_unet import predict_mask, _extract_staff_lines_from_mask, _group_lines_into_staves

img = cv2.imread(sys.argv[1])
print(f'Image size: {img.shape[1]}x{img.shape[0]}')

prob, binary = predict_mask(img, threshold=0.5)
print(f'Foreground px: {(binary > 0).sum()} ({100*(binary>0).mean():.2f}%)')

lines = _extract_staff_lines_from_mask(binary)
print(f'\nExtracted {len(lines)} candidate lines:')
for i, l in enumerate(lines):
    print(f'  Line {i+1:2d}: y={l["mean_y"]:6.1f}  '
          f'x={l["left_x"]}-{l["right_x"]}  '
          f'width={l["right_x"]-l["left_x"]}  '
          f'track_pts={len(l["track"])}')

if len(lines) >= 2:
    ys = np.array([l['mean_y'] for l in lines])
    gaps = np.diff(sorted(ys))
    print(f'\nGaps between consecutive lines: {np.round(gaps, 1).tolist()}')
    print(f'Median gap: {np.median(gaps):.1f}')
    print(f'Min gap:    {gaps.min():.1f}')
    print(f'Max gap:    {gaps.max():.1f}')

grouped = _group_lines_into_staves(lines)
print(f'\nGrouped into {len(grouped)} staves')