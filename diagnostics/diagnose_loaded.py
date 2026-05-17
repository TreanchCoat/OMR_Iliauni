"""Verify what the dataset loader actually produces."""
import sys, json
sys.path.insert(0, '.')
from staff_unet_train import StafflineDataset
import numpy as np
import cv2

with open('muscima_manifest.json') as f:
    manifest = json.load(f)

ds = StafflineDataset(manifest['train'][:1], size=512, augment=False)
img_t, mask_t = ds[0]

img = img_t.numpy().transpose(1, 2, 0)   # CHW → HWC
mask = mask_t.numpy()[0]

print(f'Image: shape={img.shape}, range=[{img.min():.3f}, {img.max():.3f}], mean={img.mean():.3f}')
print(f'  → as uint8: % black (<0.5) = {100 * (img < 0.5).mean():.2f}%')
print(f'Mask:  shape={mask.shape}, unique={np.unique(mask)[:5]}, mean={mask.mean():.3f}')
print(f'  → % positive (==1.0) = {100 * (mask > 0.5).mean():.2f}%')

# Save
cv2.imwrite('check_img.png', (img * 255).astype(np.uint8)[:,:,::-1])
cv2.imwrite('check_mask.png', (mask * 255).astype(np.uint8))
print('Saved check_img.png and check_mask.png')