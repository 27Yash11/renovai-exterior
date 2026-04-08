import cv2
import numpy as np

img = cv2.imread('static/uploads/home-design.jpg')
if img is None:
    print("Failed to load")
    exit()

# scale down for faster testing
h, w = img.shape[:2]
scale = 800 / max(h, w)
img = cv2.resize(img, (int(w*scale), int(h*scale)))
h, w = img.shape[:2]

mask = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)

# Center area is probable foreground
rect = (int(w * 0.1), int(h * 0.1), int(w * 0.9), int(h * 0.9))
cv2.rectangle(mask, (rect[0], rect[1]), (rect[2], rect[3]), cv2.GC_PR_FGD, -1)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
hue = hsv[:,:,0]
sat = hsv[:,:,1]
val = hsv[:,:,2]

# 1. Vegetation (Greenish)
veg_mask = (hue > 30) & (hue < 90) & (sat > 40)
mask[veg_mask] = cv2.GC_BGD

# 2. Sky (Blueish or very bright/low sat at top)
blue_sky = (hue > 90) & (hue < 140) & (sat > 40) & (val > 100)
mask[blue_sky] = cv2.GC_BGD

# White sky / clouds at top half
y_indices = np.indices((h, w))[0]
white_sky = (y_indices < h*0.5) & (sat < 40) & (val > 200)
mask[white_sky] = cv2.GC_BGD

# 3. Road / Ground at bottom
road = (y_indices > h*0.8) & (sat < 50) & (val < 150)
mask[road] = cv2.GC_BGD

# 4. Deep shadows or windows
dark = val < 30
mask[dark] = cv2.GC_BGD

bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)

cv2.grabCut(img, mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)

mask2 = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')

out = img.copy()
out[mask2 == 0] = (0, 0, 0)
cv2.imwrite('static/outputs/test_mask.jpg', out)
print("Saved to static/outputs/test_mask.jpg")
