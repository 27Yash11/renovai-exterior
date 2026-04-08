import cv2
import numpy as np
import os

os.makedirs('static/textures', exist_ok=True)

# 1. Tile Texture (Checkerboard-like)
tile = np.zeros((200, 200, 3), dtype=np.uint8)
tile[:] = (220, 220, 220)
cv2.rectangle(tile, (0, 0), (100, 100), (200, 200, 200), -1)
cv2.rectangle(tile, (100, 100), (200, 200), (200, 200, 200), -1)
cv2.imwrite('static/textures/tile.jpg', tile)

# 2. Stone Texture (Random noise + blur)
stone = np.random.randint(100, 180, (200, 200, 3), dtype=np.uint8)
stone = cv2.GaussianBlur(stone, (5, 5), 0)
# Add some lines to mimic stone blocks
for i in range(0, 200, 40):
    cv2.line(stone, (0, i), (200, i), (50, 50, 50), 2)
for i in range(0, 200, 50):
    offset = 20 if (i//50)%2 == 0 else 0
    cv2.line(stone, (i, 0), (i, 200), (50, 50, 50), 2)
cv2.imwrite('static/textures/stone.jpg', stone)

# 3. Plaster/Texture (Fine noise)
plaster = np.random.randint(200, 240, (200, 200, 3), dtype=np.uint8)
plaster = cv2.GaussianBlur(plaster, (3, 3), 0)
cv2.imwrite('static/textures/plaster.jpg', plaster)

print("Textures generated.")
