# import cv2
# import numpy as np

# def apply_advanced(image_path, material):
#     image = cv2.imread(image_path)

#     # Edge detection (simulate segmentation)
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     edges = cv2.Canny(gray, 100, 200)

#     mask = cv2.dilate(edges, None)
#     mask = cv2.GaussianBlur(mask, (5,5), 0)

#     mask = mask / 255.0
#     mask = np.stack([mask]*3, axis=-1)

#     if material == "paint":
#         color = np.full_like(image, (255, 0, 0))
#         overlay = cv2.addWeighted(image, 0.5, color, 0.5, 0)

#     else:
#         overlay = image

#     output = image * (1 - mask) + overlay * mask

#     output = output.astype("uint8")

#     output_path = "static/outputs/advanced.jpg"
#     cv2.imwrite(output_path, output)

#     return output_path

import cv2
import numpy as np

def apply_advanced(image_path, material):
    image = cv2.imread(image_path)

    # 🔹 Edge-based mask (soft segmentation)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)

    edges = cv2.GaussianBlur(edges, (7,7), 0)
    edges = edges / 255.0
    edges = np.stack([edges]*3, axis=-1)

    # 🔹 MATERIAL LOGIC
    if material == "paint":
        color = np.full_like(image, (200, 200, 180))  # soft paint
        overlay = cv2.addWeighted(image, 0.85, color, 0.15, 0)

    elif material == "tiles":
        texture = cv2.imread("static/textures/tile.jpg")

        # tile properly
        h, w = image.shape[:2]
        th, tw = texture.shape[:2]
        texture = np.tile(texture, (h//th + 1, w//tw + 1, 1))[:h, :w]

        texture = cv2.GaussianBlur(texture, (3,3), 0)
        overlay = cv2.addWeighted(image, 0.75, texture, 0.25, 0)

    elif material == "plaster":
        texture = cv2.imread("static/textures/plaster.jpg")

        h, w = image.shape[:2]
        th, tw = texture.shape[:2]
        texture = np.tile(texture, (h//th + 1, w//tw + 1, 1))[:h, :w]

        texture = cv2.GaussianBlur(texture, (11,11), 0)
        texture = cv2.addWeighted(texture, 0.6, image, 0.4, 0)

        overlay = cv2.addWeighted(image, 0.85, texture, 0.15, 0)

    else:
        overlay = image.copy()

    # 🔥 BLEND USING EDGE MASK
    output = image * (1 - edges) + overlay * edges
    output = output.astype("uint8")

    # 🔥 LIMIT TO WALL REGION
    h, w, _ = image.shape
    region_mask = np.zeros_like(image)
    region_mask[:int(h*0.65), :] = 1

    output = image * (1 - region_mask) + output * region_mask
    output = output.astype("uint8")

    output_path = "static/outputs/advanced.jpg"
    cv2.imwrite(output_path, output)

    return output_path