# import cv2
# import numpy as np

# def apply_simple(image_path, material):
#     image = cv2.imread(image_path)
    
# def tile_texture(texture, shape):
#     h, w = shape[:2]
#     th, tw = texture.shape[:2]

#     tiled = np.tile(texture, (h//th + 1, w//tw + 1, 1))
#     return tiled[:h, :w]

#     if material == "paint":
#         def apply_paint(image, color):
#             overlay = np.full_like(image, color)

#             blended = cv2.addWeighted(image, 0.85, overlay, 0.15, 0)

#             gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#             gray = cv2.GaussianBlur(gray, (21,21), 0)
#             gray = gray / 255.0
#             gray = np.stack([gray]*3, axis=-1)

#             blended = (blended * gray).astype("uint8")

#             return blended

#     elif material == "tiles":
#         texture = cv2.imread("static/textures/tile.jpg")
#         texture = tile_texture(texture, image.shape)
#         texture = cv2.GaussianBlur(texture, (3,3), 0)

#         output = cv2.addWeighted(image, 0.75, texture, 0.25, 0)

#     else:
#         texture = cv2.imread("static/textures/plaster.jpg")
#         texture = tile_texture(texture, image.shape)

#         texture = cv2.GaussianBlur(texture, (11,11), 0)
#         texture = cv2.addWeighted(texture, 0.6, image, 0.4, 0)

#         output = cv2.addWeighted(image, 0.85, texture, 0.15, 0)

#     h, w, _ = image.shape

#     mask = np.zeros_like(image)
#     mask[:int(h*0.65), :] = 1

#     output = image * (1 - mask) + output * mask
#     output = output.astype("uint8")
#     cv2.imwrite(output_path, output)

#     return output_path



import cv2
import numpy as np

# 🔹 Helper: tile texture properly
def tile_texture(texture, shape):
    h, w = shape[:2]
    th, tw = texture.shape[:2]

    tiled = np.tile(texture, (h//th + 1, w//tw + 1, 1))
    return tiled[:h, :w]

# 🔹 Helper: paint logic
def apply_paint(image, color):
    overlay = np.full_like(image, color)

    blended = cv2.addWeighted(image, 0.9, overlay, 0.1, 0)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21,21), 0)
    gray = gray / 255.0
    gray = np.stack([gray]*3, axis=-1)

    blended = (blended * gray).astype("uint8")

    return blended

# 🔥 MAIN FUNCTION
def apply_simple(image_path, material):
    image = cv2.imread(image_path)

    # default output path
    output_path = "static/outputs/output.jpg"

    if material == "paint":
        # light beige color (change if needed)
        color = (200, 200, 180)
        output = apply_paint(image, color)

    elif material == "tiles":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (9,9), 0)

        gray = np.stack([gray]*3, axis=-1)

        output = cv2.addWeighted(image, 0.8, gray, 0.2, 0)

    elif material == "plaster":
        blur = cv2.GaussianBlur(image, (21,21), 0)

        output = cv2.addWeighted(image, 0.9, blur, 0.1, 0)
    else:
        output = image.copy()

    # 🔥 MASK (wall region only)
    h, w, _ = image.shape
    mask = np.zeros_like(image)
    mask[:int(h*0.65), :] = 1

    output = image * (1 - mask) + output * mask
    output = output.astype("uint8")

    cv2.imwrite(output_path, output)

    return output_path