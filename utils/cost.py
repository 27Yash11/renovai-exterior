import cv2

def calculate_cost(image_path):
    img = cv2.imread(image_path)
    h, w, _ = img.shape

    area = (h * w) * 0.0001
    material_cost = 50
    labor = 1000

    total = area * material_cost + labor

    return round(total, 2)