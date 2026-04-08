import cv2
import numpy as np
import os
import base64
from PIL import Image

def check_segformer_available():
    """Check if Segformer model is already downloaded."""
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    segformer_path = os.path.join(cache_dir, "models--nvidia--segformer-b0-finetuned-ade-512-512")
    return os.path.exists(segformer_path)

def generate_base_mask(image_path):
    """
    Hybrid approach: Use Segformer if available, else fast OpenCV.
    """
    use_segformer = check_segformer_available()
    
    if use_segformer:
        print("  Using Segformer AI (high accuracy)...")
        return generate_mask_segformer(image_path)
    else:
        print("  Using fast OpenCV (good accuracy, 10x faster)...")
        return generate_mask_opencv(image_path)

def generate_mask_opencv(image_path):
    """Fast OpenCV-only approach (2-3 seconds)."""
    img_cv = cv2.imread(image_path)
    if img_cv is None:
        raise ValueError("Image could not be loaded")
    
    h, w = img_cv.shape[:2]
    
    # Resize if too large
    max_dim = 1200
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img_cv = cv2.resize(img_cv, (int(w * scale), int(h * scale)))
        h, w = img_cv.shape[:2]
    
    # GrabCut
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    mask = np.zeros((h, w), np.uint8)
    mask[:] = cv2.GC_BGD
    
    rect = (int(w * 0.15), int(h * 0.15), int(w * 0.85), int(h * 0.85))
    cv2.rectangle(mask, (rect[0], rect[1]), (rect[2], rect[3]), cv2.GC_PR_FGD, -1)
    
    hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
    hue, sat, val = cv2.split(hsv)
    y_idx = np.indices((h, w))[0]
    
    mask[(hue > 95) & (hue < 135) & (sat > 30) & (val > 120)] = cv2.GC_BGD
    mask[(sat < 15) & (val < 40)] = cv2.GC_BGD
    mask[y_idx > int(h * 0.90)] = cv2.GC_BGD
    mask[(hue > 25) & (hue < 80) & (sat > 50) & (val > 40) & (y_idx < int(h * 0.40))] = cv2.GC_BGD
    
    cv2.grabCut(img_cv, mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
    grabcut_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    
    # Fast window detection
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    dark_mask = (val < 60).astype('uint8')
    edges = cv2.Canny(gray, 50, 150)
    edges_dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    window_door_mask = cv2.bitwise_and(dark_mask, edges_dilated)
    
    kernel = np.ones((5, 5), np.uint8)
    window_door_mask = cv2.morphologyEx(window_door_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    window_door_mask = cv2.morphologyEx(window_door_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    window_door_mask = cv2.dilate(window_door_mask, np.ones((9, 9), np.uint8), iterations=1)
    
    wall_mask_bool = np.clip(grabcut_mask - window_door_mask, 0, 1).astype('uint8')
    
    kernel = np.ones((7, 7), np.uint8)
    wall_mask_bool = cv2.morphologyEx(wall_mask_bool, cv2.MORPH_CLOSE, kernel, iterations=2)
    wall_mask_bool = cv2.morphologyEx(wall_mask_bool, cv2.MORPH_OPEN, kernel, iterations=1)
    
    return save_mask(image_path, wall_mask_bool)

def generate_mask_segformer(image_path):
    """Segformer approach (higher accuracy, slower)."""
    img_cv = cv2.imread(image_path)
    if img_cv is None:
        raise ValueError("Image could not be loaded")
    h, w = img_cv.shape[:2]
    
    # GrabCut
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    mask = np.zeros((h, w), np.uint8)
    mask[:] = cv2.GC_BGD
    
    rect = (int(w * 0.15), int(h * 0.15), int(w * 0.85), int(h * 0.85))
    cv2.rectangle(mask, (rect[0], rect[1]), (rect[2], rect[3]), cv2.GC_PR_FGD, -1)
    
    hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
    hue, sat, val = cv2.split(hsv)
    y_idx = np.indices((h, w))[0]
    
    mask[(hue > 95) & (hue < 135) & (sat > 30) & (val > 120)] = cv2.GC_BGD
    mask[(sat < 15) & (val < 40)] = cv2.GC_BGD
    mask[y_idx > int(h * 0.90)] = cv2.GC_BGD
    mask[(hue > 25) & (hue < 80) & (sat > 50) & (val > 40) & (y_idx < int(h * 0.40))] = cv2.GC_BGD
    
    cv2.grabCut(img_cv, mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
    grabcut_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    
    # Segformer
    from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
    import torch
    
    processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
    model = AutoModelForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
    
    img_pil = Image.open(image_path).convert("RGB")
    inputs = processor(images=img_pil, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    logits_resized = torch.nn.functional.interpolate(
        logits, size=(h, w), mode="bilinear", align_corners=False
    )
    predicted = logits_resized.argmax(dim=1).squeeze().cpu().numpy()
    
    carve_classes = [8, 14, 149, 36, 25, 53, 15, 30, 62, 76]
    carve_mask = np.isin(predicted, carve_classes).astype('uint8')
    carve_dilated = cv2.dilate(carve_mask, np.ones((9, 9), np.uint8), iterations=1)
    
    wall_mask_bool = np.clip(grabcut_mask - carve_dilated, 0, 1).astype('uint8')
    
    kernel = np.ones((7, 7), np.uint8)
    wall_mask_bool = cv2.morphologyEx(wall_mask_bool, cv2.MORPH_CLOSE, kernel, iterations=2)
    wall_mask_bool = cv2.morphologyEx(wall_mask_bool, cv2.MORPH_OPEN, kernel, iterations=1)
    
    return save_mask(image_path, wall_mask_bool)

def save_mask(image_path, wall_mask_bool):
    """Save mask as RGBA PNG."""
    h, w = wall_mask_bool.shape
    mask_path = image_path.replace("uploads", "masks").replace(".jpg", "_mask.png").replace(".png", "_mask.png").replace(".jpeg", "_mask.png")
    os.makedirs(os.path.dirname(mask_path), exist_ok=True)
    
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, 2] = 239
    rgba[:, :, 1] = 68
    rgba[:, :, 0] = 68
    rgba[:, :, 3] = wall_mask_bool * 255
    
    cv2.imwrite(mask_path, rgba)
    return mask_path

def apply_material_with_mask(image_path, mask_data_url, material_type, material_color_hex=None, texture_path=None):
    img = cv2.imread(image_path)
    
    header, encoded = mask_data_url.split(",", 1)
    mask_data = base64.b64decode(encoded)
    np_arr = np.frombuffer(mask_data, np.uint8)
    user_mask_img = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
    
    if user_mask_img.shape[-1] == 4:
        wall_mask_bool = (user_mask_img[:, :, 3] > 10).astype('uint8')
    else:
        wall_mask_bool = (user_mask_img[:, :, 0] > 10).astype('uint8')
        
    h, w = img.shape[:2]
    
    wall_mask_blurred = cv2.GaussianBlur(wall_mask_bool * 255, (5, 5), 0)
    wall_mask_norm = wall_mask_blurred / 255.0
    wall_mask_3d = np.stack([wall_mask_norm]*3, axis=-1)
    
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(img_lab)
    
    if material_type == 'paint' and material_color_hex:
        h_hex = material_color_hex.lstrip('#')
        rgb = tuple(int(h_hex[i:i+2], 16) for i in (0, 2, 4))
        bgr = (rgb[2], rgb[1], rgb[0])
        
        target_bgr = np.uint8([[[bgr[0], bgr[1], bgr[2]]]])
        target_lab = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2LAB)
        
        # Luminance-weighted tinting (prevents painting over bright sun/white trim)
        l_norm = L.astype(np.float32) / 255.0
        tint_strength = 1.0 - np.power(l_norm, 3)
        tint_strength = np.clip(tint_strength, 0.4, 1.0)
        
        new_A = A.astype(np.float32) * (1 - tint_strength) + target_lab[0, 0, 1] * tint_strength
        new_B = B.astype(np.float32) * (1 - tint_strength) + target_lab[0, 0, 2] * tint_strength
        
        merged = cv2.merge([L, new_A.astype(np.uint8), new_B.astype(np.uint8)])
        overlay = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
        
    elif texture_path:
        texture = cv2.imread(texture_path)
        if texture is not None:
            # Physical tiling logic
            door_height_pixels = int(h * 0.25)
            pixels_per_meter = door_height_pixels / 2.13
            
            # Real-world material sizing
            if 'tile' in texture_path.lower():
                phys_w, phys_h = 0.60, 0.60  # 60x60cm tiles
            elif 'stone' in texture_path.lower():
                phys_w, phys_h = 0.45, 0.30  # 45x30cm stone cladding
            else:
                phys_w, phys_h = 1.00, 1.00  # 1m texture patch

            tile_pw = max(10, int(phys_w * pixels_per_meter))
            tile_ph = max(10, int(phys_h * pixels_per_meter))
            
            tex_small = cv2.resize(texture, (tile_pw, tile_ph), interpolation=cv2.INTER_AREA)
            
            repeats_y = int(np.ceil(h / tile_ph))
            repeats_x = int(np.ceil(w / tile_pw))
            tiled_tex = np.tile(tex_small, (repeats_y, repeats_x, 1))
            tiled_tex = tiled_tex[:h, :w, :]
            
            # Apply CLAHE to tiled texture for rich details
            tiled_lab = cv2.cvtColor(tiled_tex, cv2.COLOR_BGR2LAB)
            tl_L, tl_A, tl_B = cv2.split(tiled_lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            tl_L = clahe.apply(tl_L)
            
            # Combine 3D shadows (L) with flat Tiled Texture (tiled_tex)
            L_norm = (L.astype(np.float32) / 255.0)
            L_norm = np.clip(L_norm * 1.5, 0, 1.2)
            L_norm_3d = np.stack([L_norm]*3, axis=-1)
            
            tiled_enriched = cv2.cvtColor(cv2.merge([tl_L, tl_A, tl_B]), cv2.COLOR_LAB2BGR)
            overlay = (tiled_enriched.astype(np.float32) * L_norm_3d).astype(np.uint8)
            overlay = np.clip(overlay, 0, 255)
        else:
            overlay = img
    else:
        overlay = img
        
    final_img = img.astype(np.float32) * (1 - wall_mask_3d) + overlay.astype(np.float32) * wall_mask_3d
    final_img = final_img.astype(np.uint8)
    
    output_path = image_path.replace("uploads", "outputs").replace(".jpg", f"_{material_type}.jpg").replace(".png", f"_{material_type}.png").replace(".jpeg", f"_{material_type}.jpeg")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, final_img)
    
    masked_pixels = np.sum(wall_mask_bool > 0)
    
    return output_path, masked_pixels
