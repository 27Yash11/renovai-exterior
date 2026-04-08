import cv2
import numpy as np
import os
import base64
from PIL import Image

def generate_base_mask(image_path):
    """
    Fast wall detection using OpenCV only (no deep learning).
    Takes 2-3 seconds instead of 30+ seconds.
    """
    img_cv = cv2.imread(image_path)
    if img_cv is None:
        raise ValueError("Image could not be loaded")
    
    h, w = img_cv.shape[:2]
    
    # Resize if too large (speeds up processing)
    max_dim = 1200
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img_cv = cv2.resize(img_cv, (int(w * scale), int(h * scale)))
        h, w = img_cv.shape[:2]
    
    print(f"  Processing {w}×{h} image...")
    
    # ── Step 1: GrabCut to isolate building ──────────────────────────────────
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    mask = np.zeros((h, w), np.uint8)
    mask[:] = cv2.GC_BGD
    
    # Conservative rectangle - building in center 70%
    rect = (int(w * 0.15), int(h * 0.15), int(w * 0.85), int(h * 0.85))
    cv2.rectangle(mask, (rect[0], rect[1]), (rect[2], rect[3]), cv2.GC_PR_FGD, -1)
    
    hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
    hue, sat, val = cv2.split(hsv)
    y_idx = np.indices((h, w))[0]
    
    # Remove obvious background
    mask[(hue > 95) & (hue < 135) & (sat > 30) & (val > 120)] = cv2.GC_BGD  # blue sky
    mask[(sat < 15) & (val < 40)] = cv2.GC_BGD  # dark areas
    mask[y_idx > int(h * 0.90)] = cv2.GC_BGD  # bottom 10% = ground
    mask[(hue > 25) & (hue < 80) & (sat > 50) & (val > 40) & (y_idx < int(h * 0.40))] = cv2.GC_BGD  # trees
    
    print("  Running GrabCut...")
    cv2.grabCut(img_cv, mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
    grabcut_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    
    # ── Step 2: Fast window/door detection using color + edges ───────────────
    print("  Detecting windows and doors...")
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    # Windows are usually dark (glass reflects less light)
    dark_mask = (val < 60).astype('uint8')
    
    # Doors/windows have strong edges
    edges = cv2.Canny(gray, 50, 150)
    edges_dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    
    # Combine: dark regions with strong edges = windows/doors
    window_door_mask = cv2.bitwise_and(dark_mask, edges_dilated)
    
    # Clean up small noise
    kernel = np.ones((5, 5), np.uint8)
    window_door_mask = cv2.morphologyEx(window_door_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    window_door_mask = cv2.morphologyEx(window_door_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Dilate to remove frames
    window_door_mask = cv2.dilate(window_door_mask, np.ones((9, 9), np.uint8), iterations=1)
    
    # ── Step 3: Combine ──────────────────────────────────────────────────────
    wall_mask_bool = np.clip(grabcut_mask - window_door_mask, 0, 1).astype('uint8')
    
    # Final cleanup
    kernel = np.ones((7, 7), np.uint8)
    wall_mask_bool = cv2.morphologyEx(wall_mask_bool, cv2.MORPH_CLOSE, kernel, iterations=2)
    wall_mask_bool = cv2.morphologyEx(wall_mask_bool, cv2.MORPH_OPEN, kernel, iterations=1)
    
    print("  Mask generated successfully!")
    
    # Save as transparent RGBA
    mask_path = image_path.replace("uploads", "masks").replace(".jpg", "_mask.png").replace(".png", "_mask.png").replace(".jpeg", "_mask.png")
    os.makedirs(os.path.dirname(mask_path), exist_ok=True)
    
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, 2] = 239   # R
    rgba[:, :, 1] = 68    # G
    rgba[:, :, 0] = 68    # B
    rgba[:, :, 3] = wall_mask_bool * 255  # Alpha
    
    cv2.imwrite(mask_path, rgba)
    return mask_path

def apply_material_with_mask(image_path, mask_data_url, material_type, material_color_hex=None, texture_path=None):
    img = cv2.imread(image_path)
    
    # Decode user-painted base64 mask
    header, encoded = mask_data_url.split(",", 1)
    mask_data = base64.b64decode(encoded)
    np_arr = np.frombuffer(mask_data, np.uint8)
    user_mask_img = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
    
    if user_mask_img.shape[-1] == 4: # RGBA
        wall_mask_bool = (user_mask_img[:, :, 3] > 10).astype('uint8')
    else:
        wall_mask_bool = (user_mask_img[:, :, 0] > 10).astype('uint8')
        
    h, w = img.shape[:2]
    
    # Feather boundary
    wall_mask_blurred = cv2.GaussianBlur(wall_mask_bool * 255, (5, 5), 0)
    wall_mask_norm = wall_mask_blurred / 255.0
    wall_mask_3d = np.stack([wall_mask_norm]*3, axis=-1)
    
    # Convert to LAB to preserve shadows
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(img_lab)
    
    if material_type == 'paint' and material_color_hex:
        h_hex = material_color_hex.lstrip('#')
        rgb = tuple(int(h_hex[i:i+2], 16) for i in (0, 2, 4))
        bgr = (rgb[2], rgb[1], rgb[0])
        
        target_bgr = np.uint8([[[bgr[0], bgr[1], bgr[2]]]])
        target_lab = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2LAB)
        
        # Replace color, keep lightness
        new_A = np.full_like(A, target_lab[0, 0, 1])
        new_B = np.full_like(B, target_lab[0, 0, 2])
        
        merged = cv2.merge([L, new_A, new_B])
        overlay = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
        
    elif texture_path:
        texture = cv2.imread(texture_path)
        if texture is not None:
            texture_resized = cv2.resize(texture, (w, h))
            L_norm = (L.astype(np.float32) / 255.0)
            L_norm = np.clip(L_norm * 1.15, 0, 1)
            L_norm_3d = np.stack([L_norm]*3, axis=-1)
            
            overlay = (texture_resized.astype(np.float32) * L_norm_3d).astype(np.uint8)
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
