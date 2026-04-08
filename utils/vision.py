import cv2
import numpy as np
import os
import base64

from PIL import Image

# Global model cache to avoid slow reloading on every request
processor = None
model = None

def get_segmentation_model():
    global processor, model
    if processor is None or model is None:
        from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
        # 14MB lightweight model trained specifically on housing and architectural datasets (ADE20K)
        processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
        model = AutoModelForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
    return processor, model

def generate_base_mask(image_path):
    """
    Generate a wall-only mask using pure semantic segmentation (Segformer on ADE20K).
    No GrabCut — fully model-driven, robust to all lighting and image styles.
    
    ADE20K class IDs are 0-indexed. Key classes:
        0=wall, 1=building/edifice, 2=sky, 4=tree, 5=ceiling, 6=road,
        8=windowpane, 9=grass, 13=earth/ground, 14=door, 17=plant,
        25=house, 32=fence, 38=railing, 42=column, 43=signboard, 87=awning
    """
    import torch
    img_pil = Image.open(image_path).convert("RGB")
    w_img, h_img = img_pil.size
    
    proc, mdl = get_segmentation_model()
    inputs = proc(images=img_pil, return_tensors="pt")
    
    with torch.no_grad():
        logits = mdl(**inputs).logits

    logits_resized = torch.nn.functional.interpolate(
        logits, size=(h_img, w_img), mode="bilinear", align_corners=False
    )
    predicted = logits_resized.argmax(dim=1).squeeze().cpu().numpy()

    # ── Include: structural wall and exterior facade classes ──────────────────
    # 0=wall (catches rendered brick/concrete/plaster surfaces)
    # 25=house (exterior shell of a residential building)
    # 48=skyscraper, 85=tower (for commercial buildings)
    # ── Wall mask: argmax-first + confidence guard ────────────────────────────
    # Strategy: a pixel is "wall" if:
    #  (a) The model's #1 prediction for that pixel IS wall or house (argmax check)
    #  (b) That wall probability is at least 20% (not just a 1-in-150 default)
    #  (c) The pixel isn't confidently a window or door (carve guard)
    # This naturally excludes sky (argmax=2), tree (argmax=4), grass (argmax=9) etc.

    import torch
    probs = torch.softmax(logits_resized, dim=1)[0].cpu().numpy()  # shape: (150, H, W)
    probs_np = probs  # already a numpy array (150, H, W)
    predicted_class = probs_np.argmax(axis=0)           # (H, W) — top class per pixel

    wall_prob  = np.maximum(probs_np[0], probs_np[25])  # wall(0) or house(25)
    carve_prob = np.maximum.reduce([
        probs_np[8],   # windowpane
        probs_np[14],  # door
        probs_np[38],  # railing
    ])

    wall_mask = (
        np.isin(predicted_class, [0, 25]) &  # model's TOP guess is wall or house
        (wall_prob  > 0.20) &                  # at least 20% confident
        (carve_prob < 0.40)                    # not strongly a window/door
    ).astype('uint8')


    # ── Spatial constraint: always drop top 15% (sky / foreground bokeh) ─────
    wall_mask[:int(h_img * 0.15)] = 0

    # ── Morphological cleanup ─────────────────────────────────────────────────
    kernel = np.ones((7, 7), np.uint8)
    wall_mask = cv2.morphologyEx(wall_mask, cv2.MORPH_CLOSE, kernel)
    wall_mask = cv2.morphologyEx(wall_mask, cv2.MORPH_OPEN, kernel)

    # ── Connected Component Analysis ─────────────────────────────────────────
    # The main building facade is ALWAYS the largest connected region.
    # Small disconnected patches = misclassified bokeh, leaves, or noise → discard.
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(wall_mask, connectivity=8)
    if num_labels > 1:
        # Sort all components (excluding background label 0) by pixel area
        areas = stats[1:, cv2.CC_STAT_AREA]
        # Keep only the top 3 largest regions to handle multi-structure scenes
        top_n = min(3, len(areas))
        top_labels = np.argsort(areas)[-top_n:] + 1  # +1 because label 0 = background
        # Threshold: only keep components whose area is > 2% of image
        min_area = h_img * w_img * 0.02
        top_labels = [lbl for lbl in top_labels if stats[lbl, cv2.CC_STAT_AREA] > min_area]
        wall_mask = np.isin(labels, top_labels).astype('uint8')
    else:
        wall_mask = np.zeros_like(wall_mask)  # nothing found → blank canvas for user

    # ── Build multi-color RGBA structural overlay ─────────────────────────────
    # Walls / house → Red   #ef4444  (239, 68, 68)
    # Windows       → Blue  #3b82f6  (59, 130, 246)
    # Doors         → Green #22c55e  (34, 197, 94)
    # Railings      → Amber #f59e0b  (245, 158, 11)
    rgba = np.zeros((h_img, w_img, 4), dtype=np.uint8)

    # Walls first (the main connected-component-filtered mask)
    rgba[wall_mask == 1, 0] = 68   # B
    rgba[wall_mask == 1, 1] = 68   # G
    rgba[wall_mask == 1, 2] = 239  # R
    rgba[wall_mask == 1, 3] = 220  # Alpha (semi-transparent)

    # Overlay windows on top (blue)
    win_mask = (probs[8] > 0.30).astype('uint8')  # windowpane class
    rgba[win_mask == 1, 0] = 246   # B
    rgba[win_mask == 1, 1] = 130   # G
    rgba[win_mask == 1, 2] = 59    # R
    rgba[win_mask == 1, 3] = 200

    # Overlay doors on top (green)
    door_mask = (probs[14] > 0.25).astype('uint8')  # door class
    rgba[door_mask == 1, 0] = 94   # B
    rgba[door_mask == 1, 1] = 197  # G
    rgba[door_mask == 1, 2] = 34   # R
    rgba[door_mask == 1, 3] = 200

    # Overlay railings (amber)
    rail_mask = (probs[38] > 0.20).astype('uint8')  # railing class
    rgba[rail_mask == 1, 0] = 11   # B
    rgba[rail_mask == 1, 1] = 158  # G
    rgba[rail_mask == 1, 2] = 245  # R
    rgba[rail_mask == 1, 3] = 200

    mask_path = (image_path
                 .replace("uploads", "masks")
                 .replace(".jpg", "_mask.png")
                 .replace(".png", "_mask.png")
                 .replace(".jpeg", "_mask.png"))
    os.makedirs(os.path.dirname(mask_path), exist_ok=True)
    cv2.imwrite(mask_path, rgba)
    return mask_path


def apply_material_with_mask(image_path, mask_data_url, material_type, material_color_hex=None, texture_path=None, door_height_pixels=None, door_height_ft=7.0):
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
    
    # Feather boundary to reduce "pasted flat" look
    wall_mask_blurred = cv2.GaussianBlur(wall_mask_bool * 255, (5, 5), 0)
    wall_mask_norm = wall_mask_blurred / 255.0
    wall_mask_3d = np.stack([wall_mask_norm]*3, axis=-1)
    
    # Convert image to LAB color space to separate Lightness (shadows) from Color
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(img_lab)
    
    # ── FIX 1: Luminance-based tinting attenuation ────────────────────────────
    # Very bright pixels (white overhangs, light pillars) should resist colour tinting.
    # We build a "colour strength" multiplier: 0.0 for pure white, 1.0 for mid-tones.
    L_float  = L.astype(np.float32) / 255.0
    # Pixels above 0.88 brightness fade to no-tint (white architectural elements)
    tint_strength = np.clip(1.0 - np.maximum(0.0, (L_float - 0.65) / 0.35), 0.0, 1.0)
    tint_strength_3d = np.stack([tint_strength]*3, axis=-1)
    
    if material_type == 'paint' and material_color_hex:
        h_hex = material_color_hex.lstrip('#')
        rgb = tuple(int(h_hex[i:i+2], 16) for i in (0, 2, 4))
        bgr = (rgb[2], rgb[1], rgb[0])
        
        target_bgr = np.uint8([[[bgr[0], bgr[1], bgr[2]]]])
        target_lab = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2LAB)
        
        # Replace colour channels A and B, keeping original Lightness (shadows)
        new_A = np.full_like(A, target_lab[0, 0, 1]).astype(np.float32)
        new_B = np.full_like(B, target_lab[0, 0, 2]).astype(np.float32)
        
        # Blend: where tint_strength is low (bright areas), keep original A, B channels
        blended_A = (A.astype(np.float32) * (1 - tint_strength) + new_A * tint_strength).astype(np.uint8)
        blended_B = (B.astype(np.float32) * (1 - tint_strength) + new_B * tint_strength).astype(np.uint8)
        
        merged = cv2.merge([L, blended_A, blended_B])
        overlay = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
        
    elif texture_path:
        texture = cv2.imread(texture_path)
        if texture is not None:
            # ── Realistic Tile/Stone Sizing ────────────────────────────────────
            # Calculate pixels per meter using door height as reference.
            DOOR_HEIGHT_METERS = door_height_ft * 0.3048  # convert ft → meters
            if door_height_pixels and door_height_pixels > 0:
                pixels_per_meter = door_height_pixels / DOOR_HEIGHT_METERS
            else:
                # Fallback: estimate door as 25% of image height
                pixels_per_meter = (h * 0.25) / DOOR_HEIGHT_METERS

            # Real-world tile sizes (in meters)
            TILE_SIZES = {
                'tiles':   (0.60, 0.60),  # standard 60×60cm wall tile
                'stone':   (0.45, 0.30),  # natural stone cladding slab
                'texture': (1.00, 1.00),  # decorative plaster repeat
            }
            tw_m, th_m = TILE_SIZES.get(material_type, (0.60, 0.60))
            tile_w_px = max(30, int(tw_m * pixels_per_meter))
            tile_h_px = max(30, int(th_m * pixels_per_meter))

            # Scale the texture image to exactly tile_w_px × tile_h_px
            texture_tile = cv2.resize(texture, (tile_w_px, tile_h_px), interpolation=cv2.INTER_CUBIC)

            # Tile it across the full image canvas using np.tile
            reps_y = int(np.ceil(h / tile_h_px))
            reps_x = int(np.ceil(w / tile_w_px))
            texture_tiled = np.tile(texture_tile, (reps_y, reps_x, 1))
            texture_resized = texture_tiled[:h, :w]  # crop to exact image size

            # Apply with CLAHE-boosted luminance so shadows stay 3D
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            L_enhanced = clahe.apply(L)
            L_norm = L_enhanced.astype(np.float32) / 255.0
            L_norm = np.power(L_norm, 0.75)  # lift midtone shadows
            L_norm_3d = np.stack([L_norm]*3, axis=-1)

            texture_lit = (texture_resized.astype(np.float32) * L_norm_3d)
            overlay = np.clip(texture_lit, 0, 255).astype(np.uint8)
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

    
    return output_path, masked_pixels
