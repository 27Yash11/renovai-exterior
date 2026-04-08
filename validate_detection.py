"""
Validation script to test the image processing pipeline.
Run this to see what the AI detects in a sample image.
"""

import cv2
import numpy as np
from PIL import Image
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from utils.vision import get_segmentation_model

def validate_detection(image_path):
    """
    Test the detection pipeline and show statistics.
    """
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return
    
    print(f"📸 Loading image: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        print("❌ Failed to load image")
        return
    
    h, w = img.shape[:2]
    print(f"✅ Image loaded: {w}×{h} pixels")
    
    # Test Segformer detection
    print("\n🤖 Running Segformer semantic segmentation...")
    img_pil = Image.open(image_path).convert("RGB")
    proc, mdl = get_segmentation_model()
    
    import torch
    inputs = proc(images=img_pil, return_tensors="pt")
    with torch.no_grad():
        outputs = mdl(**inputs)
        logits = outputs.logits
    
    logits_resized = torch.nn.functional.interpolate(
        logits, size=(h, w), mode="bilinear", align_corners=False
    )
    predicted = logits_resized.argmax(dim=1).squeeze().cpu().numpy()
    
    # Count detected classes
    unique_classes = np.unique(predicted)
    
    # ADE20K class names (subset relevant to houses)
    class_names = {
        0: "background",
        1: "wall",
        2: "building",
        3: "sky",
        4: "floor",
        5: "ceiling",
        6: "roof",
        8: "windowpane",
        14: "door",
        15: "railing",
        25: "signboard",
        30: "fence",
        36: "curtain",
        53: "screen door",
        62: "light",
        76: "window blind",
        149: "glass"
    }
    
    print("\n📊 Detected Classes:")
    print("-" * 50)
    total_pixels = h * w
    
    for cls in sorted(unique_classes):
        count = np.sum(predicted == cls)
        percentage = (count / total_pixels) * 100
        name = class_names.get(cls, f"class_{cls}")
        
        # Highlight important classes
        if cls in [8, 14, 15, 30, 149, 53, 62, 76]:
            marker = "🚫"  # Will be excluded from wall mask
        elif cls in [1, 2]:
            marker = "✅"  # Wall/building
        else:
            marker = "  "
        
        print(f"{marker} Class {cls:3d} ({name:20s}): {count:7d} pixels ({percentage:5.2f}%)")
    
    # Calculate what will be carved out
    carve_classes = [8, 14, 149, 36, 25, 53, 15, 30, 62, 76]
    carved_pixels = sum(np.sum(predicted == cls) for cls in carve_classes)
    carved_percentage = (carved_pixels / total_pixels) * 100
    
    print("\n" + "=" * 50)
    print(f"🔪 Total pixels to be EXCLUDED: {carved_pixels:,} ({carved_percentage:.2f}%)")
    print(f"🎨 Estimated paintable area: ~{100 - carved_percentage:.2f}% of detected building")
    print("=" * 50)
    
    # Save visualization
    output_path = image_path.replace(".jpg", "_segmentation_viz.jpg").replace(".png", "_segmentation_viz.png")
    
    # Create color-coded visualization
    viz = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Color map
    viz[predicted == 1] = [100, 200, 100]  # wall = green
    viz[predicted == 2] = [100, 200, 100]  # building = green
    viz[predicted == 8] = [0, 0, 255]      # window = red
    viz[predicted == 14] = [255, 0, 0]     # door = blue
    viz[predicted == 15] = [255, 255, 0]   # railing = cyan
    viz[predicted == 149] = [0, 0, 255]    # glass = red
    viz[predicted == 3] = [200, 200, 200]  # sky = gray
    
    # Blend with original
    blended = cv2.addWeighted(img, 0.6, viz, 0.4, 0)
    cv2.imwrite(output_path, blended)
    print(f"\n💾 Visualization saved: {output_path}")
    print("   Green = Wall/Building | Red = Windows/Glass | Blue = Doors | Cyan = Railings")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Default test image
        image_path = "static/uploads/test_house.jpg"
        if not os.path.exists(image_path):
            print("Usage: python validate_detection.py <path_to_house_image>")
            print("\nOr place a test image at: static/uploads/test_house.jpg")
            sys.exit(1)
    
    validate_detection(image_path)
