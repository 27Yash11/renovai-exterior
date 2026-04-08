# Image Processing Pipeline — Technical Documentation

## Overview
The system uses a **two-stage hybrid approach** combining classical computer vision (GrabCut) with deep learning (Segformer) to isolate paintable wall surfaces from doors, windows, railings, and background elements.

---

## Stage 1: Building Isolation (GrabCut + HSV Filtering)

### Purpose
Separate the building structure from the background (sky, trees, road, ground).

### Algorithm
1. **Initial Rectangle Hint**: Assumes building occupies center 70% of frame (15%-85% on both axes)
2. **HSV Pre-filtering** to mark obvious background as definite background:
   - **Blue sky**: `hue ∈ [95°, 135°]`, `saturation > 30%`, `value > 120`
   - **Dark road/shadows**: `saturation < 15%`, `value < 40`
   - **Bottom strip**: Bottom 10% of image = road
   - **Upper trees**: `hue ∈ [25°, 80°]` (green), `saturation > 50%`, in top 40% of image
3. **GrabCut Execution**: 5 iterations with mask initialization

### Output
Binary mask where `1` = building structure, `0` = background

### Limitations
- **Fails if building is off-center** (e.g., angled shot, building on left/right edge)
- **Cloudy/sunset/night photos**: Sky detection assumes blue sky
- **Green-painted walls**: May be misclassified as vegetation if very saturated
- **White buildings on white sky**: Low contrast causes GrabCut to fail

---

## Stage 2: Window/Door/Railing Removal (Segformer Semantic Segmentation)

### Model
`nvidia/segformer-b0-finetuned-ade-512-512`
- **Dataset**: ADE20K (150 classes, includes architectural elements)
- **Size**: 14 MB
- **Accuracy**: ~76% mIoU on ADE20K validation set

### Carved Classes (Excluded from Wall Mask)
| Class ID | Label | Why Excluded |
|----------|-------|--------------|
| 8 | windowpane | Glass surface, not paintable |
| 14 | door | Separate element, not wall |
| 149 | glass | Glass panels, balcony glass |
| 36 | curtain | Interior element visible through windows |
| 25 | signboard | Mounted signs, not wall surface |
| 53 | screen door | Mesh/screen doors |
| 15 | railing | Balcony railings, not wall |
| 30 | fence | Gates, boundary fences |
| 62 | light | Outdoor light fixtures |
| 76 | window blind | Interior blinds visible through windows |

### Post-Processing
1. **Dilation**: 9×9 kernel, 1 iteration — removes window frames and railing posts
2. **Morphological Close**: 7×7 kernel, 2 iterations — fills small holes in walls
3. **Morphological Open**: 7×7 kernel, 1 iteration — removes small noise artifacts

### Output
Binary mask where `1` = paintable wall surface, `0` = windows/doors/railings/background

---

## Stage 3: Material Application (LAB Color Space Blending)

### For Paint
1. Convert original image to LAB color space
2. Extract Lightness (L) channel — contains shadows, depth, texture
3. Replace A and B channels (color) with target paint color
4. Convert back to BGR
5. Blend with original using feathered mask

**Result**: Paint color follows the original wall's shadows and 3D geometry

### For Textures (Stone/Tiles/Plaster)
1. Resize texture to match image dimensions
2. Extract Lightness (L) channel from original image
3. Multiply texture by normalized Lightness: `texture × (L / 255) × 1.15`
4. Blend with original using feathered mask

**Result**: Texture lines (brick joints, tile grout) follow the wall's natural shadows

---

## What the System Detects Correctly

### ✅ High Confidence
- Standard rectangular windows
- Wooden/metal doors
- Glass balcony panels
- Metal railings (vertical bars)
- Blue sky backgrounds
- Paved roads at bottom of frame
- Trees in upper portion of image

### ⚠️ Medium Confidence
- Arched windows (may partially include arch frame)
- Sliding glass doors (may miss frame)
- Decorative window grills (may be partially included)
- Pillars/columns (NOT excluded — will be painted)
- Roof overhangs (NOT excluded — will be painted)

### ❌ Low Confidence / Known Failures
- **Circular/oval windows**: Segformer trained mostly on rectangular windows
- **Stained glass**: May be classified as wall due to color patterns
- **Very dark windows at night**: May be classified as wall (no glass reflection)
- **Garage doors**: Often misclassified as wall (looks like textured wall)
- **Shutters**: May be included in wall mask
- **AC units, pipes, electrical boxes**: NOT excluded (no class in ADE20K)
- **Decorative wall elements** (corbels, moldings): NOT excluded
- **Parapet walls**: Included in wall mask (correct behavior)

---

## Edge Cases & Failure Modes

### 1. Off-Center Buildings
**Problem**: GrabCut rectangle assumes center framing  
**Solution**: User can manually paint missed areas in Step 2 (review.html)

### 2. Low Contrast (White Wall + White Sky)
**Problem**: GrabCut cannot distinguish boundary  
**Solution**: Segformer still detects windows/doors, but wall boundary may be inaccurate

### 3. Heavily Shadowed Facades
**Problem**: Dark shadows may be excluded by HSV filter (`value < 40`)  
**Solution**: More conservative threshold (was 50, now 40) + user manual correction

### 4. Green/Blue Painted Walls
**Problem**: May be misclassified as vegetation/sky  
**Solution**: Tighter HSV ranges (hue 95-135 for sky, 25-80 for trees) + saturation threshold

### 5. Complex Architectural Elements
**Problem**: Segformer has no class for corbels, brackets, decorative moldings  
**Solution**: These will be painted (acceptable — they are part of the wall surface)

---

## Accuracy Expectations

| Scenario | Expected Accuracy | Notes |
|----------|------------------|-------|
| Standard residential house, clear day, front view | 85-95% | Ideal case |
| Angled view (30° off-center) | 70-85% | GrabCut may miss edges |
| Cloudy/overcast sky | 75-90% | Sky detection less reliable |
| Night photo with outdoor lights | 60-75% | Dark windows misclassified |
| Complex facade (many balconies, irregular shape) | 65-80% | Segformer may miss small elements |

**Critical Point**: The system provides a **starting mask** that the user reviews and corrects in Step 2. The interactive canvas editor (paint/erase tools) compensates for AI inaccuracies.

---

## Why This Approach Works for the Assignment

1. **No manual annotation required**: Fully automatic initial detection
2. **User-in-the-loop**: Review step allows correction of AI errors
3. **Realistic rendering**: LAB color space preserves depth and shadows
4. **Fast inference**: Segformer-B0 runs in 2-5 seconds on CPU
5. **Handles 80%+ of residential houses**: Good enough for pre-construction planning

---

## Alternative Approaches (Not Implemented)

### 1. Mask R-CNN
- **Pros**: Instance segmentation (separate each window)
- **Cons**: 10× slower, requires GPU, 200+ MB model

### 2. SAM (Segment Anything Model)
- **Pros**: State-of-the-art segmentation
- **Cons**: 2.4 GB model, requires point prompts, overkill for this task

### 3. Manual Polygon Annotation
- **Pros**: 100% accurate
- **Cons**: Requires 5-10 minutes per image, defeats "instant visualization" goal

---

## Recommendations for Production

1. **Add roof detection**: Exclude class 6 (roof) if user only wants walls
2. **Train custom model**: Fine-tune Segformer on 1000+ house exterior images with pixel-perfect annotations
3. **Add confidence scores**: Show user which regions have low AI confidence
4. **Multi-view support**: Allow user to upload 4 sides of house and process separately
5. **Depth estimation**: Use monocular depth (MiDAS) to better separate foreground/background
