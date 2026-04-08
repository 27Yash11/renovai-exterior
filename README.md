# RenovAI — AI-Based Exterior House Renovation & Cost Estimation System

## 🚀 Quick Start (30 seconds)

```bash
cd project
.\env\Scripts\activate          # Windows
python generate_textures.py      # One-time only
python app.py                     # Start server
```

Open `http://127.0.0.1:5000` → Upload house photo → Get instant redesign + cost estimate!

---

## Overview

RenovAI is a web-based pre-construction planning assistant that allows homeowners to:

1. Upload a photo of their house exterior
2. Review and refine AI-detected wall regions
3. Apply design materials (paint, stone cladding, tiles, texture plaster)
4. Visualize a realistic redesigned version of their own house
5. Receive a detailed material quantity and cost estimate report

---

## Setup Instructions

### 1. Activate Virtual Environment
```bash
.\env\Scripts\activate        # Windows PowerShell
source env/bin/activate       # macOS / Linux
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
> **Note:** First install downloads the Segformer AI model (~14 MB) from HuggingFace automatically.

### 3. Generate Texture Assets *(run once)*
```bash
python generate_textures.py
```

### 4. Run the Application
```bash
python app.py
```
Open `http://127.0.0.1:5000` in your browser.

---

## User Workflow

```
Upload Photo → AI Wall Detection → Review & Edit Mask → Select Material → Render → Cost Report
```

| Step | Page | Description |
|------|------|-------------|
| 1 | `index.html` | Upload exterior photo with live preview |
| 2 | `review.html` | AI detects walls (red overlay). User can paint missed areas or erase windows/gates using brush tools |
| 3 | `review.html` | Select material, color, and reference door height |
| 4 | `result.html` | Side-by-side original vs. redesigned view + full cost report |

---

## How the Estimation Works

### Surface Area Calculation
The system uses a **reference scaling** approach:
- The user inputs a known real-world dimension (default: door height = 7 ft)
- The system assumes the door occupies ~25% of the image height in a standard exterior shot
- This gives a `feet_per_pixel` scale factor
- The total masked (wall) pixel count is multiplied by `sq_ft_per_pixel` to get the surface area

### Material Quantity
```
Total Material = Surface Area × (1 + 10% wastage)
```

### Cost Calculation
Predefined flat-rate lookup table:

| Material | Material Rate | Labor Rate | Unit |
|----------|--------------|------------|------|
| Paint | $0.50 / sq ft | $1.00 / sq ft | gallons (350 sq ft/gal) |
| Tiles | $5.00 / sq ft | $4.00 / sq ft | tiles (1×1 ft) |
| Stone Cladding | $12.00 / sq ft | $8.00 / sq ft | sq ft |
| Texture Plaster | $2.00 / sq ft | $2.50 / sq ft | bags (50 sq ft/bag) |

---

## AI & Computer Vision Pipeline

### Stage 1 — GrabCut Foreground Isolation (`cv2.grabCut`)
Separates the building from sky, road, and vegetation using HSV color thresholding as seed hints.

### Stage 2 — Segformer Semantic Segmentation
Uses `nvidia/segformer-b0-finetuned-ade-512-512` (14 MB, ADE20K dataset) to identify and carve out:
- Windows (class 8)
- Doors (class 14)
- Glass surfaces (class 149)

### Stage 3 — Material Application
- **Paint**: Replaces LAB color channels A & B while preserving original Lightness (L) — retains shadows and 3D depth
- **Textures**: Multiplies texture by original luminance map — stone/tile lines follow the building's natural shadow geometry

---

## System Architecture

```
Browser (Glassmorphism UI)
        │
        ▼ multipart form / base64 mask
Flask Backend (app.py)
        │
        ├── utils/vision.py     ← GrabCut + Segformer + material blending
        └── utils/estimation.py ← pixel→sq ft scaling + cost table
        │
        ▼ Jinja2 render
Result Page (result.html) → printable PDF report
```

### Production-Scale Architecture (Future)
- **Frontend**: React/Next.js with interactive per-section material painting
- **Backend**: FastAPI + Celery job queue for async processing
- **Segmentation**: Mask2Former or DeepLabV3+ trained on housing datasets
- **Storage**: AWS S3 for images, PostgreSQL for user projects and live material rates

---

## Limitations

- Area estimation is heuristic (reference-based), not laser-measured — accuracy is ±15–25%
- Works best on clear, well-lit, front-facing exterior photos
- Very dark or heavily shadowed images may reduce texture realism
- Cost rates are predefined flat rates — actual contractor quotes will vary
- Interior renovation is out of scope

---

## Project Structure

```
project/
├── app.py                  # Flask routes
├── generate_textures.py    # One-time texture asset generator
├── requirements.txt
├── static/
│   ├── index.css           # Glassmorphism UI styles
│   ├── textures/           # tile.jpg, stone.jpg, plaster.jpg
│   ├── uploads/            # User-uploaded images
│   ├── masks/              # AI-generated wall masks
│   └── outputs/            # Rendered redesign outputs
├── templates/
│   ├── index.html          # Upload page
│   ├── review.html         # Mask editor + material selector
│   └── result.html         # Report page
└── utils/
    ├── vision.py           # CV pipeline (GrabCut + Segformer + blending)
    └── estimation.py       # Area & cost calculation engine
```
