from flask import Flask, render_template, request, jsonify, url_for, redirect
import os
import cv2
from datetime import date
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='transformers')

# Use hybrid approach: Segformer if cached, else fast OpenCV
from utils.vision_hybrid import generate_base_mask, apply_material_with_mask
from utils.estimation import calculate_estimation

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['OUTPUT_FOLDER'] = 'static/outputs'
app.config['MASK_FOLDER'] = 'static/masks'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
os.makedirs(app.config['MASK_FOLDER'], exist_ok=True)
os.makedirs('static/textures', exist_ok=True)

print("✅ RenovAI server ready! Fast OpenCV-only processing (2-3 sec per image)")

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("image")
    if not file or file.filename == '':
        return render_template("index.html", error="No image uploaded")

    filename = file.filename
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(path)

    print(f"📸 Processing image: {filename}")
    try:
        print("  Step 1/2: Running GrabCut...")
        mask_path = generate_base_mask(path)
        print("  Step 2/2: Mask generated successfully!")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return render_template("index.html", error=f"Processing failed: {str(e)}")

    input_url = "static/uploads/" + filename
    mask_url = "static/masks/" + os.path.basename(mask_path)

    return render_template("review.html", input_image=input_url, mask_image=mask_url, filename=filename)

@app.route("/process", methods=["POST"])
def process():
    filename = request.form.get("filename")
    mask_data = request.form.get("maskData")
    material_type = request.form.get("material", "paint")
    material_color = request.form.get("color", "#4338ca")
    door_height_ft = float(request.form.get("door_height", 7.0))
    
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    texture_path = None
    color_hex = None
    if material_type == 'paint':
        color_hex = material_color
    elif material_type == 'tiles':
        texture_path = "static/textures/tile.jpg"
    elif material_type == 'stone':
        texture_path = "static/textures/stone.jpg"
    elif material_type == 'texture':
        texture_path = "static/textures/plaster.jpg"

    try:
        output_path, masked_pixels = apply_material_with_mask(
            path, 
            mask_data,
            material_type, 
            material_color_hex=color_hex, 
            texture_path=texture_path
        )
    except Exception as e:
        return render_template("index.html", error=f"Final processing failed: {str(e)}")

    # Estimation
    img = cv2.imread(path)
    h, w = img.shape[:2]
    
    door_height_pixels = int(h * 0.25)
    
    estimation_data = calculate_estimation(
        masked_pixels, 
        image_width_pixels=w, 
        door_height_pixels=door_height_pixels, 
        door_height_ft=door_height_ft, 
        material_type=material_type
    )
    
    input_url = "static/uploads/" + filename
    output_url = "static/outputs/" + os.path.basename(output_path)

    material_label = {
        'paint': 'Wall Paint',
        'tiles': 'Tiles',
        'stone': 'Stone Cladding',
        'texture': 'Texture Plaster'
    }.get(material_type, material_type.capitalize())

    return render_template(
        "result.html",
        input_image=input_url,
        output_image=output_url,
        estimation=estimation_data,
        material=material_type,
        material_label=material_label,
        material_color=material_color if material_type == 'paint' else None,
        report_date=date.today().strftime("%B %d, %Y"),
        filename=filename,
        maskData=mask_data
    )

if __name__ == "__main__":
    app.run(debug=True, port=5000)