import os
import re
import zipfile
import csv
from io import BytesIO
from flask import Flask, render_template, request, redirect, flash, url_for
from rex_omni import RexOmniWrapper, RexOmniVisualize
from PIL import Image

UPLOAD_FOLDER = 'static/uploads/'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'EZEc6*AASw&&Zx'

# Initialize the Rex-Omni OCR model
model_path = "IDEA-Research/Rex-Omni"

rex_model = RexOmniWrapper(
    model_path=model_path,
    backend="vllm",  # Choose "transformers" or "vllm"
    max_tokens=2048,
    temperature=0.0,
    top_p=0.05,
    top_k=1,
    repetition_penalty=1.05,
)

def create_crops_archive(image_path, raw_output, zip_path):
    """
    Parses raw_output, crops images, creates CSV, and saves to a zip file.
    """
    try:
        img = Image.open(image_path).convert("RGB")
        width, height = img.size
        
        # Regex to capture Label and Coordinate Block
        # Looks for: <|object_ref_start|> LABEL <|object_ref_end|> <|box_start|> COORDS <|box_end|>
        pattern = r"<\|object_ref_start\|>(.*?)<\|object_ref_end\|>\s*<\|box_start\|>(.*?)<\|box_end\|>"
        matches = re.findall(pattern, raw_output)
        
        crops_data = [] # List of [filename, label]
        
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            crop_count = 0
            
            for label, box_str in matches:
                label = label.strip()
                # Extract all numbers from the coordinate string (ignoring commas and tags)
                coords = [int(c) for c in re.findall(r"<(\d+)>", box_str)]
                
                # Process coordinates in groups of 4 (x1, y1, x2, y2)
                for i in range(0, len(coords), 4):
                    if i + 3 >= len(coords):
                        break
                        
                    n_x1, n_y1, n_x2, n_y2 = coords[i], coords[i+1], coords[i+2], coords[i+3]
                    
                    # Convert normalized (0-1000) to pixel coordinates
                    x1 = int((n_x1 / 1000) * width)
                    y1 = int((n_y1 / 1000) * height)
                    x2 = int((n_x2 / 1000) * width)
                    y2 = int((n_y2 / 1000) * height)
                    
                    # Basic validation to ensure positive area
                    if x2 <= x1 or y2 <= y1:
                        continue
                        
                    # Crop
                    crop = img.crop((x1, y1, x2, y2))
                    
                    # Generate ID/Filename
                    crop_filename = f"crop_{crop_count:04d}.jpg"
                    
                    # Save crop to memory and add to zip
                    img_byte_arr = BytesIO()
                    crop.save(img_byte_arr, format='JPEG')
                    zipf.writestr(crop_filename, img_byte_arr.getvalue())
                    
                    # Add to data list
                    crops_data.append([crop_filename, label])
                    crop_count += 1
            
            # Create CSV content
            csv_output = "image_id,object_name\n"
            for filename, lbl in crops_data:
                # Simple CSV escaping for the label
                clean_label = lbl.replace('"', '""')
                if ',' in clean_label:
                    clean_label = f'"{clean_label}"'
                csv_output += f"{filename},{clean_label}\n"
            
            zipf.writestr("annotations.csv", csv_output)
            
        return len(crops_data) > 0
        
    except Exception as e:
        print(f"Error creating zip: {e}")
        return False

def get_text_from_image(filepath, mode="ocr", custom_categories=None):
    try:
        image = Image.open(filepath).convert("RGB")
        
        # Determine task and categories based on mode
        if mode == "detection":
            task = "detection"
            # Use custom categories if provided, otherwise default fallback
            if custom_categories:
                categories = custom_categories
            else:
                categories = ["person", "animal", "angel", "flower"]
        else:
            # Default to OCR
            task = "ocr_box"
            categories = ["word"]

        results = rex_model.inference(images=image, task=task, categories=categories)
        result = results[0]

        raw_output = results[0]['raw_output']

        if result["success"]:
            predictions = result["extracted_predictions"]
            vis_image_pil = RexOmniVisualize(
                image=image,
                predictions=predictions,
                font_size=20,
                draw_width=5,
                show_labels=True,
            )
            
            return vis_image_pil, raw_output
        else:
            return None, raw_output

    except Exception as e:
        print(f"Eroare in timpul procesarii OCR: {e}")
        return None, f"Eroare la procesarea imaginii: {e}"


@app.route("/", methods=['GET', 'POST'])
def index():
    original_filename = None
    visualized_filename = None
    raw_output = None
    zip_filename = None

    if request.method == 'POST':
        if 'file' not in request.files:
            flash('Niciun fișier selectat')
            return redirect(request.url)
        
        file = request.files['file']
        mode = request.form.get('mode', 'ocr') # Get the selected mode, default to 'ocr'
        
        # Get custom categories from form
        custom_categories_str = request.form.get('custom_categories')
        custom_categories = None
        if custom_categories_str:
            # Parse comma separated string into list
            custom_categories = [x.strip() for x in custom_categories_str.split(',') if x.strip()]

        if file.filename == '':
            flash('Niciun fișier selectat')
            return redirect(request.url)
        
        if file:
            original_filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], original_filename)
            
            file.save(filepath)
            
            flash('Imaginea a fost încărcată cu succes!')

            try:
                # Pass the selected mode and categories to the processing function
                vis_image_pil, raw_output_text = get_text_from_image(filepath, mode, custom_categories)
                raw_output = raw_output_text

                if vis_image_pil:
                    name, ext = os.path.splitext(original_filename)
                    # Add mode to filename to prevent caching issues
                    visualized_filename = f"{name}_{mode}_vis{ext}"
                    visualized_filepath = os.path.join(app.config['UPLOAD_FOLDER'], visualized_filename)
                    
                    vis_image_pil.save(visualized_filepath)
                    flash(f'Imaginea a fost procesata ({mode}) cu succes!')
                    
                    # --- Generate ZIP Archive ---
                    zip_name = f"{name}_{mode}_crops.zip"
                    zip_path = os.path.join(app.config['UPLOAD_FOLDER'], zip_name)
                    
                    if create_crops_archive(filepath, raw_output, zip_path):
                        zip_filename = zip_name
                        flash('Arhivă cu decupaje generată!')
                    else:
                        flash('Nu s-au putut genera decupajele (niciun obiect găsit sau eroare).')
                    # ----------------------------
                    
                else:
                    flash('Imaginea a fost incarcata, dar procesarea a esuat.')

            except Exception as e:
                flash(f'A aparut o eroare in timpul procesarii: {e}')
                raw_output = f"Eroare: {e}"
            
            return render_template('index.html', 
                                   original_filename=original_filename,
                                   visualized_filename=visualized_filename,
                                   raw_output=raw_output,
                                   zip_filename=zip_filename)

        
    return render_template('index.html', 
                           original_filename=None, 
                           visualized_filename=None, 
                           raw_output=None,
                           zip_filename=None)

@app.route("/reset")
def reset():
    return redirect(url_for('index'))

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", debug=True)