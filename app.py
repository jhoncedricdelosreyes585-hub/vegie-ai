import os
import traceback
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from veg_detector import detect_vegetables

app = Flask(__name__, template_folder="templates", static_folder="static")

UPLOAD_FOLDER = os.path.join("static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5MB max file size

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ----------------------------
# Homepage Route
# ----------------------------
@app.route("/")
def home():
    return render_template("index.html")

# ----------------------------
# Prediction Route (POST only)
# ----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Check if file is in request
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        
        file = request.files["file"]
        
        # Check if filename is empty
        if file.filename == "":
            return jsonify({"error": "Empty filename"}), 400
        
        # Check file extension
        if not allowed_file(file.filename):
            return jsonify({"error": "Invalid file type. Use JPG, PNG, GIF, or BMP"}), 400
        
        # Save file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)
        
        print(f"✅ File saved: {filepath}")
        
        # Run model prediction
        print("🔍 Running prediction...")
        result = detect_vegetables(filepath)
        
        # Fix image path for web display
        result["image_path"] = "/" + filepath.replace("\\", "/")
        
        print(f"✅ Prediction complete: {result['overall_prediction']}")
        
        return jsonify(result), 200
    
    except FileNotFoundError as e:
        error_msg = f"File not found: {str(e)}"
        print(f"❌ ERROR: {error_msg}")
        return jsonify({"error": error_msg}), 500
    
    except Exception as e:
        # Print full error traceback in terminal
        error_msg = str(e)
        print("❌ ERROR OCCURRED:")
        print(traceback.format_exc())
        return jsonify({"error": error_msg}), 500

# ----------------------------
# Health check endpoint
# ----------------------------
@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": True
    })

if __name__ == "__main__":
    print("🚀 Starting Flask server...")
    print(f"📁 Upload folder: {UPLOAD_FOLDER}")
    print(f"🌐 Open browser to: http://localhost:5001")
    app.run(debug=True, host="0.0.0.0", port=5001)
