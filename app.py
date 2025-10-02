import os
import uuid
from flask import Flask, request, render_template
from veg_detector import detect_vegetables  # keep original name

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        if "file" not in request.files:
            return "No file part"
        file = request.files["file"]
        if file.filename == "":
            return "No selected file"
        if file:
            # Avoid filename collisions
            filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(file_path)

            # Run detection
            result = detect_vegetables(file_path)

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
