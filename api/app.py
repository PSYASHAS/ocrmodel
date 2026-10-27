"""
REST API server for Prescription & Medicine Packaging OCR.
Allows easy integration with Flutter, React Native, iOS, Android, and Web apps.
"""

import os
import io
from flask import Flask, request, jsonify, render_template_string
from werkzeug.utils import secure_filename
from rx_ocr.pipeline import PrescriptionOCRPipeline

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max image size

# Initialize the pipeline
pipeline = PrescriptionOCRPipeline(lang="en", use_angle_cls=True, use_gpu=False)

HTML_DEMO_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Medicine OCR Engine - Live Test</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; max-width: 900px; margin: 40px auto; padding: 0 20px; background: #f8fafc; color: #1e293b; }
        .card { background: white; border-radius: 12px; padding: 24px; box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1); margin-bottom: 24px; }
        h1 { color: #0f172a; margin-top: 0; }
        .btn { background: #2563eb; color: white; border: none; padding: 10px 20px; border-radius: 8px; font-weight: 600; cursor: pointer; }
        .btn:hover { background: #1d4ed8; }
        select, input[type="file"] { padding: 10px; border: 1px solid #cbd5e1; border-radius: 8px; margin-right: 12px; }
        pre { background: #0f172a; color: #f8fafc; padding: 16px; border-radius: 8px; overflow-x: auto; font-size: 14px; }
        .badge { display: inline-block; padding: 4px 10px; border-radius: 9999px; font-size: 12px; font-weight: 600; background: #e0e7ff; color: #3730a3; }
    </style>
</head>
<body>
    <div class="card">
        <h1>💊 Medicine Prescription & Strip OCR Engine <span class="badge">PaddleOCR</span></h1>
        <p>Upload a doctor's prescription or a medicine blister pack image to extract structured medical data.</p>
        
        <form id="uploadForm">
            <select id="mode">
                <option value="prescription">Doctor Prescription</option>
                <option value="strip">Medicine Packaging / Strip</option>
                <option value="raw">Raw Text & Boxes</option>
            </select>
            <input type="file" id="imageInput" accept="image/*" required>
            <button type="submit" class="btn" id="submitBtn">Process Image</button>
        </form>
    </div>

    <div class="card" id="resultCard" style="display: none;">
        <h3>Extracted Structured JSON Result</h3>
        <pre id="jsonResult">Processing...</pre>
    </div>

    <script>
        const form = document.getElementById('uploadForm');
        const submitBtn = document.getElementById('submitBtn');
        const resultCard = document.getElementById('resultCard');
        const jsonResult = document.getElementById('jsonResult');

        form.onsubmit = async (e) => {
            e.preventDefault();
            const file = document.getElementById('imageInput').files[0];
            const mode = document.getElementById('mode').value;
            if (!file) return;

            submitBtn.disabled = true;
            submitBtn.innerText = 'Extracting OCR...';
            resultCard.style.display = 'block';
            jsonResult.innerText = 'Processing with PaddleOCR...';

            const formData = new FormData();
            formData.append('image', file);

            try {
                const endpoint = mode === 'prescription' ? '/api/v1/extract/prescription'
                               : mode === 'strip' ? '/api/v1/extract/strip'
                               : '/api/v1/extract/raw';

                const res = await fetch(endpoint, { method: 'POST', body: formData });
                const data = await res.json();
                jsonResult.innerText = JSON.stringify(data, null, 2);
            } catch (err) {
                jsonResult.innerText = 'Error: ' + err.message;
            } finally {
                submitBtn.disabled = false;
                submitBtn.innerText = 'Process Image';
            }
        };
    </script>
</body>
</html>
"""

@app.route("/", methods=["GET"])
def home():
    """Live web interface for quick manual testing."""
    return render_template_string(HTML_DEMO_PAGE)

@app.route("/api/v1/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "paddle_available": pipeline.engine.is_available()
    })

@app.route("/api/v1/extract/prescription", methods=["POST"])
def extract_prescription():
    if "image" not in request.files:
        return jsonify({"error": "No 'image' file provided in form data"}), 400
    
    file = request.files["image"]
    img_bytes = file.read()
    
    try:
        data = pipeline.extract_prescription(img_bytes)
        return jsonify({
            "success": True,
            "data": data.to_dict()
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/v1/extract/strip", methods=["POST"])
def extract_strip():
    if "image" not in request.files:
        return jsonify({"error": "No 'image' file provided in form data"}), 400
    
    file = request.files["image"]
    img_bytes = file.read()
    
    try:
        data = pipeline.extract_medicine_strip(img_bytes)
        return jsonify({
            "success": True,
            "data": data.to_dict()
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/v1/extract/raw", methods=["POST"])
def extract_raw():
    if "image" not in request.files:
        return jsonify({"error": "No 'image' file provided in form data"}), 400
    
    file = request.files["image"]
    img_bytes = file.read()
    
    try:
        boxes = pipeline.extract_raw(img_bytes)
        return jsonify({
            "success": True,
            "boxes": [
                {"text": b.text, "confidence": b.confidence, "box": b.box}
                for b in boxes
            ]
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
