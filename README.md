# 💊 Medicine Prescription & Strip OCR Engine

An intelligent OCR and structured extraction engine for **doctor prescriptions** and **medicine packaging/strips**, built with **PaddleOCR**, OpenCV, and Flask.

Designed specifically to power mobile apps (Flutter, React Native, iOS, Android) and backend healthcare systems.

---

## 🌟 Key Features

1. **Dual Extraction Pipelines**:
   - **Prescription Mode**: Extracts doctor details, clinic info, date, patient demographics, and parsed drug lists with dosage form, strength, dosing frequency (`1-0-1`, `OD`, `BD`, `TDS`, etc.), food timing, and duration.
   - **Medicine Strip Mode**: Extracts brand name, generic chemical composition with strengths, batch number, manufacturing date, expiration date, MRP, and Schedule warnings (e.g., Schedule H).
2. **Packaging-Specific Preprocessing**:
   - Glare reduction (CLAHE on LAB color space) to combat shiny metallic blister foil reflections.
   - Contrast enhancement for faintly stamped or dot-matrix printed batch/expiry codes.
3. **Multi-Angle Support**: Automatic orientation classification (`use_angle_cls=True`) for rotated photos taken on mobile devices.
4. **REST API + Web UI**: Ready-to-deploy Flask server with file-upload endpoints and a live browser testing UI.
5. **CLI Utility**: Test any image directly from the command line.

---

## 📁 Project Structure

```
rx-ocr-engine/
├── rx_ocr/
│   ├── __init__.py
│   ├── schemas.py           # Dataclasses & structured models (PrescriptionData, MedicineStripData)
│   ├── preprocessor.py      # Glare reduction, CLAHE, resizing, and de-skewing
│   ├── engine.py            # PaddleOCR integration & topological reading-order line sorting
│   ├── parser.py            # Regex & rule-based parser for medical entities & schedules
│   ├── llm_structurer.py    # Optional hybrid LLM parser for messy doctor handwriting
│   └── pipeline.py          # Unified high-level pipeline
├── api/
│   ├── __init__.py
│   └── app.py               # Flask REST API + Browser Demo interface
├── samples/                 # Sample prescription and medicine packaging test images
├── cli.py                   # Command-line interface
├── generate_sample_image.py # Generates synthetic test images
├── test_engine.py           # Automated unit test suite
├── requirements.txt         # Dependencies
├── setup_github.ps1         # 1-click script to push to GitHub
└── README.md
```

---

## 🚀 Getting Started

### 1. Installation

Clone or navigate to the repository directory:
```bash
cd rx-ocr-engine
```

Install the dependencies:
```bash
# Core OCR & Image dependencies
pip install paddlepaddle paddleocr
pip install opencv-python Pillow Flask
```

> **Note for Python 3.13 on Windows**:
> If standard pip wheel for PaddlePaddle is resolving a source build, install from the official PaddlePaddle index:
> ```bash
> pip install paddlepaddle -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
> pip install paddleocr
> ```

---

### 2. Verify with Unit Tests

Run the included test suite to verify the parser and preprocessor:
```bash
python -m unittest test_engine.py
```

---

### 3. CLI Usage

Process a prescription or medicine strip image directly:

```bash
# Process a prescription image
python cli.py --image samples/sample_prescription.png --type prescription

# Process a medicine strip
python cli.py --image samples/sample_strip.png --type strip

# Save results to a JSON file
python cli.py --image samples/sample_strip.png --type strip --save-json result.json
```

---

### 4. Running the REST API & Web UI

Start the local server:
```bash
python api/app.py
```

- **Web Browser UI**: Open [http://localhost:5000](http://localhost:5000) to upload images and inspect the structured output interactively.
- **Health Check**: `GET http://localhost:5000/api/v1/health`

#### API Endpoints for Mobile Apps:

##### `POST /api/v1/extract/prescription`
- **Body**: `multipart/form-data` with key `image`
- **Sample Response**:
```json
{
  "success": true,
  "data": {
    "doctor_name": "Dr. Rajesh Kumar MBBS, MD",
    "doctor_registration": "MED-987452",
    "clinic_or_hospital": "APOLLO CARE CLINIC",
    "prescription_date": "15/04/2026",
    "patient_name": "Ramesh Sharma",
    "patient_age": "42",
    "patient_gender": "Male",
    "medicines": [
      {
        "name": "Amoxicillin",
        "dosage_form": "Tablet",
        "strength": "500MG",
        "frequency": "1-0-1",
        "timing": "After food",
        "duration": "5 days"
      },
      {
        "name": "Paracetamol",
        "dosage_form": "Tablet",
        "strength": "650 MG",
        "frequency": "SOS"
      }
    ]
  }
}
```

##### `POST /api/v1/extract/strip`
- **Body**: `multipart/form-data` with key `image`
- **Sample Response**:
```json
{
  "success": true,
  "data": {
    "brand_name": "AUGMENTIN 625 DUO",
    "dosage_form": "Tablet",
    "composition": [
      {
        "ingredient": "Amoxicillin Trihydrate eq to Amoxicillin",
        "strength": "500mg"
      },
      {
        "ingredient": "Potassium Clavulanate Diluted eq to Clavulanic Acid",
        "strength": "125mg"
      }
    ],
    "batch_number": "GK8942",
    "mfg_date": "04/2025",
    "expiry_date": "03/2027",
    "mrp": "204.50",
    "manufacturer": "GlaxoSmithKline Pharmaceuticals Ltd",
    "schedule_warning": "SCHEDULE H PRESCRIPTION DRUG"
  }
}
```

---

## 🔗 Syncing to Your GitHub Account

1. Create a new repository on your GitHub account (e.g. `medicine-prescription-ocr`).
2. Run the included PowerShell helper script or execute:
```bash
git remote add origin https://github.com/<YOUR_USERNAME>/<YOUR_REPO_NAME>.git
git branch -M main
git push -u origin main
```
