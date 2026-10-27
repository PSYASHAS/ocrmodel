"""
Automated unit tests for rx_ocr parser, preprocessor, and API.
"""

import unittest
import numpy as np
import cv2
from rx_ocr.schemas import OCRBoundingBox, PrescriptionData, MedicineStripData
from rx_ocr.parser import MedicalTextParser
from rx_ocr.preprocessor import MedicineImagePreprocessor
from api.app import app


class TestMedicalParser(unittest.TestCase):
    def setUp(self):
        self.parser = MedicalTextParser()

    def test_parse_prescription_sample(self):
        # Simulated OCR output lines from a typical prescription
        boxes = [
            OCRBoundingBox(box=[[0,0],[100,0],[100,20],[0,20]], text="CITY HEALTHCARE CLINIC", confidence=0.98),
            OCRBoundingBox(box=[[0,25],[100,25],[100,45],[0,45]], text="Dr. Rajesh Kumar MBBS, MD", confidence=0.99),
            OCRBoundingBox(box=[[0,50],[100,50],[100,70],[0,70]], text="Reg. No: MED-987452", confidence=0.96),
            OCRBoundingBox(box=[[0,75],[100,75],[100,95],[0,95]], text="Date: 15/04/2026", confidence=0.97),
            OCRBoundingBox(box=[[0,100],[100,100],[100,120],[0,120]], text="Patient Name: Ramesh Sharma", confidence=0.95),
            OCRBoundingBox(box=[[0,125],[100,125],[100,145],[0,145]], text="Age: 42 Yrs   Sex: Male", confidence=0.94),
            OCRBoundingBox(box=[[0,150],[100,150],[100,170],[0,170]], text="1. Tab Amoxicillin 500mg 1-0-1 after food x 5 days", confidence=0.96),
            OCRBoundingBox(box=[[0,175],[100,175],[100,195],[0,195]], text="2. Tab Paracetamol 650 mg SOS for fever", confidence=0.95),
            OCRBoundingBox(box=[[0,200],[100,200],[100,220],[0,220]], text="3. Syr Ascoril D 10ml TDS before food 3 days", confidence=0.92),
        ]

        result = self.parser.parse_prescription(boxes)

        self.assertEqual(result.doctor_name, "Dr. Rajesh Kumar MBBS, MD")
        self.assertEqual(result.doctor_registration, "MED-987452")
        self.assertEqual(result.clinic_or_hospital, "CITY HEALTHCARE CLINIC")
        self.assertEqual(result.prescription_date, "15/04/2026")
        self.assertEqual(result.patient_name, "Ramesh Sharma")
        self.assertEqual(result.patient_age, "42")
        self.assertEqual(result.patient_gender, "Male")

        # Verify medicines
        self.assertEqual(len(result.medicines), 3)

        # Med 1: Amoxicillin
        med1 = result.medicines[0]
        self.assertIn("Amoxicillin", med1.name)
        self.assertEqual(med1.strength, "500MG")
        self.assertEqual(med1.dosage_form, "Tablet")
        self.assertEqual(med1.frequency, "1-0-1")
        self.assertEqual(med1.timing, "After food")
        self.assertEqual(med1.duration, "5 days")

        # Med 2: Paracetamol
        med2 = result.medicines[1]
        self.assertIn("Paracetamol", med2.name)
        self.assertEqual(med2.strength, "650 MG")
        self.assertEqual(med2.frequency, "SOS")

        # Med 3: Syr Ascoril
        med3 = result.medicines[2]
        self.assertEqual(med3.dosage_form, "Syrup")
        self.assertEqual(med3.strength, "10ML")
        self.assertEqual(med3.frequency, "TDS")

    def test_parse_medicine_strip_sample(self):
        # Simulated OCR output lines from a blister pack
        boxes = [
            OCRBoundingBox(box=[[0,0],[100,0],[100,20],[0,20]], text="AUGMENTIN 625 DUO", confidence=0.97),
            OCRBoundingBox(box=[[0,25],[100,25],[100,45],[0,45]], text="Amoxicillin and Potassium Clavulanate Tablets IP", confidence=0.95),
            OCRBoundingBox(box=[[0,50],[100,50],[100,70],[0,70]], text="Each film coated tablet contains: Amoxicillin Trihydrate IP eq to Amoxicillin 500mg, Clavulanic Acid 125mg", confidence=0.92),
            OCRBoundingBox(box=[[0,75],[100,75],[100,95],[0,95]], text="B.No. GK8942", confidence=0.99),
            OCRBoundingBox(box=[[0,100],[100,100],[100,120],[0,120]], text="MFG. 04/2025", confidence=0.98),
            OCRBoundingBox(box=[[0,125],[100,125],[100,145],[0,145]], text="EXP. 03/2027", confidence=0.98),
            OCRBoundingBox(box=[[0,150],[100,150],[100,170],[0,170]], text="M.R.P. Rs. 204.50", confidence=0.96),
            OCRBoundingBox(box=[[0,175],[100,175],[100,195],[0,195]], text="Mfd by: GlaxoSmithKline Pharmaceuticals Ltd", confidence=0.94),
            OCRBoundingBox(box=[[0,200],[100,200],[100,220],[0,220]], text="SCHEDULE H PRESCRIPTION DRUG - CAUTION", confidence=0.97)
        ]

        result = self.parser.parse_medicine_strip(boxes)

        self.assertEqual(result.brand_name, "AUGMENTIN 625 DUO")
        self.assertEqual(result.batch_number, "GK8942")
        self.assertEqual(result.mfg_date, "04/2025")
        self.assertEqual(result.expiry_date, "03/2027")
        self.assertEqual(result.mrp, "204.50")
        self.assertIn("GlaxoSmithKline", result.manufacturer)
        self.assertIn("SCHEDULE H", result.schedule_warning)


class TestPreprocessor(unittest.TestCase):
    def setUp(self):
        self.preprocessor = MedicineImagePreprocessor(target_max_dim=500)

    def test_image_resizing_and_glare_reduction(self):
        # Create a synthetic test image
        dummy_img = np.ones((1000, 800, 3), dtype=np.uint8) * 128
        processed = self.preprocessor.preprocess_for_strip(dummy_img)
        h, w = processed.shape[:2]
        self.assertLessEqual(max(h, w), 500)


class TestAPIEndpoints(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_health_check(self):
        response = self.client.get("/api/v1/health")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["status"], "healthy")


if __name__ == "__main__":
    unittest.main()
