"""
Unified pipeline for Prescription and Medicine Strip OCR Extraction.
"""

from typing import Union, List, Dict, Any, Optional
import numpy as np
from PIL import Image

from .preprocessor import MedicineImagePreprocessor
from .engine import PaddleOCREngine
from .parser import MedicalTextParser
from .schemas import PrescriptionData, MedicineStripData, OCRBoundingBox


class PrescriptionOCRPipeline:
    def __init__(
        self,
        lang: str = "en",
        use_angle_cls: bool = True,
        use_gpu: bool = False
    ):
        self.preprocessor = MedicineImagePreprocessor()
        self.engine = PaddleOCREngine(
            lang=lang,
            use_angle_cls=use_angle_cls,
            use_gpu=use_gpu
        )
        self.parser = MedicalTextParser()

    def extract_raw(self, image_input: Union[str, bytes, np.ndarray, Image.Image]) -> List[OCRBoundingBox]:
        """Runs preprocessor + PaddleOCR detection returning bounding boxes."""
        img = self.preprocessor.load_image(image_input)
        return self.engine.detect(img)

    def extract_prescription(
        self,
        image_input: Union[str, bytes, np.ndarray, Image.Image]
    ) -> PrescriptionData:
        """
        Full end-to-end pipeline for doctor prescriptions:
        1. Preprocess & enhance contrast
        2. PaddleOCR detection & reading order sort
        3. Parse structured doctor, patient, and medicine items
        """
        img = self.preprocessor.preprocess_for_prescription(image_input)
        boxes = self.engine.detect(img)
        return self.parser.parse_prescription(boxes)

    def extract_medicine_strip(
        self,
        image_input: Union[str, bytes, np.ndarray, Image.Image]
    ) -> MedicineStripData:
        """
        Full end-to-end pipeline for blister packs / medicine packaging:
        1. Glare reduction & contrast enhancement
        2. PaddleOCR detection with angle classification
        3. Parse brand, composition, batch, mfg, expiry, and MRP
        """
        img = self.preprocessor.preprocess_for_strip(image_input)
        boxes = self.engine.detect(img)
        return self.parser.parse_medicine_strip(boxes)
