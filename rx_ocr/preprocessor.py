"""
Image preprocessing routines optimized for medicine packaging and prescriptions.
Addresses:
- Foil glare and metallic reflections on blister packs.
- Low-contrast dot-matrix expiry date and batch stamps.
- Uneven lighting and paper skew on prescriptions.
"""

import cv2
import numpy as np
from typing import Union, Tuple, Optional
from PIL import Image
import io


class MedicineImagePreprocessor:
    def __init__(self, target_max_dim: int = 1920):
        self.target_max_dim = target_max_dim

    def load_image(self, image_input: Union[str, bytes, np.ndarray, Image.Image]) -> np.ndarray:
        """Loads image into standard BGR numpy array."""
        if isinstance(image_input, np.ndarray):
            if len(image_input.shape) == 2:
                return cv2.cvtColor(image_input, cv2.COLOR_GRAY2BGR)
            return image_input.copy()
        
        if isinstance(image_input, str):
            img = cv2.imread(image_input)
            if img is None:
                raise ValueError(f"Could not load image from path: {image_input}")
            return img

        if isinstance(image_input, bytes):
            nparr = np.frombuffer(image_input, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Could not decode image bytes")
            return img

        if isinstance(image_input, Image.Image):
            rgb = np.array(image_input)
            if len(rgb.shape) == 2:
                return cv2.cvtColor(rgb, cv2.COLOR_GRAY2BGR)
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    def resize_keep_aspect(self, img: np.ndarray) -> np.ndarray:
        """Resizes high-resolution images to target max dimension to optimize OCR speed."""
        h, w = img.shape[:2]
        max_dim = max(h, w)
        if max_dim <= self.target_max_dim:
            return img

        scale = self.target_max_dim / float(max_dim)
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def reduce_foil_glare(self, bgr_img: np.ndarray) -> np.ndarray:
        """
        Suppresses harsh specular highlights and reflections common on
        metallic medicine blister foil using CLAHE on LAB luminance.
        """
        lab = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        cl = clahe.apply(l)

        # Merge equalized L-channel back with color channels
        merged = cv2.merge((cl, a, b))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    def enhance_dot_matrix_stamps(self, bgr_img: np.ndarray) -> np.ndarray:
        """
        Enhances faintly stamped or dot-matrix printed text
        (e.g., EXP, BATCH, MFG dates) on medicine packaging.
        """
        gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
        
        # Bilateral filter removes noise while keeping edges sharp
        filtered = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
        
        # Adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 21, 5
        )
        return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

    def preprocess_for_prescription(self, image_input: Union[str, bytes, np.ndarray]) -> np.ndarray:
        """Pipeline optimized for paper doctor prescriptions."""
        img = self.load_image(image_input)
        img = self.resize_keep_aspect(img)
        
        # Apply mild contrast adjustment
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        enhanced = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)
        return enhanced

    def preprocess_for_strip(self, image_input: Union[str, bytes, np.ndarray]) -> np.ndarray:
        """Pipeline optimized for medicine strips / blister packs."""
        img = self.load_image(image_input)
        img = self.resize_keep_aspect(img)
        glare_reduced = self.reduce_foil_glare(img)
        return glare_reduced
