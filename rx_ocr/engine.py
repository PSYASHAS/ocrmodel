"""
High-Performance OCR Engine supporting both RapidOCR (ONNX Runtime - 10x Faster on CPU)
and native PaddleOCR with automatic fallback.
"""

import os
import logging
from typing import List, Tuple, Optional, Union
import numpy as np

# Disable oneDNN MKLDNN to avoid PIR conversion regression on CPU if PaddleOCR is used
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "0"

from .schemas import OCRBoundingBox

logger = logging.getLogger("rx_ocr.engine")


class PaddleOCREngine:
    def __init__(
        self,
        lang: str = "en",
        use_angle_cls: bool = True,
        use_gpu: bool = False,
        prefer_rapidocr: bool = True
    ):
        self.lang = lang
        self.use_angle_cls = use_angle_cls
        self.use_gpu = use_gpu
        self.prefer_rapidocr = prefer_rapidocr
        
        self._engine = None
        self._engine_type = None  # 'rapidocr' or 'paddleocr'
        self._is_initialized = False

    def _initialize(self):
        """Lazy load and pre-warm model on first request or during app startup."""
        if self._is_initialized:
            return

        # 1. Try RapidOCR (ONNX Runtime: ~10x faster on CPU, no oneDNN bugs)
        if self.prefer_rapidocr:
            try:
                from rapidocr_onnxruntime import RapidOCR
                logger.info("Initializing high-speed RapidOCR (ONNX Runtime)...")
                self._engine = RapidOCR()
                self._engine_type = "rapidocr"
                self._is_initialized = True
                return
            except ImportError:
                logger.info("rapidocr-onnxruntime not installed, falling back to PaddleOCR...")

        # 2. Native PaddleOCR (with heavy 3D unwarper disabled for maximum CPU speed)
        try:
            from paddleocr import PaddleOCR
            logger.info("Initializing PaddleOCR with optimized fast flags (no UVDoc)...")
            try:
                self._ocr = PaddleOCR(
                    lang=self.lang,
                    use_doc_unwarping=False,               # Disables heavy 3D mesh unwarping
                    use_doc_orientation_classify=False,    # Fast direct detection
                    use_textline_orientation=True          # Maintains angled line detection
                )
            except Exception:
                self._ocr = PaddleOCR(
                    use_angle_cls=self.use_angle_cls,
                    lang=self.lang,
                    use_gpu=self.use_gpu
                )
            self._engine = self._ocr
            self._engine_type = "paddleocr"
            self._is_initialized = True
        except ImportError as e:
            logger.error("No OCR engine found: %s", e)
            raise RuntimeError(
                "Neither RapidOCR nor PaddleOCR is installed. Please run: pip install rapidocr-onnxruntime paddleocr"
            ) from e

    def is_available(self) -> bool:
        try:
            self._initialize()
            return self._engine is not None
        except Exception:
            return False

    def warmup(self):
        """Pre-warms model in memory so first user request has zero cold-start delay."""
        self._initialize()
        dummy_img = np.zeros((100, 300, 3), dtype=np.uint8)
        self.detect(dummy_img)
        logger.info("OCR Engine pre-warmed and ready.")

    def detect(self, img_input: Union[str, np.ndarray]) -> List[OCRBoundingBox]:
        """
        Runs text detection and recognition on an image (file path or numpy array).
        Returns a topologically sorted list of OCRBoundingBox objects.
        """
        self._initialize()
        boxes: List[OCRBoundingBox] = []

        # Route A: RapidOCR (ONNX Runtime)
        if self._engine_type == "rapidocr":
            try:
                results, _ = self._engine(img_input)
                if results:
                    for item in results:
                        # item format: [box, text, score]
                        box_coords = item[0]
                        text = str(item[1]).strip()
                        confidence = float(item[2])
                        if text:
                            boxes.append(OCRBoundingBox(box=box_coords, text=text, confidence=confidence))
                return self._sort_reading_order(boxes)
            except Exception as e:
                logger.warning("RapidOCR failed: %s", e)

        # Route B: PaddleOCR v3 (PaddleX)
        if hasattr(self._engine, "predict"):
            try:
                preds = list(self._engine.predict(img_input))
                if preds and len(preds) > 0:
                    first = preds[0]
                    if isinstance(first, dict) and "rec_texts" in first:
                        texts = first.get("rec_texts", [])
                        scores = first.get("rec_scores", [])
                        polys = first.get("rec_polys", [])
                        for i in range(len(texts)):
                            txt = str(texts[i]).strip()
                            score = float(scores[i]) if i < len(scores) else 1.0
                            poly = polys[i].tolist() if i < len(polys) and hasattr(polys[i], 'tolist') else []
                            if txt:
                                boxes.append(OCRBoundingBox(box=poly, text=txt, confidence=score))
                        return self._sort_reading_order(boxes)
            except Exception as e:
                logger.warning("PaddleOCR predict failed: %s", e)

        # Route C: PaddleOCR v2 fallback
        if hasattr(self._engine, "ocr"):
            raw_results = self._engine.ocr(img_input, cls=self.use_angle_cls)
            if raw_results and raw_results[0] is not None:
                for line in raw_results[0]:
                    box_coords = line[0]
                    text, confidence = line[1]
                    clean_text = text.strip()
                    if clean_text:
                        boxes.append(OCRBoundingBox(
                            box=box_coords,
                            text=clean_text,
                            confidence=float(confidence)
                        ))

        return self._sort_reading_order(boxes)

    @staticmethod
    def _sort_reading_order(boxes: List[OCRBoundingBox]) -> List[OCRBoundingBox]:
        """Sorts bounding boxes into natural reading order (top-to-bottom, left-to-right)."""
        if not boxes:
            return []

        def get_center_y(b: OCRBoundingBox) -> float:
            return sum(pt[1] for pt in b.box) / 4.0 if len(b.box) >= 4 else 0.0

        def get_min_x(b: OCRBoundingBox) -> float:
            return min(pt[0] for pt in b.box) if len(b.box) >= 4 else 0.0

        heights = [abs(b.box[2][1] - b.box[0][1]) for b in boxes if len(b.box) >= 4]
        avg_height = max(np.median(heights) if heights else 20.0, 10.0)
        y_threshold = avg_height * 0.65

        sorted_by_y = sorted(boxes, key=get_center_y)
        lines: List[List[OCRBoundingBox]] = []
        for box in sorted_by_y:
            placed = False
            cy = get_center_y(box)
            for line in lines:
                line_avg_y = sum(get_center_y(item) for item in line) / len(line)
                if abs(cy - line_avg_y) < y_threshold:
                    line.append(box)
                    placed = True
                    break
            if not placed:
                lines.append([box])

        ordered_boxes: List[OCRBoundingBox] = []
        for line in lines:
            line_sorted = sorted(line, key=get_min_x)
            ordered_boxes.extend(line_sorted)

        return ordered_boxes
