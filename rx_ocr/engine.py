"""
PaddleOCR integration engine.
Wraps PaddleOCR text detection, recognition, and angle classification.
Includes topological line-sorting and graceful error handling.
"""

import os
import logging
from typing import List, Tuple, Optional, Union
import numpy as np

# Disable oneDNN MKLDNN to avoid PIR conversion regression on CPU
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
        det_limit_side_len: int = 1920,
        det_db_unclip_ratio: float = 1.8
    ):
        self.lang = lang
        self.use_angle_cls = use_angle_cls
        self.use_gpu = use_gpu
        self.det_limit_side_len = det_limit_side_len
        self.det_db_unclip_ratio = det_db_unclip_ratio
        
        self._ocr = None
        self._is_initialized = False

    def _initialize(self):
        """Lazy load PaddleOCR model on first inference request."""
        if self._is_initialized:
            return

        try:
            from paddleocr import PaddleOCR
            logger.info("Initializing PaddleOCR with lang=%s", self.lang)
            # PaddleOCR v3 / PaddleX initialization
            try:
                self._ocr = PaddleOCR(lang=self.lang)
            except Exception:
                self._ocr = PaddleOCR(
                    use_angle_cls=self.use_angle_cls,
                    lang=self.lang,
                    use_gpu=self.use_gpu
                )
            self._is_initialized = True
        except ImportError as e:
            logger.error("PaddleOCR is not installed or dependencies are missing: %s", e)
            self._ocr = None
            self._is_initialized = False
            raise RuntimeError(
                "PaddleOCR is not installed. Please install via: pip install paddlepaddle paddleocr"
            ) from e

    def is_available(self) -> bool:
        try:
            self._initialize()
            return self._ocr is not None
        except Exception:
            return False

    def detect(self, img_input: Union[str, np.ndarray]) -> List[OCRBoundingBox]:
        """
        Runs text detection and recognition on an image (path or numpy array).
        Returns a topologically sorted list of OCRBoundingBox objects.
        """
        self._initialize()

        boxes: List[OCRBoundingBox] = []

        # PaddleOCR v3 (PaddleX) pipeline support
        if hasattr(self._ocr, "predict"):
            try:
                preds = list(self._ocr.predict(img_input))
                if preds and len(preds) > 0:
                    first = preds[0]
                    # v3 returns rec_texts, rec_scores, rec_polys
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
                logger.warning("Predict failed, falling back to ocr(): %s", e)

        # PaddleOCR v2 fallback
        raw_results = self._ocr.ocr(img_input, cls=self.use_angle_cls)

        if not raw_results or raw_results[0] is None:
            return []

        for line in raw_results[0]:
            box_coords = line[0]        # 4 vertices: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
            text, confidence = line[1]   # Recognized string & confidence (0.0 to 1.0)
            
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
        """
        Sorts bounding boxes into natural reading order (top-to-bottom, left-to-right).
        Groups boxes with similar Y coordinates into the same line.
        """
        if not boxes:
            return []

        def get_center_y(b: OCRBoundingBox) -> float:
            return sum(pt[1] for pt in b.box) / 4.0

        def get_min_x(b: OCRBoundingBox) -> float:
            return min(pt[0] for pt in b.box)

        # Estimate average box height for vertical thresholding
        heights = [abs(b.box[2][1] - b.box[0][1]) for b in boxes]
        avg_height = max(np.median(heights) if heights else 20.0, 10.0)
        y_threshold = avg_height * 0.65

        # Sort primarily by center Y
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

        # Sort items inside each line by X coordinate
        ordered_boxes: List[OCRBoundingBox] = []
        for line in lines:
            line_sorted = sorted(line, key=get_min_x)
            ordered_boxes.extend(line_sorted)

        return ordered_boxes
