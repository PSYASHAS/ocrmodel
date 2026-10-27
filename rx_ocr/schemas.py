"""
Data schemas for structured medical prescription and medicine packaging OCR.
Supports both standard Python dataclasses and JSON serialization.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any


@dataclass
class OCRBoundingBox:
    box: List[List[float]]  # 4 corner coordinates [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    text: str
    confidence: float


@dataclass
class MedicinePrescriptionItem:
    name: str
    strength: Optional[str] = None          # e.g., "500 mg", "625 mg"
    dosage_form: Optional[str] = None       # e.g., "Tablet", "Capsule", "Syrup"
    frequency: Optional[str] = None         # e.g., "1-0-1", "Twice daily (BD)", "OD"
    timing: Optional[str] = None            # e.g., "After food", "Before meals"
    duration: Optional[str] = None          # e.g., "5 days", "1 week"
    instructions: Optional[str] = None      # e.g., "Take with plenty of water"
    raw_text: Optional[str] = None


@dataclass
class PrescriptionData:
    doctor_name: Optional[str] = None
    doctor_registration: Optional[str] = None
    clinic_or_hospital: Optional[str] = None
    prescription_date: Optional[str] = None
    patient_name: Optional[str] = None
    patient_age: Optional[str] = None
    patient_gender: Optional[str] = None
    medicines: List[MedicinePrescriptionItem] = field(default_factory=list)
    diagnosis_or_notes: List[str] = field(default_factory=list)
    raw_lines: List[str] = field(default_factory=list)
    average_confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class IngredientComposition:
    ingredient: str
    strength: Optional[str] = None


@dataclass
class MedicineStripData:
    brand_name: Optional[str] = None
    generic_name: Optional[str] = None
    dosage_form: Optional[str] = None
    composition: List[IngredientComposition] = field(default_factory=list)
    batch_number: Optional[str] = None
    mfg_date: Optional[str] = None
    expiry_date: Optional[str] = None
    mrp: Optional[str] = None
    manufacturer: Optional[str] = None
    schedule_warning: Optional[str] = None  # e.g., "Schedule H Prescription Drug"
    storage_conditions: Optional[str] = None
    raw_lines: List[str] = field(default_factory=list)
    average_confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
