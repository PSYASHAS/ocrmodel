"""
Rule-based parser for structured extraction from medical prescriptions and strips.
Extracts:
- Doctor and patient metadata
- Drug items with dosage form, strength, frequency, timing, and duration
- Blister pack details: Brand, Composition, Batch, Expiry, MRP, Manufacturer
"""

import re
from typing import List, Optional, Tuple, Dict, Any
from .schemas import (
    OCRBoundingBox,
    PrescriptionData,
    MedicinePrescriptionItem,
    MedicineStripData,
    IngredientComposition
)


class MedicalTextParser:
    def __init__(self):
        # Common dosage forms
        self.form_pattern = re.compile(
            r'\b(TAB(?:LET)?|CAP(?:SULE)?|SYR(?:UP)?|INJ(?:ECTION)?|OINT(?:MENT)?|DROPS?|CREAM|GEL|SUSPENSION|LOTION)\b',
            re.IGNORECASE
        )

        # Strength (e.g., 500 mg, 650mg, 10 ml, 5%, 0.5 mcg)
        self.strength_pattern = re.compile(
            r'\b(\d+(?:\.\d+)?)\s*(MG|MCG|G|GM|ML|IU|%|MILLIGRAMS?)\b',
            re.IGNORECASE
        )

        # Frequencies:
        # 1) Numeric patterns: 1-0-1, 1-1-1, 1-0-0, 0-0-1, 0-1-0, 1-1-0, 1/2-0-1/2
        self.numeric_freq_pattern = re.compile(
            r'\b([01](?:/2)?\s*[-–]\s*[01](?:/2)?\s*[-–]\s*[01](?:/2)?(?:\s*[-–]\s*[01](?:/2)?)?)\b'
        )

        # 2) Latin/English abbreviations: OD, BD, BID, TDS, TID, QID, SOS, HS, PRN, STAT
        self.abbr_freq_pattern = re.compile(
            r'\b(ONCE\s+DAILY|TWICE\s+DAILY|THRICE\s+DAILY|OD|BD|BID|TDS|TID|QID|Q4H|Q6H|Q8H|SOS|HS|PRN|STAT)\b',
            re.IGNORECASE
        )

        # Timing relative to food
        self.timing_pattern = re.compile(
            r'\b(AFTER\s+(?:FOOD|MEALS?)|BEFORE\s+(?:FOOD|MEALS?)|WITH\s+(?:FOOD|MEALS?)|EMPTY\s+STOMACH|AT\s+BEDTIME|PC|AC)\b',
            re.IGNORECASE
        )

        # Duration: e.g. 5 days, 1 week, 10 d, 1 month
        self.duration_pattern = re.compile(
            r'\b(?:X\s*|FOR\s*)?(\d+)\s*(DAYS?|WEEKS?|MONTHS?|D)\b',
            re.IGNORECASE
        )

        # Strip specific patterns
        self.batch_pattern = re.compile(
            r'(?:B\.?\s*NO\.?|BATCH(?:\s*NO\.?)?|LOT(?:\s*NO\.?)?)[:\s.-]*([A-Z0-9\-\/]+)',
            re.IGNORECASE
        )

        self.mfg_pattern = re.compile(
            r'(?:MFG\.?\s*DATE|MFD\.?|MFG\.?|DATE\s*OF\s*MFG\.?)[:\s.-]*([0-9]{1,2}[/-][0-9]{2,4}|[A-Za-z]{3}[/-][0-9]{2,4})',
            re.IGNORECASE
        )

        self.exp_pattern = re.compile(
            r'(?:EXP\.?\s*DATE|EXPIRY(?:\s*DATE)?|EXP\.?|USE\s*BEFORE)[:\s.-]*([0-9]{1,2}[/-][0-9]{2,4}|[A-Za-z]{3}[/-][0-9]{2,4})',
            re.IGNORECASE
        )

        self.mrp_pattern = re.compile(
            r'(?:M\.?R\.?P\.?|PRICE)[:\s.-]*(?:RS\.?|INR|₹)?\s*([0-9]+(?:\.[0-9]{1,2})?)',
            re.IGNORECASE
        )

        self.schedule_pattern = re.compile(
            r'(SCHEDULE\s+[A-Z0-9]+\s+PRESCRIPTION\s+DRUG[^\n\r.]*)',
            re.IGNORECASE
        )

    def parse_prescription(self, boxes: List[OCRBoundingBox]) -> PrescriptionData:
        """Parses OCR bounding boxes into structured PrescriptionData."""
        lines = [b.text for b in boxes]
        confidences = [b.confidence for b in boxes]
        avg_conf = float(sum(confidences) / len(confidences)) if confidences else 0.0

        full_text = "\n".join(lines)
        data = PrescriptionData(raw_lines=lines, average_confidence=round(avg_conf, 3))

        # 1. Doctor & Clinic Extraction
        for line in lines:
            if not data.doctor_name:
                dr_match = re.search(r'\b(Dr\.?\s+[A-Z][a-zA-Z\s.,]+)', line, re.IGNORECASE)
                if dr_match:
                    data.doctor_name = dr_match.group(1).strip().rstrip(',')

            if not data.doctor_registration:
                reg_match = re.search(r'(?:REG(?:N)?\.?\s*NO\.?|REGISTRATION)[:\s]*([A-Z0-9\-\/]+)', line, re.IGNORECASE)
                if reg_match:
                    data.doctor_registration = reg_match.group(1).strip()

            if not data.clinic_or_hospital:
                clinic_match = re.search(r'\b([A-Za-z\s]+(?:HOSPITAL|CLINIC|HEALTHCARE|MEDICAL CENTRE))\b', line, re.IGNORECASE)
                if clinic_match:
                    data.clinic_or_hospital = clinic_match.group(1).strip()

            if not data.prescription_date:
                date_match = re.search(r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4})\b', line)
                if date_match:
                    data.prescription_date = date_match.group(1).strip()

        # 2. Patient Extraction
        for line in lines:
            if not data.patient_name:
                pt_match = re.search(r'(?:PATIENT(?:\s*NAME)?|NAME|PT\.?)[:\s]+([A-Z][a-zA-Z\s]+)', line, re.IGNORECASE)
                if pt_match:
                    candidate = pt_match.group(1).strip()
                    # Filter out short false positives like "Age", "Sex"
                    if len(candidate) > 2 and not candidate.lower().startswith(("age", "dr", "sex", "date")):
                        data.patient_name = candidate

            if not data.patient_age:
                age_match = re.search(r'(?:AGE|YRS|YEARS?)[:\s]*(\d{1,3})\s*(?:YRS?|YEARS?)?', line, re.IGNORECASE)
                if age_match:
                    data.patient_age = age_match.group(1).strip()

            if not data.patient_gender:
                gender_match = re.search(r'\b(MALE|FEMALE|TRANSGENDER|\bM\b|\bF\b)\b', line, re.IGNORECASE)
                if gender_match:
                    val = gender_match.group(1).upper()
                    data.patient_gender = "Male" if val in ["M", "MALE"] else "Female" if val in ["F", "FEMALE"] else val

        # 3. Medicines Extraction
        medicines: List[MedicinePrescriptionItem] = []
        for line in lines:
            item = self._extract_medicine_line(line)
            if item:
                medicines.append(item)

        data.medicines = medicines
        return data

    def _extract_medicine_line(self, line: str) -> Optional[MedicinePrescriptionItem]:
        """Detects if a single text line contains a medicine prescription entry."""
        form_match = self.form_pattern.search(line)
        strength_match = self.strength_pattern.search(line)
        num_freq = self.numeric_freq_pattern.search(line)
        abbr_freq = self.abbr_freq_pattern.search(line)

        # A prescription medicine line typically contains at least dosage form, strength, or frequency
        is_medicine = bool(form_match or strength_match or (num_freq and len(line.split()) >= 2))
        if not is_medicine:
            return None

        form_raw = form_match.group(1).capitalize() if form_match else None
        form_norm_map = {
            "Tab": "Tablet", "Tablet": "Tablet",
            "Cap": "Capsule", "Capsule": "Capsule",
            "Syr": "Syrup", "Syrup": "Syrup",
            "Inj": "Injection", "Injection": "Injection",
            "Oint": "Ointment", "Ointment": "Ointment",
            "Drops": "Drops", "Drop": "Drops"
        }
        dosage_form = form_norm_map.get(form_raw, form_raw)
        strength = strength_match.group(0).upper() if strength_match else None

        frequency = None
        if num_freq:
            frequency = num_freq.group(1)
        elif abbr_freq:
            frequency = abbr_freq.group(1).upper()

        timing_match = self.timing_pattern.search(line)
        timing = timing_match.group(0).capitalize() if timing_match else None

        duration_match = self.duration_pattern.search(line)
        duration = None
        if duration_match:
            d_count = duration_match.group(1)
            d_unit = duration_match.group(2).lower()
            if d_unit == 'd':
                d_unit = 'days'
            duration = f"{d_count} {d_unit}"

        # Clean line to determine medicine name
        name_clean = line
        for pat in [
            self.form_pattern,
            self.strength_pattern,
            self.numeric_freq_pattern,
            self.abbr_freq_pattern,
            self.timing_pattern,
            self.duration_pattern
        ]:
            name_clean = pat.sub('', name_clean)

        # Remove numbering like "1.", "2)", "Tab "
        name_clean = re.sub(r'^\s*\d+[\.\)]\s*', '', name_clean)
        name_clean = re.sub(r'[^\w\s-]', ' ', name_clean).strip()
        name_tokens = name_clean.split()
        med_name = " ".join(name_tokens[:3]) if name_tokens else "Unknown Medicine"

        return MedicinePrescriptionItem(
            name=med_name,
            strength=strength,
            dosage_form=dosage_form,
            frequency=frequency,
            timing=timing,
            duration=duration,
            raw_text=line
        )

    def parse_medicine_strip(self, boxes: List[OCRBoundingBox]) -> MedicineStripData:
        """Parses OCR bounding boxes from a medicine packaging / blister pack."""
        lines = [b.text for b in boxes]
        confidences = [b.confidence for b in boxes]
        avg_conf = float(sum(confidences) / len(confidences)) if confidences else 0.0

        data = MedicineStripData(raw_lines=lines, average_confidence=round(avg_conf, 3))

        full_text = "\n".join(lines)

        # Extract Batch, Mfg, Exp, MRP
        batch_m = self.batch_pattern.search(full_text)
        if batch_m:
            data.batch_number = batch_m.group(1).strip()

        mfg_m = self.mfg_pattern.search(full_text)
        if mfg_m:
            data.mfg_date = mfg_m.group(1).strip()

        exp_m = self.exp_pattern.search(full_text)
        if exp_m:
            data.expiry_date = exp_m.group(1).strip()

        mrp_m = self.mrp_pattern.search(full_text)
        if mrp_m:
            data.mrp = mrp_m.group(1).strip()

        sched_m = self.schedule_pattern.search(full_text)
        if sched_m:
            data.schedule_warning = sched_m.group(1).strip()

        # Extract Manufacturer
        mfr_match = re.search(r'(?:MFD(?:\.?\s*BY)?|MANUFACTURED\s*BY|MARKETED\s*BY)[:\s]+([^\n\r]+)', full_text, re.IGNORECASE)
        if mfr_match:
            data.manufacturer = mfr_match.group(1).strip()

        # Extract Ingredients & Composition
        composition_items: List[IngredientComposition] = []
        for line in lines:
            # Check for lines like "Paracetamol IP 500mg" or "Amoxicillin Trihydrate IP eq to Amoxicillin 500mg"
            strength_m = self.strength_pattern.search(line)
            if strength_m and not any(k in line.lower() for k in ["exp", "batch", "mfg", "mrp", "rs"]):
                strength_val = strength_m.group(0).strip()
                # Clean ingredient name
                ing_name = self.strength_pattern.sub('', line)
                ing_name = re.sub(r'(?:EACH\s+(?:FILM\s+COATED\s+)?TABLET\s+CONTAINS|IP|BP|USP|CONTAINS)[:\s]*', '', ing_name, flags=re.IGNORECASE).strip()
                if ing_name:
                    composition_items.append(IngredientComposition(ingredient=ing_name, strength=strength_val))

        data.composition = composition_items

        # Identify Brand Name: usually the largest/cleanest line near the top that isn't metadata
        for line in lines[:4]:
            if not any(k in line.lower() for k in ["exp", "mfg", "batch", "b.no", "contains", "store", "keep", "schedule"]):
                form_m = self.form_pattern.search(line)
                if form_m:
                    data.dosage_form = form_m.group(1).capitalize()
                data.brand_name = line.strip()
                break

        return data
