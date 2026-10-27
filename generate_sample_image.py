"""
Generates synthetic sample images for testing the Medicine OCR Engine:
1. samples/sample_prescription.png
2. samples/sample_strip.png
"""

import os
from PIL import Image, ImageDraw, ImageFont


def create_sample_prescription(output_path="samples/sample_prescription.png"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # White prescription notepad background
    img = Image.new("RGB", (800, 1000), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Header
    draw.rectangle([(20, 20), (780, 140)], fill=(240, 245, 255), outline=(100, 150, 220), width=2)
    draw.text((40, 35), "APOLLO CARE CLINIC", fill=(20, 40, 100))
    draw.text((40, 60), "Dr. Rajesh Kumar MBBS, MD (Internal Medicine)", fill=(30, 30, 30))
    draw.text((40, 85), "Reg. No: MED-987452 | Ph: +91 98765 43210", fill=(70, 70, 70))
    draw.text((580, 85), "Date: 15/04/2026", fill=(70, 70, 70))

    # Patient Details Bar
    draw.line([(20, 160), (780, 160)], fill=(200, 200, 200), width=1)
    draw.text((40, 175), "Patient Name: Ramesh Sharma", fill=(20, 20, 20))
    draw.text((450, 175), "Age: 42 Yrs", fill=(20, 20, 20))
    draw.text((600, 175), "Sex: Male", fill=(20, 20, 20))
    draw.line([(20, 210), (780, 210)], fill=(200, 200, 200), width=1)

    # Rx Symbol
    draw.text((40, 240), "Rx", fill=(10, 50, 150))

    # Medicine Lines
    med_lines = [
        "1. Tab Amoxicillin 500mg 1-0-1 after food x 5 days",
        "2. Tab Paracetamol 650 mg SOS for fever",
        "3. Tab Pantocid 40mg 1-0-0 before food x 5 days",
        "4. Syr Ascoril D 10ml TDS after food x 3 days"
    ]

    y = 300
    for line in med_lines:
        draw.text((60, y), line, fill=(10, 10, 10))
        y += 60

    # Instructions & Footer
    draw.text((60, 600), "Advice: Drink warm water, avoid cold drinks.", fill=(60, 60, 60))
    draw.text((550, 850), "Dr. Rajesh Kumar", fill=(30, 30, 30))
    draw.text((550, 875), "(Authorized Signatory)", fill=(90, 90, 90))

    img.save(output_path)
    print(f"[+] Created sample prescription at: {output_path}")


def create_sample_strip(output_path="samples/sample_strip.png"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Metallic blister strip background (silver-grey)
    img = Image.new("RGB", (900, 450), color=(225, 228, 232))
    draw = ImageDraw.Draw(img)

    # Red Schedule warning banner on side
    draw.rectangle([(20, 20), (320, 70)], fill=(180, 20, 20))
    draw.text((30, 30), "SCHEDULE H PRESCRIPTION DRUG", fill=(255, 255, 255))
    draw.text((30, 48), "CAUTION: Not to be sold without prescription", fill=(255, 255, 255))

    # Brand name in bold large lettering
    draw.text((40, 100), "AUGMENTIN 625 DUO", fill=(10, 20, 80))
    draw.text((40, 135), "Amoxicillin and Potassium Clavulanate Tablets IP", fill=(40, 40, 40))

    # Composition
    draw.text((40, 180), "Each film coated tablet contains:", fill=(50, 50, 50))
    draw.text((40, 205), "Amoxicillin Trihydrate IP eq to Amoxicillin 500mg", fill=(20, 20, 20))
    draw.text((40, 230), "Potassium Clavulanate Diluted IP eq to Clavulanic Acid 125mg", fill=(20, 20, 20))

    # Batch, Mfg, Exp, MRP (dot matrix / stamped box)
    draw.rectangle([(550, 100), (860, 320)], fill=(245, 245, 245), outline=(180, 180, 180), width=1)
    draw.text((570, 120), "B.No. GK8942", fill=(10, 10, 10))
    draw.text((570, 155), "MFG. 04/2025", fill=(10, 10, 10))
    draw.text((570, 190), "EXP. 03/2027", fill=(10, 10, 10))
    draw.text((570, 225), "M.R.P. Rs. 204.50", fill=(10, 10, 10))
    draw.text((570, 260), "INCL. OF ALL TAXES", fill=(80, 80, 80))

    # Manufacturer
    draw.text((40, 340), "Mfd by: GlaxoSmithKline Pharmaceuticals Ltd", fill=(30, 30, 30))
    draw.text((40, 365), "Plot No. 42, Industrial Area, Solan (H.P.)", fill=(70, 70, 70))

    img.save(output_path)
    print(f"[+] Created sample medicine strip at: {output_path}")


if __name__ == "__main__":
    create_sample_prescription()
    create_sample_strip()
