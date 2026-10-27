"""
Command-Line Interface (CLI) for Medicine OCR.
Usage:
    python cli.py --image samples/prescription.jpg --type prescription
    python cli.py --image samples/strip.jpg --type strip
"""

import argparse
import json
import sys
from rx_ocr.pipeline import PrescriptionOCRPipeline


def main():
    parser = argparse.ArgumentParser(description="Medicine Prescription & Strip OCR Extractor")
    parser.add_argument("--image", "-i", required=True, help="Path to prescription or medicine strip image")
    parser.add_argument("--type", "-t", choices=["prescription", "strip", "raw"], default="prescription",
                        help="Extraction type: 'prescription', 'strip', or 'raw'")
    parser.add_argument("--save-json", "-s", help="Optional path to save JSON output")

    args = parser.parse_args()

    pipeline = PrescriptionOCRPipeline()

    print(f"[*] Processing image: {args.image} (Mode: {args.type})...")
    
    if args.type == "prescription":
        result = pipeline.extract_prescription(args.image)
        output = result.to_dict()
    elif args.type == "strip":
        result = pipeline.extract_medicine_strip(args.image)
        output = result.to_dict()
    else:
        boxes = pipeline.extract_raw(args.image)
        output = [{"text": b.text, "confidence": b.confidence, "box": b.box} for b in boxes]

    json_str = json.dumps(output, indent=2)
    print("\n[+] Extracted Structured Data:\n")
    print(json_str)

    if args.save_json:
        with open(args.save_json, "w", encoding="utf-8") as f:
            f.write(json_str)
        print(f"\n[+] Saved JSON output to: {args.save_json}")


if __name__ == "__main__":
    main()
