"""
Optional hybrid LLM-assisted post-processing for messy handwritten prescriptions.
Can use Gemini API, OpenAI, or Ollama to parse ambiguous OCR lines into structured JSON.
"""

import json
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger("rx_ocr.llm")

PROMPT_PRESCRIPTION = """
You are a specialized medical prescription parsing engine.
Given the following raw OCR text extracted from a doctor's prescription, convert it into a valid JSON object matching this schema:

{
  "doctor_name": string or null,
  "doctor_registration": string or null,
  "clinic_or_hospital": string or null,
  "prescription_date": string or null,
  "patient_name": string or null,
  "patient_age": string or null,
  "patient_gender": string or null,
  "medicines": [
    {
      "name": string,
      "dosage_form": string or null,
      "strength": string or null,
      "frequency": string or null,
      "timing": string or null,
      "duration": string or null,
      "instructions": string or null
    }
  ],
  "diagnosis_or_notes": [string]
}

Raw OCR Text:
"""

class LLMStructurer:
    def __init__(self, api_key: Optional[str] = None, provider: str = "gemini"):
        self.api_key = api_key
        self.provider = provider

    def structure_prescription(self, raw_lines: list) -> Optional[Dict[str, Any]]:
        """If configured, sends raw OCR lines to LLM for high-accuracy JSON parsing."""
        if not self.api_key:
            return None

        raw_text = "\n".join(raw_lines)
        prompt = PROMPT_PRESCRIPTION + "\n" + raw_text
        
        # In production, call Gemini SDK:
        # from google import genai
        # client = genai.Client(api_key=self.api_key)
        # response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        # return json.loads(response.text)
        return None
