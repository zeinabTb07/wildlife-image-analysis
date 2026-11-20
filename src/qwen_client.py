
import base64
import json
import re
from pathlib import Path

import requests


class QwenClient:

    def __init__(self, base_url="http://localhost:8080"):
        self.base_url = base_url.rstrip("/")
        self.endpoint = f"{self.base_url}/v1/chat/completions"

    def is_available(self):
        try:
            response = requests.get(
                f"{self.base_url}/health",
                timeout=5
            )
            return response.status_code == 200
        except requests.RequestException:
            return False

    @staticmethod
    def encode_image(image_path):
        image_path = Path(image_path)

        mime_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp"
        }

        mime_type = mime_types.get(image_path.suffix.lower())

        if mime_type is None:
            raise ValueError(
                f"Unsupported image format: {image_path.suffix}"
            )

        image_data = base64.b64encode(
            image_path.read_bytes()
        ).decode("utf-8")

        return f"data:{mime_type};base64,{image_data}"

    def assess_image(self, image_path, expected_species):
        image_url = self.encode_image(image_path)

        prompt = f"""
Analyze this image carefully.

The species recorded in the dataset is: {expected_species}

Identify the animal visible in the image independently.
Do not assume the recorded species is correct.

Important rules:
- A recognizable living or dead animal body counts as an animal.
- Bones, skeletons, skulls, isolated feathers, footprints and tracks do NOT count.
- Toys, drawings, statues and unrelated objects do NOT count.
- If no animal is visible, return animal_present=false.
- If you cannot determine the animal, use "uncertain".
- Give a short description of what is visible.
- Do not invent species names.

Return ONLY valid JSON:

{{
    "animal_present": true,
    "predicted_species": "Persian leopard",
    "description": "A leopard standing on a rocky surface.",
    "confidence": 0.95
}}
"""

        payload = {
            "model": "local-model",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": image_url}
                        }
                    ]
                }
            ],
            "temperature": 0.1,
            "max_tokens": 250
        }

        response = requests.post(
            self.endpoint,
            json=payload,
            timeout=180
        )

        response.raise_for_status()

        content = response.json()["choices"][0]["message"]["content"]

        match = re.search(r"\{.*\}", content, re.DOTALL)

        if not match:
            raise ValueError(f"Invalid JSON response: {content}")

        result = json.loads(match.group())

        if result.get("animal_present") not in (True, False, "uncertain"):
            raise ValueError("Invalid animal_present value")

        return {
            "animal_present": result["animal_present"],
            "predicted_species": result.get("predicted_species"),
            "description": result.get("description"),
            "confidence": float(result["confidence"])
        }
