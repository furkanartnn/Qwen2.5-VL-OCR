from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import json
import os
import re
import cv2
import numpy as np
from pdf2image import convert_from_path
import time
from PIL import Image, ImageEnhance, ImageFilter

# --- Model ve processor ---
device = torch.device("mps")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct",
    torch_dtype=torch.float16,
    device_map="auto"
)
#model = torch.compile(model)

min_pixels = 256*28*28
max_pixels = 1280*28*28
processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct",
    min_pixels=min_pixels,
    max_pixels=max_pixels
)
# ----PDF to IMAGE----
pages = convert_from_path('cc_pay2.pdf', 300)
images = [page for page in pages]

# --- Function ---
def build_messages(image):
    return [
        {
            "role": "system",
        "content": (
            "You are a strict data extraction assistant. "
            "Your only job is to extract structured data from medical documents (scanned images). "
            "Always respond with a single valid JSON object in this exact schema:\n\n"
            "{\n"
            '  "patient_name": "string",\n'
            '  "date_of_service": "MM/DD/YYYY",\n'
            '  "document_type": "invoice" | "physical_medicine_note", \n'
            "}\n\n"
            "Rules:\n"
            "- Do not include markdown, text, or explanations.\n"
            "- If a field cannot be found, use an empty string \"\".\n"
            "- Always use MM/DD/YYYY for dates (e.g., 07/15/2024). Convert if necessary.\n"
            "- patient_name must be a clean string without extra spaces or titles (e.g., no 'Patient:' prefix).\n"
            "- document_type must be exactly one of the two allowed values."
        )
        },
        {
            "role": "user",
        "content": [
            {"type": "image", "image": image},
            {
                "type": "text",
                "text": "Extract patient_name, date_of_service, and document_type."
            },
        ]
        }
    ]

# --- Model inference ---
def infer_and_save(image, json_file: str = "result_new_cc_2_3B_pixels.json"):
    t0 = time.time()

    messages = build_messages(image)

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    result = output_text[0]

    # --- JSON parse ---
    try:
        result_json = json.loads(result)
    except json.JSONDecodeError:
        cleaned = result
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`").replace("json", "", 1).strip()
        try:
            result_json = json.loads(cleaned)
        except json.JSONDecodeError:
            result_json = {"response": cleaned}

    if os.path.exists(json_file):
        with open(json_file, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
    else:
        data = []

    if "response" in result_json:
        print("⚠️ Invalid JSON type")
    else:
        exists = False #any(entry["date_of_service"] == result_json["date_of_service"] for entry in data)
        if not exists:
            data.append(result_json)
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            print("✅ New record added.")
        else:
            print("⚠️ Duplicate date_of_service, record not added.")
    t1 = time.time()
    print(f"Processing time: {t1 - t0} seconds")


for image in images:
    infer_and_save(image)
