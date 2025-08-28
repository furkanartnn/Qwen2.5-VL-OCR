from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import json
import os
import re
from pdf2image import convert_from_path
import time

# --- Model ve processor ---
device = torch.device("mps")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct",
    torch_dtype="auto",
    device_map=device
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
pages = convert_from_path('vi_1.pdf', 100)
images = [page for page in pages]

# --- Function ---
def build_messages(image):
    return [
        {
            "role": "system",
            "content": (
                "You are an assistant that extracts structured data "
                "(patient_name, date_of_birth, date_of_service, and document type) from medical text or OCR outputs. "
                "Always respond with only valid JSON in this exact format:\n\n"
                "{\n"
                '    "patient_name": "string",\n'
                '    "date_of_birth": "MM/DD/YYYY",\n'
                '    "date_of_service": "MM/DD/YYYY"\n'
                "}"
            )
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {
                    "type": "text",
                    "text": "Extract patient_name, date_of_birth, date_of_service, and document type such as invoice or note or report or authorization from this image. Only valid JSON, no markdown, no explanation."
                },
            ]
        }
    ]

# --- Model inference ---
def infer_and_save(image, json_file: str = "result_vi_1_1.json"):
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
        exists = any(entry["date_of_service"] == result_json["date_of_service"] for entry in data)
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
