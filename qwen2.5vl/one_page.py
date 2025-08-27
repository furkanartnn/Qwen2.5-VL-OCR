from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import json
import os
import re

# --- Model ve processor ---
device = torch.device("mps")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map=device
)

min_pixels = 256*28*28
max_pixels = 1280*28*28
processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct",
    min_pixels=min_pixels,
    max_pixels=max_pixels
)

# --- Function ---
def build_messages(image_path: str):
    return [
        {
            "role": "system",
            "content": (
                "You are an assistant that extracts structured data "
                "(patient_name, date_of_birth, date_of_service) from medical text or OCR outputs. "
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
                {"type": "image", "image": image_path},
                {
                    "type": "text",
                    "text": "Extract patient_name, date_of_birth, and date_of_service from this image. Only valid JSON, no markdown, no explanation."
                },
            ]
        }
    ]

# --- Model inference ---
def infer_and_save(image_path: str, json_file: str = "result2.json"):
    messages = build_messages(image_path)

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
        print("⚠️ Model çıktısı geçerli JSON değil, dosyaya eklenmedi.")
    else:
        exists = any(entry["date_of_service"] == result_json["date_of_service"] for entry in data)
        if not exists:
            data.append(result_json)
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            print("✅ Yeni kayıt eklendi.")
        else:
            print("⚠️ Aynı date_of_service bulundu, kayıt eklenmedi.")

infer_and_save("page_3.png")
infer_and_save("page_4.png") 
infer_and_save("page_5.png")
infer_and_save("page_6.png") 
