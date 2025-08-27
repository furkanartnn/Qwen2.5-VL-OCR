from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from transformers import BitsAndBytesConfig
import torch
import json
import re
from pdf2image import convert_from_path

device = torch.device("mps")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct", torch_dtype = torch.bfloat16, device_map= device
)
print(model.device)

#model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#    "Qwen/Qwen2.5-VL-7B-Instruct",
#    torch_dtype=torch.bfloat16,
#    attn_implementation="flash_attention_2",
#    device_map="auto",
#)

#processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
min_pixels = 256*28*28
max_pixels = 1280*28*28
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

pages = convert_from_path('cc_pay1.pdf', 200)
images = []
for i, page in enumerate(pages):
    image_path = f"page_{i}.png"
    page.save(image_path, 'PNG')
    images.append(image_path)

all_results = []
seen_keys = set()


for img in images:

    messages = [
        {
            "role": "user",
            "content": [

                {"type": "image", "image": img},
                {"type": "text", "text": "Give me ONLY a valid JSON with the fields: patient_name, date_of_birth, date_of_service. Do not include any markdown, explanation or text outside the JSON."},
            ],
        }
    ]

# Preparation for inference
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
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    try:
        result_json = json.loads(output_text)
    except :
        result_json = {"response": output_text}

    patient_name = result_json.get("patient_name", "")
    date_of_service = result_json.get("date_of_service", "")
    key = (patient_name, date_of_service)
    if key not in seen_keys:
        all_results.append(result_json)
        seen_keys.add(key)

    with open('final_result.json', 'w', encoding="utf-8") as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)

print("saved", all_results)
