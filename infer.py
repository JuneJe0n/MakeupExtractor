"""
Full inference pipeline
Input : model ckpt, GT makeup img folder
Output : generated options JSON, generated makeup applied img folder

Base model : "Qwen/Qwen2.5-VL-7B-Instruct"
"""
import os
import json
import re
import glob
import torch
from datetime import datetime
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import PeftModel
from utils import SYSTEM_PROMPT, QUESTION
from utils import pil_to_rgba_array, build_makeup_options
from utils import FormatReward_, AccuracyReward_, weighted_
from lviton import LViton



# --- Load data & model ---

def load_model_and_processor(checkpoint_path: str, base_model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct"):
    """ Load the fine-tuned model and processor """
    print(f"Loading base model: {base_model_id}")
    processor = AutoProcessor.from_pretrained(base_model_id, use_fast=True, padding_side="left")
    
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    print(f"Loading LoRA weights from: {checkpoint_path}")
    model = PeftModel.from_pretrained(base_model, checkpoint_path)
    model.eval()
    
    return model, processor


def extract_ids_from_filename(filename: str) -> tuple:
    """Extract makeup ID and FFHQ ID from filename pattern {makeupid}_{FFHQ_id}.png"""
    basename = os.path.basename(filename)
    name_without_ext = os.path.splitext(basename)[0]
    
    # Get makeup_id and FFHQ_id
    parts = name_without_ext.split('_')
    if len(parts) >= 2:
        makeup_id = parts[0]  # First part is makeup_id (e.g., "A13")
        ffhq_id = parts[-1]   # Last part is FFHQ_id (e.g., "023932")
        return makeup_id, ffhq_id
    else:
        raise ValueError(f"Invalid filename format: {filename}. Expected format: {{makeupid}}_{{FFHQ_id}}.png")


def extract_ffhq_id_from_filename(filename: str) -> str:
    """Extract FFHQ ID from filename pattern {makeupid}_{FFHQ_id}.png"""
    _, ffhq_id = extract_ids_from_filename(filename)
    return ffhq_id


def load_gt_data(gt_json_path: str) -> dict:
    """Load ground truth data from JSON file and create lookup by makeup ID"""
    with open(gt_json_path, 'r', encoding='utf-8') as f:
        gt_data = json.load(f)
    
    gt_lookup = {}
    for item in gt_data:
        makeup_id = item["id"]
        options = []
        for product in item["products"]:
            for option in product["options"]:
                # Keep only shape and color for reward calculation
                options.append({
                    "shape": option["shape"],
                    "color": option["color"]
                })
        gt_lookup[makeup_id] = options
    
    return gt_lookup


def get_bare_face_path(ffhq_id: str, ffhq_folder: str = "/home/jiyoon/data/FFHQ") -> str:
    """Get the path to the bare face image based on FFHQ_id"""

    extensions = ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']
    
    for ext in extensions:
        potential_path = os.path.join(ffhq_folder, f"{ffhq_id}{ext}")
        if os.path.exists(potential_path):
            return potential_path
    
    raise FileNotFoundError(f"Bare face image for FFHQ_id '{ffhq_id}' not found in {ffhq_folder}")



# --- Generate response

def generate_response(model, processor, image_path: str, max_new_tokens: int = 512):
    """Generate response for given image and prompt"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    image = Image.open(image_path).convert("RGB")

    messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": QUESTION},
                    ],
                },
            ]
    
    text_input = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(
        text=text_input,
        images=image,
        return_tensors="pt",
        padding=True
    )
    
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            pad_token_id=processor.tokenizer.eos_token_id
        )
    
    generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]
    response = processor.decode(generated_ids, skip_special_tokens=True)
    
    return response




# --- Clean response ---

def hex_to_rgb(hex_color: str) -> dict:
    """Convert hex color to RGB dict"""
    hex_color = hex_color.lstrip('#')
    return {
        "r": int(hex_color[0:2], 16),
        "g": int(hex_color[2:4], 16),
        "b": int(hex_color[4:6], 16)
    }

def format_makeup_options(response: str) -> list:
    """Convert model response to formatted makeup options"""
    try:
        cleaned = response.replace('\\n', '\n').replace('\\t', '\t').replace('\\"', '"').replace("\\'", "'")
        answer_match = re.search(r'<answer>(.*?)</answer>', cleaned, re.DOTALL)
        if answer_match:
            json_str = answer_match.group(1).strip()
        else:
            json_str = response.strip()
            
        data = json.loads(json_str)
        formatted_options = []
        
        for item in data:
            if isinstance(item, dict) and "shape" in item and "color" in item:
                shape = item["shape"]
                hex_color = item["color"]
                rgb_color = hex_to_rgb(hex_color)

                shape_params = {
                    "LIP_FULL_BASIC": {"alpha": 190, "sigma": 70, "gamma": 0},
                    "BLUSHER_CENTER_WIDE_BASIC": {"alpha": 80, "sigma": 200, "gamma": 0},
                    "EYESHADOW_OVEREYE_FULL_BASIC": {"alpha": 180, "sigma": 100, "gamma": 50},
                }

                params = shape_params.get(shape)
                if params is None:
                    print(f"Warning: Unknown shape '{shape}', skipping")
                    continue
                    
                alpha = params["alpha"]
                sigma = params["sigma"]
                gamma = params["gamma"]

                
                formatted_option = {
                    "shape": shape,
                    "color": rgb_color,
                    "alpha": alpha,
                    "sigma": sigma,
                    "gamma": gamma,
                    "split": 0
                }
                formatted_options.append(formatted_option)
        
        return formatted_options
        
    except (json.JSONDecodeError, KeyError, TypeError):
        return []


def calculate_shape_based_scores(predicted_response: str, gt_solution: list) -> dict:
    """Calculate color similarity scores grouped by shape"""
    from utils.reward_f1 import _extract_json_block, _safe_load_json, _color_score
    
    shape_based_results = {}
    if not gt_solution:
        return shape_based_results
    
    try:
        # Extract predicted colors from response
        arr = _extract_json_block(predicted_response)
        if not arr:
            return shape_based_results
            
        pred_list = _safe_load_json(arr)
        if not pred_list:
            return shape_based_results
        
        # Match GT and predicted items by shape
        for gt_item in gt_solution:
            gt_shape = gt_item.get("shape", "")
            gt_color = gt_item.get("color", "")
            
            # Find matching predicted item by shape
            matched_pred = None
            for pred_item in pred_list:
                if pred_item.get("shape", "") == gt_shape:
                    matched_pred = pred_item
                    break
            
            if matched_pred and gt_color and matched_pred.get("color"):
                pred_color = matched_pred["color"]
                c_score = _color_score(gt_color, pred_color)
                
                shape_based_results[gt_shape] = {
                    "gt_hex_code": gt_color,
                    "predicted_hex_code": pred_color,
                    "color_similarity": round(c_score, 3)
                }
            elif gt_color:
                # GT exists but no matching prediction
                shape_based_results[gt_shape] = {
                    "gt_hex_code": gt_color,
                    "predicted_hex_code": None,
                    "color_similarity": 0.0
                }
            
        for pred_item in pred_list:
            pred_shape = pred_item.get("shape", "")
            pred_color = pred_item.get("color", "")
            
            if pred_shape not in shape_based_results and pred_color:
                shape_based_results[pred_shape] = {
                    "gt_hex_code": None,
                    "predicted_hex_code": pred_color,
                    "color_similarity": 0.0
                }
                
    except Exception as e:
        print(f"Error calculating shape-based scores: {e}")
    
    return shape_based_results

def extract_hex_codes_from_response(response: str) -> list:
    """Extract hex color codes from model response"""
    hex_codes = []
    try:
        cleaned = response.replace('\\n', '\n').replace('\\t', '\t').replace('\\"', '"').replace("\\'", "'")
        answer_match = re.search(r'<answer>(.*?)</answer>', cleaned, re.DOTALL)
        if answer_match:
            cleaned = answer_match.group(1).strip()
        data = json.loads(cleaned)
        for item in data:
            if isinstance(item, dict) and "color" in item:
                hex_codes.append(item["color"])
                
    except (json.JSONDecodeError, KeyError, TypeError):
        pass
    
    return hex_codes



# --- Save to json ---

def clean_response(response: str) -> str:
    cleaned = response.replace('\\n', '\n').replace('\\t', '\t').replace('\\"', '"').replace("\\'", "'")

    answer_match = re.search(r'<answer>(.*?)</answer>', cleaned, re.DOTALL)
    if answer_match:
        cleaned = answer_match.group(1).strip()
    
    if not cleaned.startswith('<answer>') and not cleaned.endswith('</answer>'):
        cleaned = f"<answer>{cleaned}</answer>"
    
    return cleaned

def save_result_to_json(response: str, image_path: str, output_dir: str, base_model_id: str, checkpoint_path: str, gt_solution: dict = None):
    os.makedirs(output_dir, exist_ok=True)
    
    cleaned_response = clean_response(response)
    formatted_options = format_makeup_options(cleaned_response)
    predicted_hex_codes = extract_hex_codes_from_response(response)

    shape_based_scores = {}
    if gt_solution and isinstance(gt_solution, list):
        shape_based_scores = calculate_shape_based_scores(cleaned_response, gt_solution)
    
    gt_hex_code = None
    if gt_solution and isinstance(gt_solution, list) and len(gt_solution) > 0
        gt_hex_code = [item["color"] for item in gt_solution if "color" in item]
        if len(gt_hex_code) == 1:
            gt_hex_code = gt_hex_code[0] 
    elif gt_solution:
        print(f"🚨 Debug: gt_solution type: {type(gt_solution)}, value: {gt_solution}")
    
    # Calculate rewards 
    fmt_reward = FormatReward_(w_tags=0.3, w_json=0.3, w_schema=0.4)
    acc_reward = AccuracyReward_(reference_key="solution", tau=0.6) 
    L_F, L_A = 0.3, 1.0
    
    format_reward_score = fmt_reward([cleaned_response])[0] if cleaned_response else 0.0
    accuracy_reward_score = 0.0
    if gt_solution and isinstance(gt_solution, list):
        accuracy_reward_score = acc_reward([cleaned_response], solution=[gt_solution])[0]
    

    weighted_format_reward = weighted_(fmt_reward, L_F)([cleaned_response])[0] if cleaned_response else 0.0
    weighted_accuracy_reward = 0.0
    if gt_solution and isinstance(gt_solution, list):
        weighted_accuracy_reward = weighted_(acc_reward, L_A)([cleaned_response], solution=[gt_solution])[0]
    
    total_weighted_reward = weighted_format_reward + weighted_accuracy_reward
    
    result_data = {
        "formatted_options": formatted_options,
        "raw_response": cleaned_response,
        "shape_based_results": shape_based_scores,
        "rewards": {
            "format_reward": format_reward_score,
            "accuracy_reward": accuracy_reward_score,
            "weighted_format_reward": weighted_format_reward,
            "weighted_accuracy_reward": weighted_accuracy_reward,
            "total_weighted_reward": total_weighted_reward
        },
        "metadata": {
            "base_model": base_model_id,
            "checkpoint_path": checkpoint_path,
            "timestamp": datetime.now().isoformat(),
            "image_path": image_path
        }
    }
    
    image_name = os.path.basename(image_path).split('.')[0]
    output_filename = f"{image_name}.json"
    output_path = os.path.join(output_dir, output_filename)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)
    
    return output_path



# --- Apply makeup ---

def apply_makeup_to_bare_face(json_path: str, bare_face_path: str, gt_image_path: str, ffhq_id: str, output_dir: str):
    with open(json_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    
    ffhq_output_dir = os.path.join(output_dir, ffhq_id)
    os.makedirs(ffhq_output_dir, exist_ok=True)
    
    if not os.path.exists(bare_face_path):
        raise FileNotFoundError(f"Bare face image not found: {bare_face_path}")
    
    bare_face = Image.open(bare_face_path).convert("RGB")
    img_rgba = pil_to_rgba_array(bare_face)
    
    bare_output_path = os.path.join(ffhq_output_dir, "bare.png")
    bare_face.save(bare_output_path)
    print(f"Bare face saved to: {bare_output_path}")
    
    gt_image = Image.open(gt_image_path).convert("RGB")
    gt_output_path = os.path.join(ffhq_output_dir, "gt.png")
    gt_image.save(gt_output_path)
    print(f"GT image saved to: {gt_output_path}")
    
    lib_path = "/home/jiyoon/LViton_GRPO/LViton/lib/liblviton-x86_64-linux-3.0.3.so"
    face_landmarker_path = "/home/jiyoon/LViton_GRPO/LViton/model/face_landmarker.task"
    lviton = LViton(lib_path=lib_path, face_landmarker_path=face_landmarker_path)
    
    if not lviton.set_image(img_rgba):
        raise RuntimeError("No face detected in the bare face image")

    if isinstance(json_data, dict) and "formatted_options" in json_data:
        formatted_options = json_data["formatted_options"]
    else:
        formatted_options = json_data
    
    products = [{"options": formatted_options}]
    makeup_options = build_makeup_options(products)
    if not makeup_options:
        raise ValueError("No valid makeup options found in JSON")
    
    # Apply makeup
    result_rgb = lviton.apply_makeup(makeup_options)
    
    applied_output_path = os.path.join(ffhq_output_dir, "applied.png")
    lviton.save_png(result_rgb, applied_output_path)
    print(f"Applied makeup saved to: {applied_output_path}")
    
    return applied_output_path


def main():
    BASE_MODEL =  "Qwen/Qwen2.5-VL-7B-Instruct"
    CKPT_PATH = "/home/jiyoon/data/ckpts/shape3/Qwen2.5-VL-7B-Instruct-v1-run0/checkpoint-1400"
    IMG_PATH = "/home/jiyoon/data/imgs/looks_lips/test"     # GT makeup img folder
    JSON_OUTPUT_DIR = "/home/jiyoon/data/json/test_results/shape3"
    APPLIED_OUTPUT_DIR = "/home/jiyoon/data/imgs/test/test_results_applied/test_shape3"
    GT_JSON_PATH = "/home/jiyoon/data/json/makeup_looks_shape3/random_look_2.json"
    FFHQ_FOLDER = "/home/jiyoon/data/imgs/test/bare_face"
    max_tokens = 512

    # Load imgs
    print(f"Processing images from folder: {IMG_PATH}")
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(IMG_PATH, ext)))
    if not image_files:
        print(f"No image files found in {IMG_PATH}")
        return
    print(f"Found {len(image_files)} images to process")
    

    # Load GT data
    print("Loading ground truth data...")
    gt_data = load_gt_data(GT_JSON_PATH)
    print(f"Loaded GT data for {len(gt_data)} makeup looks")


    # Load model
    print("Loading model...")
    model, processor = load_model_and_processor(CKPT_PATH, BASE_MODEL)

    
    for i, image_path in enumerate(image_files, 1):
        print(f"\n--- Processing image {i}/{len(image_files)}: {os.path.basename(image_path)} ---")
        
        try:
            # Get paired files
            makeup_id, ffhq_id = extract_ids_from_filename(image_path)
            gt_solution = gt_data.get(makeup_id, None)
            if not gt_solution:
                print(f"🚨 Warning: No GT solution found for makeup_id {makeup_id}")
            bare_face_path = get_bare_face_path(ffhq_id, FFHQ_FOLDER)
            
            # Generate response
            print("Generating response...")
            response = generate_response(model, processor, image_path, max_tokens)
            json_path = save_result_to_json(response, image_path, JSON_OUTPUT_DIR, BASE_MODEL, CKPT_PATH, gt_solution)
            print(f"✅ JSON result saved to: {json_path}")
            
            print("Applying makeup to bare face image...")
            applied_path = apply_makeup_to_bare_face(json_path, bare_face_path, image_path, ffhq_id, APPLIED_OUTPUT_DIR)
            print(f"✅ Applied makeup image saved to: {applied_path}")
            
        except Exception as e:
            print(f"🚨 Error processing {os.path.basename(image_path)}: {str(e)}")
            continue
    
    print(f"\n🎉 Completed processing all images!")


if __name__ == "__main__": 
    main()