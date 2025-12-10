import os
import json
from PIL import Image
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
import torch
from diffusers import StableDiffusionPipeline

# 載入環境變數
load_dotenv()

# 預設模型 ID
DEFAULT_REPO_ID_TEXT = "Qwen/Qwen2.5-72B-Instruct"
# SD 1.5 本地執行 ID
DEFAULT_REPO_ID_IMAGE = "runwayml/stable-diffusion-v1-5"

def get_client():
    token = os.getenv("HUGGINGFACE_TOKEN")
    if not token:
        return None
    try:
        token.encode('ascii')
    except UnicodeEncodeError:
        return None
    return InferenceClient(token=token)

def load_image_pipeline(model_id=DEFAULT_REPO_ID_IMAGE):
    """
    載入本地 Stable Diffusion 模型 (Diffusers)。
    這會下載模型權重 (約 4GB)，建議使用 cache。
    """
    token = os.getenv("HUGGINGFACE_TOKEN")
    
    print(f"Loading local pipeline for: {model_id}")
    
    # 判斷是否有 GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device}")
    
    try:
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id, 
            use_auth_token=token,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        )
        pipe = pipe.to(device)
        # 啟用記憶體優化 (如果有的話)
        # pipe.enable_attention_slicing() 
        return pipe
    except Exception as e:
        print(f"Error loading pipeline: {e}")
        raise e # 將錯誤拋出，讓前端可以顯示

def analyze_diary(text, model_id=DEFAULT_REPO_ID_TEXT):
    """
    使用 Hugging Face InferenceClient 分析日記 (維持 API 呼叫)。
    """
    if not text:
        return None

    client = get_client()
    if not client:
        return {"error": "請輸入有效的 Hugging Face Token (僅限英文數字)"}

    # 建構 Prompt
    system_prompt = (
        "You are a warm, empathetic AI counselor. Read the user's diary and provide a response in the following JSON format ONLY:\n"
        "{\n"
        '  "emotion": "Identify the main emotion (in Traditional Chinese)",\n'
        '  "feedback": "A warm, healing response (in Traditional Chinese, around 50 words)",\n'
        '  "image_prompt": "A detailed artistic description of a scene representing the mood for image generation (in English)"\n'
        "}\n"
        "Do not output anything else. Just the JSON."
    )
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text}
    ]

    try:
        response = client.chat_completion(
            messages=messages,
            model=model_id,
            max_tokens=500,
            temperature=0.7
        )
        
        generated_text = response.choices[0].message.content
        
        try:
            start = generated_text.find('{')
            end = generated_text.rfind('}') + 1
            if start != -1 and end != -1:
                json_str = generated_text[start:end]
                result = json.loads(json_str)
                return result
            else:
                return {
                    "emotion": "分析完成",
                    "feedback": generated_text.replace(system_prompt, "").strip(),
                    "image_prompt": "Calm healing atmosphere, soft lighting"
                }
        except Exception:
            return {
                "emotion": "複雜",
                "feedback": generated_text.replace(system_prompt, "").strip(),
                "image_prompt": "Abstract healing art, soft colors"
            }

    except Exception as e:
        error_msg = str(e)
        if "401" in error_msg:
            return {"error": "Token 無效或權限不足。"}
        if "404" in error_msg:
            return {"error": f"模型找不到 ({model_id})。請嘗試更換其他模型 ID。"}
        if "503" in error_msg:
            return {"error": "模型正在啟動中 (Cold Boot)，請稍後再試。"}
        if "not supported" in error_msg:
            return {"error": f"模型不支援 ({model_id})。請嘗試更換其他模型 ID。"}
        print(f"Error in analyze_diary: {e}")
        return {"error": f"API 錯誤: {error_msg}"}

def generate_image_api(prompt, model_id=DEFAULT_REPO_ID_IMAGE):
    """
    使用 Hugging Face InferenceClient 生成圖片 (API 模式)。
    """
    if not prompt:
        return None

    client = get_client()
    if not client:
        return None

    try:
        # 使用 text_to_image API
        image = client.text_to_image(
            prompt,
            model=model_id
        )
        return image
    except Exception as e:
        print(f"Error in generate_image_api: {e}")
        return None

def generate_image_local(pipeline, prompt):
    """
    使用本地 Diffusers Pipeline 生成圖片 (本地模式)。
    """
    if not prompt or not pipeline:
        return None

    try:
        # 生成圖片
        image = pipeline(prompt).images[0]
        return image
    except Exception as e:
        print(f"Error in generate_image_local: {e}")
        return None
