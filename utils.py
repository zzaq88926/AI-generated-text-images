import os
import time
import random
import json
from PIL import Image
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

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
    單次嘗試生成圖片 (內部使用)。
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
        print(f"Error in generate_image_api ({model_id}): {e}")
        raise e

def generate_image_with_retry_and_fallback(prompt, model_list, status_callback=None):
    """
    嘗試生成圖片，包含重試機制與模型自動切換 (Fallback)。
    
    Args:
        prompt (str): 圖片提示詞
        model_list (list): 模型 ID 列表 (優先順序)
        status_callback (callable): 用於更新 UI 狀態的回呼函數 (msg, state)
    
    Returns:
        tuple: (image, success_model_id) or (None, None)
    """
    if not prompt or not model_list:
        return None, None

    max_retries_per_model = 2
    
    for model_id in model_list:
        if status_callback:
            status_callback(f"正在嘗試模型: {model_id}...", "running")
            
        for attempt in range(max_retries_per_model + 1):
            try:
                print(f"Attempt {attempt+1}/{max_retries_per_model+1} for model {model_id}")
                image = generate_image_api(prompt, model_id)
                if image:
                    return image, model_id
            except Exception as e:
                print(f"Failed attempt {attempt+1} for {model_id}: {e}")
                if attempt < max_retries_per_model:
                    wait_time = (attempt + 1) * 2 + random.uniform(0, 1) # Exponential backoff with jitter
                    if status_callback:
                        status_callback(f"模型 {model_id} 暫時忙碌，{int(wait_time)} 秒後重試 ({attempt+1}/{max_retries_per_model})...", "running")
                    time.sleep(wait_time)
                else:
                    if status_callback:
                        status_callback(f"模型 {model_id} 多次嘗試失敗，準備切換下一個模型...", "error")
    
    return None, None


