import streamlit as st
import utils
import os

# 設定網頁標題與圖示
st.set_page_config(page_title="AI 療癒日記", page_icon="🌿", layout="centered")

# 自定義 CSS 美化介面
st.markdown("""
<style>
    .stTextArea textarea {
        font-size: 16px;
        border-radius: 10px;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        border-radius: 20px;
        padding: 10px 24px;
    }
    .feedback-box {
        background-color: #f0f8ff;
        padding: 20px;
        border-radius: 15px;
        border-left: 5px solid #4CAF50;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.title("🌿 AI 療癒日記 Visualizer")
    st.subheader("寫下你的心情，讓 AI 為你畫出一幅畫...")

    # 側邊欄：API Key 設定
    with st.sidebar:
        # 嘗試從環境變數載入 Token (本地開發用)
        default_token = os.getenv("HUGGINGFACE_TOKEN", "")
        api_key_input = st.text_input("Hugging Face Token", value=default_token, type="password", help="請輸入你的 Hugging Face Access Token (需有 Write 權限)")
        
        if api_key_input:
            # 去除前後空白，避免複製貼上時多餘的空格導致錯誤
            os.environ["HUGGINGFACE_TOKEN"] = api_key_input.strip()
        
        if st.button("🔍 測試 Token 有效性"):
            if not api_key_input:
                st.error("請先輸入 Token")
            else:
                try:
                    from huggingface_hub import HfApi
                    api = HfApi(token=api_key_input.strip())
                    user_info = api.whoami()
                    username = user_info.get('name', 'User')
                    st.success(f"Token 有效！你好, {username}。")
                except Exception as e:
                    st.error(f"Token 無效或無法連線: {e}")
        
        st.info("💡 使用 Hugging Face 免費 API 進行文字分析與繪圖。")
        
        with st.expander("進階設定 (更換模型)"):
            text_model = st.text_input("文字模型 ID", value="Qwen/Qwen2.5-72B-Instruct", help="例如: Qwen/Qwen2.5-72B-Instruct, google/gemma-2-9b-it")
            
            # 提供多個備選模型，讓使用者在 API 忙碌時可以切換
            model_options = [
                "runwayml/stable-diffusion-v1-5",
                "CompVis/stable-diffusion-v1-4",
                "prompthero/openjourney",
                "stabilityai/stable-diffusion-2-1",
                "Custom (自訂)"
            ]
            selected_model = st.selectbox("繪圖模型 ID", model_options, index=0, help="若預設模型忙碌，請嘗試切換其他模型。")
            
            if selected_model == "Custom (自訂)":
                image_model = st.text_input("請輸入自訂模型 ID", value="runwayml/stable-diffusion-v1-5")
            else:
                image_model = selected_model



    # 主要輸入區
    diary_text = st.text_area("親愛的日記...", height=150, placeholder="今天發生了什麼事？你的心情如何？")

    if st.button("✨ 開始療癒分析"):
        if not diary_text.strip():
            st.warning("請先寫下一點內容喔！")
        else:
            if not os.getenv("HUGGINGFACE_TOKEN"):
                st.error("請先在左側輸入 Hugging Face Token！")
            else:
                with st.spinner(f"AI ({text_model}) 正在用心閱讀並構思畫面..."):
                    # 1. 分析日記
                    analysis_result = utils.analyze_diary(diary_text, model_id=text_model)
                    
                    if analysis_result and "error" not in analysis_result:
                        # 2. 顯示分析結果
                        emotion = analysis_result.get("emotion", "平靜")
                        feedback = analysis_result.get("feedback", "...")
                        image_prompt = analysis_result.get("image_prompt", "")

                        st.markdown(f"### 情緒標籤：`{emotion}`")
                        
                        st.markdown(f"""
                        <div class="feedback-box">
                            <b>AI 諮商師的回饋：</b><br>
                            {feedback}
                        </div>
                        """, unsafe_allow_html=True)

                        # 3. 生成圖片
                        if image_prompt:
                            image = None
                            status_container = st.status("正在啟動 AI 繪圖引擎...", expanded=True)
                            
                            def update_status(msg, state):
                                status_container.update(label=msg, state=state)
                                if state == "error":
                                    st.toast(msg, icon="⚠️")
                            
                            # 建構模型列表：使用者選擇的優先，接著是備選列表
                            fallback_models = [
                                "runwayml/stable-diffusion-v1-5",
                                "CompVis/stable-diffusion-v1-4",
                                "prompthero/openjourney",
                                "stabilityai/stable-diffusion-2-1"
                            ]
                            
                            # 確保使用者選擇的模型在第一個，且不重複
                            model_list = [image_model] + [m for m in fallback_models if m != image_model]
                            
                            with status_container:
                                image, success_model = utils.generate_image_with_retry_and_fallback(
                                    image_prompt, 
                                    model_list=model_list,
                                    status_callback=update_status
                                )
                            
                            if image:
                                status_container.update(label=f"圖片生成成功！(使用模型: {success_model})", state="complete", expanded=False)
                            else:
                                status_container.update(label="所有模型嘗試皆失敗", state="error", expanded=True)

                            # 顯示結果或錯誤處理
                            if image:
                                st.image(image, caption="你的心情具象化", use_column_width=True)
                                st.success("希望能讓你感覺好一點！🌻")
                            else:
                                st.warning("API 目前忙碌中或連線失敗，請稍後再試。")
                                    
                    elif analysis_result and "error" in analysis_result:
                        st.error(analysis_result["error"])
                    else:
                        st.error("分析失敗，請檢查 Token 是否正確，或 API 是否忙碌中。")

if __name__ == "__main__":
    main()
