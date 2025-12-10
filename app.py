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
        st.header("設定")
        api_key_input = "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"
        if api_key_input:
            # 去除前後空白，避免複製貼上時多餘的空格導致錯誤
            os.environ["HUGGINGFACE_TOKEN"] = api_key_input.strip()
        
        st.info("💡 使用 Hugging Face 免費 API (文字) + 本地 Diffusers (繪圖)。第一次執行繪圖需下載模型 (約 4GB)，請耐心等候。")
        
        with st.expander("進階設定 (更換模型)"):
            text_model = st.text_input("文字模型 ID", value="Qwen/Qwen2.5-72B-Instruct", help="例如: Qwen/Qwen2.5-72B-Instruct, google/gemma-2-9b-it")
            image_model = st.text_input("繪圖模型 ID (本地)", value="runwayml/stable-diffusion-v1-5", help="例如: runwayml/stable-diffusion-v1-5")

    # 初始化 session state
    if "use_local_mode" not in st.session_state:
        st.session_state.use_local_mode = False

    # 側邊欄控制
    st.sidebar.title("⚙️ 設定")
    use_local = st.sidebar.checkbox("開啟本地模式 (Local Mode)", value=st.session_state.use_local_mode, help="勾選後將使用電腦的 GPU/CPU 生成圖片，需下載模型。")
    
    # 更新 session state
    st.session_state.use_local_mode = use_local

    # 顯示目前模式
    mode_status = "💻 本地模式 (Local)" if st.session_state.use_local_mode else "☁️ 雲端模式 (API)"
    st.sidebar.markdown(f"### 目前模式：{mode_status}")

    # 快取載入繪圖模型 (只會執行一次)
    # V2: 改名以強制清除舊的 cache
    @st.cache_resource
    def get_pipeline_v2(model_name):
        return utils.load_image_pipeline(model_name)

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
                            
                            # 模式 A: 本地模式
                            if st.session_state.use_local_mode:
                                try:
                                    status_container = st.status("正在啟動本地繪圖引擎...", expanded=True)
                                    with status_container:
                                        st.write("正在檢查/下載模型權重 (首次執行需下載約 4GB)...")
                                        st.write("這可能需要幾分鐘，請勿關閉視窗...")
                                        # 呼叫 V2 函數
                                        pipeline = get_pipeline_v2(image_model)
                                        st.write("模型載入完成！正在生成圖片...")
                                        
                                        if pipeline:
                                            image = utils.generate_image_local(pipeline, image_prompt)
                                            status_container.update(label="圖片生成完成！", state="complete", expanded=False)
                                        else:
                                            status_container.update(label="模型載入失敗", state="error")
                                            st.error("無法載入 Pipeline，請檢查 Log。")
                                except Exception as e:
                                    st.error(f"本地模型執行錯誤: {str(e)}")
                                    st.info("建議：請檢查你的網路連線，或確認磁碟空間是否足夠。")
                            
                            # 模式 B: API 模式 (預設)
                            else:
                                with st.spinner(f"正在為你繪製專屬的心情畫作 (使用 API {image_model})..."):
                                    image = utils.generate_image_api(image_prompt, model_id=image_model)

                            # 顯示結果或錯誤處理
                            if image:
                                st.image(image, caption="你的心情具象化", use_column_width=True)
                                st.success("希望能讓你感覺好一點！🌻")
                            else:
                                if st.session_state.use_local_mode:
                                    st.error("本地生成失敗，請檢查上方錯誤訊息。")
                                else:
                                    st.warning("API 目前忙碌中或連線失敗。")
                                    st.info("💡 建議：請勾選左側側邊欄的 **「開啟本地模式 (Local Mode)」**，使用電腦算圖，保證成功！")
                                    
                    elif analysis_result and "error" in analysis_result:
                        st.error(analysis_result["error"])
                    else:
                        st.error("分析失敗，請檢查 Token 是否正確，或 API 是否忙碌中。")

if __name__ == "__main__":
    main()
