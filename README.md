# AI 療癒日記 Visualizer (AI Healing Diary)

這是一個結合 **大型語言模型 (LLM)** 與 **文字生成圖片 (Text-to-Image)** 技術的 Streamlit Web 應用程式。
使用者只需輸入今天的日記，AI 諮商師就會分析你的情緒，給予暖心的回饋，並自動生成一張專屬於你當下心情的藝術畫作。

## ✨ 功能特色

*   **情緒分析**：自動識別日記中的核心情緒（如：焦慮、平靜、興奮）。
*   **暖心回饋**：扮演同理心的 AI 諮商師，提供約 50 字的心理支持與建議。
*   **心情畫作**：根據日記的情境與情緒，自動生成一張藝術風格的圖片。
*   **高度客製化**：支援使用者自行更換 Hugging Face 上的文字或繪圖模型 ID。
*   **混合架構 (Hybrid Mode)**：優先使用免費 API，若遇忙碌或錯誤，自動切換至本地端執行 (需 GPU)。
*   **免費使用**：基於 Hugging Face Inference API (Free Tier)，無需綁定信用卡即可體驗 (需自行申請 Token)。

## 🛠️ 技術棧

*   **前端框架**：[Streamlit](https://streamlit.io/)
*   **API 整合**：[Hugging Face Hub](https://huggingface.co/docs/huggingface_hub/index)
*   **預設模型**：
    *   文字 (LLM)：`Qwen/Qwen2.5-72B-Instruct` (可替換)
    *   繪圖 (Image)：`runwayml/stable-diffusion-v1-5` (可替換)
*   **語言**：Python 3.8+

## 🚀 快速開始

### 1. 安裝依賴
請確保你的電腦已安裝 Python，然後執行以下指令安裝所需套件：

```bash
pip install -r requirements.txt
```

### 2. 設定 Hugging Face Token
1.  註冊/登入 [Hugging Face](https://huggingface.co/)。
2.  前往 [Settings > Access Tokens](https://huggingface.co/settings/tokens)。
3.  建立一個新 Token，權限請選擇 **"Write"** (或確保有 Inference 權限)。
4.  在專案根目錄建立 `.env` 檔案，並填入你的 Token：
    ```env
    HUGGINGFACE_TOKEN=你的_hf_token_這裡
    ```
    *(或者你也可以直接在程式介面的側邊欄輸入)*

### 3. 執行程式
在終端機 (Terminal) 執行：

```bash
streamlit run app.py
```

程式啟動後，瀏覽器會自動打開 (預設網址 `http://localhost:8501`)。

### 4. 開始使用
1.  程式會自動讀取 `.env` 中的 Token。若未設定，請在左側側邊欄輸入。
2.  (選用) 在「進階設定」中，你可以更換想要使用的模型 ID。
3.  在主畫面輸入你的日記。
4.  點擊 **「✨ 開始療癒分析」**。

## ⚠️ 常見問題與故障排除

*   **API 錯誤 (404/Model not found)**：
    *   這通常表示該模型目前在 Hugging Face 免費 API 上不可用。
    *   **解法**：請展開側邊欄的「進階設定」，嘗試更換其他模型 ID (例如 `google/gemma-2-9b-it` 或 `HuggingFaceH4/zephyr-7b-beta`)。

*   **API 錯誤 (503 Service Unavailable)**：
    *   表示模型正在冷啟動 (Cold Boot) 或伺服器忙碌。
    *   **解法**：請等待約 20-60 秒後再試一次。

*   **圖片生成失敗 (API Busy)**：
    *   免費 API 的圖片生成服務較不穩定，容易忙碌。
    *   **解法**：
        1. 多按幾次按鈕重試。
        2. 程式會自動嘗試切換到 **本地模式 (Local Execution)** (若已安裝相關套件且有 GPU)。
        3. 更換繪圖模型 ID。

## 📂 檔案結構

*   `app.py`: 主程式，負責 UI 介面與邏輯串接。
*   `utils.py`: 核心工具組，負責與 Hugging Face API 溝通。
*   `requirements.txt`: 專案依賴列表。
*   `.env`: 環境變數設定檔 (存放 Token)。
*   `agent_conversation_log.txt`: 開發過程的完整對話紀錄。

## 📚 課程理論與實作對應

本專案實作了課程中提到的 **生成式 AI (Generative AI)** 核心概念，特別是 **擴散模型 (Diffusion Model)** 的應用。

### 1. 核心模型：Stable Diffusion 1.5
*   **課程概念**：老師推薦使用 SD 1.5，因為它資源最豐富且對硬體要求較低，適合教學與 Colab 環境。
*   **本專案實作**：我們預設使用 `runwayml/stable-diffusion-v1-5` 模型，這正是課程中提到的 SD 1.5 版本。這確保了生成的穩定性與相容性，並透過 Hugging Face Inference API 讓使用者無需本地 GPU 也能體驗。

### 2. 模型架構 (Under the Hood)
雖然我們透過 API 呼叫，但背後運作的原理與課程介紹一致：
*   **Text Encoder (CLIP)**：當你在日記中輸入文字時，後端的 CLIP 模型會將其轉換為電腦可理解的向量 (Embedding)。
*   **U-Net (Denoising)**：這是擴散模型的核心，負責預測並去除雜訊，逐步從隨機雜訊中還原出符合你日記情境的圖像。
*   **VAE (Variational Autoencoder)**：最後將潛在空間 (Latent Space) 的向量解碼回我們肉眼可見的圖片。

### 3. Token 的雙重意涵
本專案中同時涉及了課程提到的兩種 Token 概念：
*   **Hugging Face Token**：你在側邊欄輸入的 `hf_...` 金鑰，用於驗證身分以存取 Hugging Face 的 API 服務 (特別是像 SD 1.5 這種需要資源的模型)。
*   **Text Token**：當 LLM 分析你的日記時，它會先將你的文字切分成一個個 Token (詞元) 才能進行運算。

---
**Note**: 本專案為教學與練習用途，使用之 API 服務穩定性取決於 Hugging Face 官方。
