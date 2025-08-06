# 🎥 AskTube-AI — Ask Questions About Any YouTube Video in Real-Time

Welcome to **AskTube-AI**, a powerful Streamlit web app that lets you:
- 🎯 Paste a YouTube link
- 📺 Watch the video directly in the app
- 🧠 Ask questions about the video in real-time
- 🌐 Supports videos in any language (auto-translates to English)
- 🤖 Powered by Google Gemini Pro (not OpenAI)

---

## 🚀 Features

- 🔗 Paste a YouTube video link
- 📝 Auto-fetch transcript (even if it's not in English)
- 🌍 Translates transcript to English using Google Translate
- 💬 Ask questions based on the content
- 🧠 Uses Gemini Pro to answer contextually
- 🎥 Embedded video player inside Streamlit
- 🧹 Clear button to reset everything

---



## 🧰 Tech Stack

- `Streamlit` — Web interface
- `youtube-transcript-api` — To fetch YouTube captions
- `googletrans` — For language translation
- `Gemini Pro` (via `langchain-google-genai`) — LLM for Q&A
- `Python` — Core backend

---

## 🛠️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Kishorsahoo934/AskTube-AI.git
cd AskTube-AI
Install Requirements
pip install -r requirements.txt
GOOGLE_API_KEY=your_api_key_here
Run the App
streamlit run app.py
Folder Structure
bash
Copy
Edit
AskTube-AI/
│
├── app.py                # Main Streamlit app
├── requirements.txt      # All Python dependencies
├── .env                  # Store your Gemini API key here
├── README.md             # You're reading it now!
