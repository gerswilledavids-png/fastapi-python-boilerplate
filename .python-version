import os
import httpx
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"status": "Face-to-Face Translation Engine Online"}

@app.post("/api/process-audio")
async def process_audio(
    file: UploadFile = File(...),
    source_lang: str = Form(...),
    target_lang: str = Form(...)
):
    openai_key = os.getenv("WHISPER_AI_KEY")
    lemonfox_key = os.getenv("LEMONFOX_API_KEY")
    
    if not openai_key or not lemonfox_key:
        raise HTTPException(status_code=500, detail="Missing API keys on server configuration.")

    async with httpx.AsyncClient(timeout=30.0) as client:
        # 1. WHISPER SPEECH RECOGNITION
        audio_bytes = await file.read()
        whisper_files = {"file": (file.filename, audio_bytes, file.content_type)}
        whisper_data = {"model": "whisper-1", "language": source_lang}
        whisper_headers = {"Authorization": f"Bearer {openai_key}"}

        try:
            whisper_response = await client.post(
                "https://api.openai.com/v1/audio/transcriptions",
                files=whisper_files,
                data=whisper_data,
                headers=whisper_headers
            )
            whisper_response.raise_for_status()
            original_text = whisper_response.json().get("text", "")
        except Exception:
            raise HTTPException(status_code=502, detail="Whisper speech recognition failed.")

        if not original_text.strip():
            return {"original_text": "", "translated_text": "", "audio_url": None, "error": "No speech detected."}

        # 2. MYMEMORY TRANSLATION
        lang_pair = f"{source_lang}|{target_lang}"
        try:
            mymemory_url = f"https://api.mymemory.translated.net/get?q={httpx.URL(original_text)}&langpair={lang_pair}"
            translation_response = await client.get(mymemory_url)
            translation_response.raise_for_status()
            translated_text = translation_response.json()["responseData"]["translatedText"]
        except Exception:
            raise HTTPException(status_code=502, detail="Translation service unavailable.")

        # 3. LEMONFOX TEXT-TO-SPEECH
        tts_headers = {"Authorization": f"Bearer {lemonfox_key}", "Content-Type": "application/json"}
        tts_payload = {"text": translated_text, "language": target_lang}
        
        try:
            tts_response = await client.post(
                "https://api.lemonfox.ai/v1/audio/speech",
                json=tts_payload,
                headers=tts_headers
            )
            tts_response.raise_for_status()
            audio_url = tts_response.json().get("audio_url")
        except Exception:
            audio_url = None

        return {
            "original_text": original_text,
            "translated_text": translated_text,
            "audio_url": audio_url
        }
