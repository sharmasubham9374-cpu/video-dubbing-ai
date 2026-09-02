# 🎙️ VideoDubber AI - English to Hindi Dubbing Studio
### *English Videos ko Hindi me Dub karein ElevenLabs "Bunty" Voice ke saath*

VideoDubber AI ek complete local AI application hai jo kisi bhi English video ko automatic Hindi me translate aur dub karta hai, using **ElevenLabs Multilingual v2** aur **"Bunty" Voice**.

---

## ✨ Features (खासियतें)

1. **Auto Video Dubbing Pipeline**:
   - 🎵 **Audio Extraction**: Original video se audio track nikalna.
   - 📝 **AI Speech-to-Text**: English dialogue aur exact timestamps detect karna.
   - 🌐 **Natural Hindi Translation**: Dialogues ko natural aur bolchaal wali Hindi me convert karna.
   - 🗣️ **ElevenLabs "Bunty" Voice**: ElevenLabs API se Bunty ki voice me high-quality Hindi speech generate karna.
   - 🎬 **Smart Video Sync & Remux**: Hindi audio ko video ke saath sync karke final output ready karna.

2. **Audio Modes**:
   - **Background Music Ducking**: Original background music/sound effects ko subtle 15% volume par rakhta hai aur Bunty ki Hindi voice clearly upar play hoti hai.
   - **Clean Voice Replace**: Original audio ko poori tarah replace karke sirf Hindi voice rakhta hai.

3. **Subtitles Generator**:
   - Hindi subtitles (`.srt` aur `.vtt`) automatically generate karta hai jise aap download kar sakte hain.

4. **Bunty Voice Integration**:
   - ElevenLabs account me available voices ko scan karke "Bunty" voice ko auto-select karta hai.
   - Studio UI me **"🔊 Test Bunty Voice"** button se sample audio preview sunne ka option.

---

## 🚀 How to Run (Kaise Chalayein)

### 1-Click Launch:
Double-click karein:
```bat
Start-Dubber.bat
```
Browser me automatic `http://localhost:5000` open ho jayega.

---

## ⚙️ Setup & API Keys (Settings)

1. UI ke top-right me **"⚙️ Settings & Keys"** par click karein.
2. Apna **ElevenLabs API Key** dalein ([ElevenLabs.io](https://elevenlabs.io) se free milta hai).
3. Apna **Google Gemini API Key** dalein ([Google AI Studio](https://aistudio.google.com/app/apikey) se free milta hai).
4. **"Bunty Voice ID"** select ya enter karein.
5. **Save & Connect** par click karein.

---

## 📁 Project Structure

```
video-dubbing-ai/
├── Start-Dubber.bat         # One-click Windows launcher
├── server.py                # FastAPI backend API
├── config.json              # Saved configuration & API keys
├── core/
│   ├── audio_extractor.py   # Audio extraction & FFmpeg metadata
│   ├── transcriber.py       # AI speech-to-text with timestamps
│   ├── translator.py        # Natural English-to-Hindi translation
│   ├── elevenlabs_tts.py    # ElevenLabs Bunty voice synthesis
│   ├── video_merger.py      # Audio-video sync & ducking
│   └── subtitles.py         # SRT/VTT subtitle generator
├── web/
│   ├── index.html           # Web Studio UI
│   ├── css/styles.css       # Dark studio styling
│   └── js/app.js            # Studio frontend logic
├── bin/                     # Standalone FFmpeg binaries
└── storage/
    ├── uploads/             # Input videos
    ├── audio/               # Segment audio tracks
    └── outputs/             # Final Hindi dubbed videos
```

