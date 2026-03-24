"""
worker.py — VoiceDoc Model Server pour Vast.ai Serverless

Ce fichier tourne sur l'instance GPU et expose un serveur HTTP FastAPI.
Le PyWorker Vast.ai (installé automatiquement) proxifie les requêtes vers ce serveur.

Architecture :
  - Port 8080 : notre serveur FastAPI (transcription + diarisation)
  - Le PyWorker Vast.ai tourne en parallèle et proxifie vers localhost:8080
  - Les modèles sont chargés UNE SEULE FOIS au démarrage

PYWORKER_REPO pointe vers ce repo — Vast.ai clone et lance python worker.py
"""

import os
import base64
import tempfile
import asyncio
import logging
import uvicorn

from fastapi import FastAPI
from pydantic import BaseModel

# ─── Logging vers fichier (lu par LogActionConfig) ─────────────────────────────
LOG_FILE = "/var/log/voicedoc/worker.log"
os.makedirs("/var/log/voicedoc", exist_ok=True)

logging.basicConfig(
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(),
    ],
    level=logging.INFO,
    format="%(asctime)s %(message)s"
)
logger = logging.getLogger("voicedoc")

# ─── Chargement modèles ────────────────────────────────────────────────────────
HF_TOKEN = os.environ.get("HF_TOKEN", "")

logger.info("Loading Whisper large-v3...")
import torch
import soundfile as sf
from faster_whisper import WhisperModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE = "float16" if DEVICE == "cuda" else "int8"
WHISPER = WhisperModel("large-v3", device=DEVICE, compute_type=COMPUTE)
logger.info("Whisper large-v3 loaded.")

logger.info("Loading pyannote speaker-diarization-3.1...")
from pyannote.audio import Pipeline as DiarizePipeline
DIARIZE = DiarizePipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=HF_TOKEN
)
if DEVICE == "cuda":
    DIARIZE = DIARIZE.to(torch.device("cuda"))
logger.info("pyannote loaded.")

# Signal de readiness capturé par LogActionConfig on_load
logger.info("Application startup complete.")

# ─── Serveur FastAPI ───────────────────────────────────────────────────────────
app = FastAPI()

class TranscribeRequest(BaseModel):
    audio_b64: str
    audio_ext: str = "mp3"
    language: str = "auto"

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/transcribe")
async def transcribe(req: TranscribeRequest):
    audio_bytes = base64.b64decode(req.audio_b64)

    with tempfile.NamedTemporaryFile(suffix=f".{req.audio_ext}", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        size_mb = len(audio_bytes) / 1024 / 1024
        logger.info(f"Processing audio: {size_mb:.1f} MB, language={req.language}")

        # Transcription
        lang = None if req.language == "auto" else req.language
        segments_gen, info = WHISPER.transcribe(
            tmp_path, language=lang,
            beam_size=5, word_timestamps=True, vad_filter=True,
        )
        segments = [
            {"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()}
            for s in segments_gen
        ]
        logger.info(f"Transcription: {len(segments)} segments, {info.duration:.1f}s")

        # Diarisation
        data_sf, sr = sf.read(tmp_path, dtype="float32", always_2d=True)
        waveform = torch.tensor(data_sf.T)
        if DEVICE == "cuda":
            waveform = waveform.to(torch.device("cuda"))
        diarization = DIARIZE({"waveform": waveform, "sample_rate": sr})
        annotation = diarization.speaker_diarization

        def get_speaker(start, end):
            times = {}
            for turn, _, spk in annotation.itertracks(yield_label=True):
                ov = min(end, turn.end) - max(start, turn.start)
                if ov > 0:
                    times[spk] = times.get(spk, 0) + ov
            return max(times, key=times.get) if times else "INCONNU"

        result_segs = [
            {"start": s["start"], "end": s["end"],
             "speaker": get_speaker(s["start"], s["end"]), "text": s["text"]}
            for s in segments
        ]

        merged = []
        for seg in result_segs:
            if merged and merged[-1]["speaker"] == seg["speaker"] and \
               seg["start"] - merged[-1]["end"] < 2.0:
                merged[-1]["end"] = seg["end"]
                merged[-1]["text"] += " " + seg["text"]
            else:
                merged.append(dict(seg))

        logger.info(f"Diarisation: {len(merged)} merged segments")

        return {
            "language": info.language,
            "language_probability": round(info.language_probability, 3),
            "duration_seconds": round(info.duration, 1),
            "segments": merged,
        }

    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

# ─── Point d'entrée ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")
