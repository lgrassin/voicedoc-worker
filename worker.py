"""
worker.py — VoiceDoc PyWorker pour Vast.ai Serverless

Ce fichier est cloné depuis le repo PYWORKER_REPO par le start-server script
de Vast.ai sur chaque instance GPU.

Architecture :
  - Les modèles Whisper + pyannote sont pré-installés dans l'image Docker
    (Dockerfile.voicedoc) mais téléchargés au premier démarrage via HF cache
  - WorkerConfig expose un endpoint /transcribe
  - HandlerConfig reçoit l'audio en base64, retourne le transcript JSON
  - BenchmarkConfig mesure les perfs pour le load balancing automatique

Variables d'environnement (injectées par Vast.ai depuis Account Settings) :
  HF_TOKEN — token HuggingFace pour pyannote
"""

import os
import base64
import tempfile
import torch
import soundfile as sf

from vastai import (
    Worker,
    WorkerConfig,
    HandlerConfig,
    BenchmarkConfig,
    LogActionConfig,
)

# ─── Configuration ─────────────────────────────────────────────────────────────

MODEL_SERVER_URL  = "http://127.0.0.1"
MODEL_SERVER_PORT = 18000   # Pas utilisé (on est notre propre backend)
MODEL_LOG_FILE    = "/var/log/voicedoc/worker.log"

HF_TOKEN = os.environ.get("HF_TOKEN", "")

# ─── Chargement modèles ────────────────────────────────────────────────────────
# Chargés UNE SEULE FOIS au démarrage du worker — c'est le gain principal
# vs l'approche instance à la demande (15 min → 30s)

import logging
logging.basicConfig(
    filename=MODEL_LOG_FILE if os.path.exists("/var/log/voicedoc") else "/tmp/voicedoc.log",
    level=logging.INFO,
    format="%(asctime)s %(message)s"
)
logger = logging.getLogger("voicedoc")

# Signal de début de chargement (capturé par LogActionConfig on_info)
print("VoiceDoc worker starting...", flush=True)
logger.info("VoiceDoc worker starting...")

print("Loading Whisper large-v3...", flush=True)
logger.info("Loading Whisper large-v3...")
from faster_whisper import WhisperModel
WHISPER = WhisperModel("large-v3", device="cuda", compute_type="float16")
print("Whisper large-v3 loaded.", flush=True)
logger.info("Whisper large-v3 loaded.")

print("Loading pyannote speaker-diarization-3.1...", flush=True)
logger.info("Loading pyannote speaker-diarization-3.1...")
from pyannote.audio import Pipeline as DiarizePipeline
DIARIZE = DiarizePipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    token=HF_TOKEN
)
DIARIZE = DIARIZE.to(torch.device("cuda"))
print("pyannote loaded.", flush=True)
logger.info("pyannote loaded.")

# Signal de fin de chargement (capturé par LogActionConfig on_load)
print("VoiceDoc worker ready.", flush=True)
logger.info("VoiceDoc worker ready.")


# ─── Logique de transcription ──────────────────────────────────────────────────

def _process_audio(audio_b64: str, audio_ext: str, language: str) -> dict:
    """Transcrit + diarise un fichier audio encodé en base64."""
    audio_bytes = base64.b64decode(audio_b64)

    with tempfile.NamedTemporaryFile(suffix=f".{audio_ext}", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        size_mb = len(audio_bytes) / 1024 / 1024
        logger.info(f"Processing audio: {size_mb:.1f} MB, language={language}")

        # Transcription
        lang = None if language == "auto" else language
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
        waveform = torch.tensor(data_sf.T).to(torch.device("cuda"))
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

        # Fusionner segments consécutifs du même locuteur
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


# ─── Benchmark ────────────────────────────────────────────────────────────────
# Petit fichier audio synthétique (silence de 30s) pour mesurer les perfs

import wave, struct, math

def _generate_benchmark_audio_b64() -> str:
    """Génère 30s de sinus 440Hz en WAV, encodé en base64."""
    sample_rate = 16000
    duration    = 30
    frequency   = 440.0
    n_samples   = sample_rate * duration

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        with wave.open(tmp.name, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            for i in range(n_samples):
                val = int(32767 * math.sin(2 * math.pi * frequency * i / sample_rate))
                wf.writeframes(struct.pack("<h", val))
        tmp_path = tmp.name

    with open(tmp_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    os.unlink(tmp_path)
    return b64

BENCHMARK_AUDIO_B64 = _generate_benchmark_audio_b64()

def benchmark_generator() -> dict:
    """Payload de benchmark : 30s d'audio synthétique."""
    return {
        "audio_b64":  BENCHMARK_AUDIO_B64,
        "audio_ext":  "wav",
        "language":   "fr",
    }


# ─── Request parser ───────────────────────────────────────────────────────────

def transcribe_request_parser(payload: dict) -> dict:
    """
    Valide et normalise le payload entrant.
    Le payload attendu du client VoiceDoc :
    {
        "audio_b64": "<base64>",
        "audio_ext": "mp3",
        "language":  "fr"
    }
    """
    if "audio_b64" not in payload:
        raise ValueError("audio_b64 manquant dans le payload")
    payload.setdefault("audio_ext", "mp3")
    payload.setdefault("language", "auto")
    return payload


# ─── Response handler ─────────────────────────────────────────────────────────

async def transcribe_response(result: dict) -> dict:
    """Passe le résultat tel quel au client."""
    return result


# ─── WorkerConfig ─────────────────────────────────────────────────────────────

# Note : model_server_url/port sont requis par WorkerConfig mais VoiceDoc
# n'a pas de serveur HTTP séparé — on utilise le handler directement.
# On pointe vers localhost sur un port inutilisé.

worker_config = WorkerConfig(
    model_server_url=MODEL_SERVER_URL,
    model_server_port=MODEL_SERVER_PORT,
    model_log_file=MODEL_LOG_FILE if os.path.exists("/var/log/voicedoc") else "/tmp/voicedoc.log",

    handlers=[
        HandlerConfig(
            route="/transcribe",

            # Un seul job GPU à la fois (transcription n'est pas parallélisable)
            allow_parallel_requests=False,

            # Timeout file d'attente : 30 min max
            max_queue_time=1800.0,

            # Workload = durée audio estimée (taille base64 / ~750 bytes par seconde)
            workload_calculator=lambda payload: max(
                30.0,
                len(payload.get("audio_b64", "")) * 0.75 / 750
            ),

            request_parser=transcribe_request_parser,

            benchmark_config=BenchmarkConfig(
                generator=benchmark_generator,
                runs=3,
                concurrency=1,  # Un seul job à la fois
            ),
        ),
    ],

    log_action_config=LogActionConfig(
        on_load=[
            "VoiceDoc worker ready.",
        ],
        on_error=[
            "Traceback (most recent call last):",
            "RuntimeError:",
            "CUDA out of memory",
        ],
        on_info=[
            "Loading Whisper",
            "Loading pyannote",
            "Processing audio",
        ],
    ),
)


# ─── Point d'entrée ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    Worker(worker_config).run()
