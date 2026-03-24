"""
pyworker_config.py — Configuration PyWorker Vast.ai pour VoiceDoc

Ce fichier configure le PyWorker Vast.ai pour qu'il proxifie
les requêtes vers notre serveur FastAPI (worker.py sur port 8080).

Le PyWorker est installé automatiquement par Vast.ai via PYWORKER_REPO.
"""

from vastai import (
    Worker,
    WorkerConfig,
    HandlerConfig,
    BenchmarkConfig,
    LogActionConfig,
)
import base64, tempfile, os, wave, struct, math

MODEL_SERVER_URL  = "http://127.0.0.1"
MODEL_SERVER_PORT = 8080
MODEL_LOG_FILE    = "/var/log/voicedoc/worker.log"

# ─── Benchmark audio synthétique (30s) ────────────────────────────────────────
def _gen_benchmark_b64():
    sr, dur, freq = 16000, 30, 440.0
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        with wave.open(tmp.name, "w") as wf:
            wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr)
            for i in range(sr * dur):
                val = int(32767 * math.sin(2 * math.pi * freq * i / sr))
                wf.writeframes(struct.pack("<h", val))
        path = tmp.name
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    os.unlink(path)
    return b64

BENCH_B64 = _gen_benchmark_b64()

def benchmark_generator():
    return {"audio_b64": BENCH_B64, "audio_ext": "wav", "language": "fr"}

# ─── WorkerConfig ──────────────────────────────────────────────────────────────
worker_config = WorkerConfig(
    model_server_url=MODEL_SERVER_URL,
    model_server_port=MODEL_SERVER_PORT,
    model_log_file=MODEL_LOG_FILE,
    handlers=[
        HandlerConfig(
            route="/transcribe",
            allow_parallel_requests=False,
            max_queue_time=1800.0,
            workload_calculator=lambda p: max(30.0, len(p.get("audio_b64","")) * 0.75 / 750),
            benchmark_config=BenchmarkConfig(
                generator=benchmark_generator,
                runs=2,
                concurrency=1,
            ),
        ),
    ],
    log_action_config=LogActionConfig(
        on_load=["Application startup complete."],
        on_error=[
            "Traceback (most recent call last):",
            "RuntimeError:",
            "CUDA out of memory",
        ],
        on_info=["Loading Whisper", "Loading pyannote", "Processing audio"],
    ),
)

Worker(worker_config).run()
