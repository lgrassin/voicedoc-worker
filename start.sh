#!/bin/bash
# start.sh — Script de démarrage VoiceDoc sur instance Vast.ai
# Lancé par le start-server script de Vast.ai après clonage du repo

# Démarrer notre serveur FastAPI en arrière-plan
python3 /workspace/worker.py &
MODEL_PID=$!
echo "Model server started (PID $MODEL_PID)"

# Attendre que le serveur soit prêt
sleep 5

# Démarrer le PyWorker Vast.ai (proxifie vers notre serveur)
python3 /workspace/pyworker_config.py
