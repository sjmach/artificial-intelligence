#!/bin/sh
echo "Starting temporary initialization server..."
ollama serve &

until ollama list 2>/dev/null; do
  echo "Waiting for engine to respond..."
  sleep 2
done

echo "1. Downloading raw base model..."
ollama pull llama3.2:3b

echo "2. Compiling optimized 4K CPU model variant..."
ollama create llama3.2:3b-clean -f /config/Modelfile

echo "Initialization complete."
exit 0
