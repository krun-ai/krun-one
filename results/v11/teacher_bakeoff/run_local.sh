#!/bin/bash
# sequential local runs (one model in memory at a time)
cd "$(dirname "$0")"
python3 run_model.py qwen3:4b --think false --num-predict 64 --deadline 900
ollama stop qwen3:4b
python3 run_model.py qwen3:8b --think false --num-predict 64 --subset --deadline 600
python3 run_model.py qwen3:8b --think false --num-predict 64 --deadline 600
ollama stop qwen3:8b
python3 run_model.py gpt-oss:20b --think low --num-predict 1500 --subset --deadline 900 --timeout 600
ollama stop gpt-oss:20b
echo LOCAL_DONE
