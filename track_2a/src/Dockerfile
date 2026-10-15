# Pipeline image: our Python code and the scenes, no model inside.
# Built from track_2a/ (see ../docker-compose.yml). The models run in
# their own boxes:
#   ollama   Qwen2.5-VL 7B, perception (scene graph)
#   apertus  vLLM serving Apertus-v1.5-8B, decision and plan
FROM python:3.11-slim

# Print the logs as they come, not at the end of the run
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY src/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ .
# The scenes are inside the image, no data folder is mounted
COPY data/ data/

# outputs/ is the only folder linked to the computer (docker-compose.yml)
CMD ["python", "run.py", "--data", "data/heldout", "--out", "outputs/results_heldout.json"]
