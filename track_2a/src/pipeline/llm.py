"""One place to talk to the local language model.

Two kinds of local servers are supported:
    ollama   the Ollama API, e.g. http://localhost:11434
    openai   any OpenAI-compatible server, e.g. vLLM serving Apertus
             at http://localhost:8001/v1

When a JSON schema is given, the server uses constrained decoding: the model
can only write answers that match the schema.

run.py first reads the environment variables of the Hack Apertus template
(configure_from_env), then the command line options (configure), which
take priority.
"""

import json
import os
import urllib.error
import urllib.request


# Defaults, changed by run.py with configure_from_env() and configure()
API = "ollama"
URL = "http://localhost:11434"
MODEL = "qwen3:8b"
API_KEY = ""

MAX_ANSWER_TOKENS = 600
CONTEXT_TOKENS = 8192


def configure(api=None, url=None, model=None):
    """Choose the server type, its address and the model."""
    global API, URL, MODEL

    if api:
        API = api
    if url:
        URL = url
    if model:
        MODEL = model


def configure_from_env(environ=None):
    """Read the variables required by the Hack Apertus template.

        LLM_BASE_URL   endpoint base URL of an OpenAI-compatible server
        LLM_NAME       name of the model on that server
        LLM_API_KEY    API key, only sent when it is set

    A variable that is not set keeps the current value.
    """
    global API, API_KEY

    if environ is None:
        environ = os.environ

    base_url = environ.get("LLM_BASE_URL", "").strip()
    name = environ.get("LLM_NAME", "").strip()
    key = environ.get("LLM_API_KEY", "").strip()

    # The template endpoint is OpenAI-compatible (e.g. vLLM serving Apertus)
    if base_url:
        API = "openai"

    configure(url=base_url, model=name)

    if key:
        API_KEY = key


def chat(prompt, schema=None, timeout=300):
    """Send one user message and return the text of the answer."""
    if API == "ollama":
        return chat_ollama(prompt, schema, timeout)
    if API == "openai":
        return chat_openai(prompt, schema, timeout)
    raise ValueError(f"Unknown API: {API}")


def post_json(url, payload, timeout):
    """POST a JSON payload and return the decoded JSON answer."""
    headers = {"Content-Type": "application/json"}

    # A hosted endpoint needs a key; a local vLLM or Ollama server ignores it
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"

    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)

    except urllib.error.HTTPError as error:
        # Show the server's explanation (e.g. prompt longer than the context)
        # and let the caller fall back instead of stopping the whole run
        details = error.read().decode("utf-8", errors="replace")
        raise ValueError(f"Server error {error.code}: {details[:300]}") from error


def chat_ollama(prompt, schema, timeout):
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {
            "temperature": 0,
            "num_ctx": CONTEXT_TOKENS,
            "num_predict": MAX_ANSWER_TOKENS,
        },
    }

    if schema is not None:
        payload["format"] = schema

    # Only thinking models (Qwen3) accept the "think" switch
    if MODEL.startswith("qwen3"):
        payload["think"] = False

    result = post_json(URL.rstrip("/") + "/api/chat", payload, timeout)
    return result["message"]["content"].strip()


def chat_openai(prompt, schema, timeout):
    url = URL.rstrip("/") + "/chat/completions"
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": MAX_ANSWER_TOKENS,
    }

    if schema is None:
        result = post_json(url, payload, timeout)
        return result["choices"][0]["message"]["content"].strip()

    payload["response_format"] = {
        "type": "json_schema",
        "json_schema": {"name": "answer", "schema": schema},
    }

    try:
        result = post_json(url, payload, timeout)

    except ValueError as error:
        # Older servers may not support schemas: ask for plain JSON instead
        print(f"  Constrained decoding failed, retrying with plain JSON: {error}")
        payload["response_format"] = {"type": "json_object"}
        result = post_json(url, payload, timeout)

    return result["choices"][0]["message"]["content"].strip()


def parse_json(text):
    """Read the JSON object in a model answer.

    Some models add a sentence or ``` fences around the JSON, so we keep only
    the part between the first "{" and the last "}".
    """
    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end <= start:
        raise ValueError(f"No JSON object in the answer: {text[:80]!r}")

    return json.loads(text[start:end + 1])
