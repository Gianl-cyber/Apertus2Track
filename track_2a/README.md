# Aperceptia — See It, Say It, Pick It

**Hack Apertus · Track 2A (ZHAW) · Team NeuralForce (Gian-Luca Tehrani)**

Aperceptia turns a photo of a robot workcell and an English instruction into a
safe pick-and-place plan: an open vision model (Qwen2.5-VL) lists the objects,
and **Apertus-v1.5-8B does all the language reasoning**: it decides whether to
execute, ask the user or refuse, and writes the plan.

## Pipeline

```
rgb.png ──► Qwen2.5-VL 7B (Ollama) ──► scene graph: objects, pixels, relations
                                             │
instruction ─────────────────────────────────┤
                                             ▼
                     Apertus-v1.5-8B (vLLM, 4-bit), one call, constrained JSON
                     ──► decision (execute | ask_user | refuse) + message + actions
                                             │
                                             ▼
                     Safety net (deterministic, after Apertus)
                     can only turn execute / ask_user into refuse
                                             │
                                             ▼
                     Planner: checks the plan against the robot limits,
                     computes target_pixel ──► results JSON (official schema)
```

| Component | Model / code | Role |
|---|---|---|
| Perception | Qwen2.5-VL 7B via Ollama, `src/pipeline/perception.py` | "Sees": lists the objects with one pixel each and their relations (inside, on top of). Adds the two documented trays if it missed them. |
| Decision | **Apertus-v1.5-8B** via vLLM, `src/pipeline/decision.py` | One call per instruction. The JSON schema only allows `execute`, `ask_user`, `refuse` and the object ids seen in the scene. The prompt gives the robot limits and a few examples with made-up objects. |
| Safety net | `src/pipeline/guardrails.py` | Keyword rules checked **after** Apertus (person or hand, uncontrolled motion such as throw or pull, self-contradiction, objects wider than the 85 mm gripper). They can only make the answer more cautious. |
| Planner | `src/pipeline/planner.py` | Repairs `pick, pick, place, place` into pairs, drops impossible picks, checks the gripper state, converts the first target to full-image pixels. |

Other open models: Qwen2.5-VL is only used to see the image. Qwen3-8B was used
during development as a stand-in reasoning model before Apertus was set up;
it is not used in the submitted pipeline.

AI assistance: ChatGPT and Claude Code were used as coding and writing
assistants during development. They are not part of the pipeline.

## Requirements

- **Runtime:** Docker with Compose v2, the
  [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/),
  and `make`. Nothing else is installed on the host: Python and all
  dependencies live in the containers. Tested on Windows 11 + WSL2.
- **Hardware:** one NVIDIA GPU.
  - `make run`: ≥ 40 GB VRAM (Apertus and Qwen2.5-VL on the same GPU; vLLM
    reserves 80 % of the GPU memory).
  - `make perception` then `make decision`: 12 GB VRAM is enough (tested on an
    RTX 4070 Super): the two models never run at the same time.
  - Disk: about 40 GB free for the Docker images and the model weights.
- **API keys:** `HF_TOKEN`, **required**: a Hugging Face token (read access)
  of an account that has accepted the conditions of
  [swiss-ai/Apertus-v1.5-8B](https://huggingface.co/swiss-ai/Apertus-v1.5-8B).
  Optional, from the Hack Apertus template: `LLM_BASE_URL`, `LLM_NAME`,
  `LLM_API_KEY` to use another OpenAI-compatible Apertus endpoint instead of
  the local vLLM container.
- **Model weights:** none in this repository. They are downloaded at each run
  inside the containers:
  - `qwen2.5vl:7b` from the Ollama library (about 6 GB);
  - `swiss-ai/Apertus-v1.5-8B` from Hugging Face (about 16 GB), quantised to
    4-bit on load with bitsandbytes.

## Run it

From `track_2a/`:

```bash
export HF_TOKEN=hf_your_token
make run
```

The answers for the held-out scenes are written to
`outputs/results_heldout.json`. The first run takes a while: Docker downloads
the images and the weights, then the healthchecks wait until both models are
loaded.

On a 12 GB GPU, run the two steps one after the other:

```bash
make perception    # Qwen2.5-VL writes the scene graphs to outputs/heldout/
make decision      # Apertus reads them and writes outputs/results_heldout.json
```

Unit tests (no model, no GPU): `make test`.

Everything stays inside Docker. The only folder written on the host is
`outputs/`. `docker compose down` stops and removes the containers.

## Data

`data/` (about 64 MB) holds the challenge scenes:

- `data/dataset/`: the 10 development scenes. `ground_truth/` is only read by
  the official scorer (`src/starter/evaluate.py`), never by the pipeline.
- `data/heldout/`: the 5 held-out scenes, without answers.

The data is released by ZHAW under CC BY 4.0. Please cite:

- F. Mohammadi Amin et al., "A Mixed-Perception Approach for Safe Human–Robot
  Collaboration in Industrial Automation", *Sensors* 20(21):6347, 2020.
  doi:10.3390/s20216347
- C. Munasinghe, F. Mohammadi Amin, D. Scaramuzza, H. W. van de Venn, "COVERED,
  CollabOratiVE Robot Environment Dataset for 3D Semantic Segmentation",
  IEEE ETFA 2022.
- F. Mohammadi Amin et al., "Enhancing Human-Robot Collaboration: A Sim2Real
  Domain Adaptation Algorithm for Point Cloud Segmentation in Industrial
  Environments", *J. Intell. Robot. Syst.* 2025. doi:10.1007/s10846-025-02290-9

## Results

Development set (10 scenes, 50 instructions), official scorer:

| Version | Grounding /35 | Safety /30 | Total /65 |
|---|---|---|---|
| Start (Qwen3, no rules) | 24.5 | 5.4 | 29.9 |
| Apertus, first prompts | 14.0 | 10.5 | 24.5 |
| Apertus, one call + JSON schema + compact prompt + safety net after | 22.8 | 18.4 | 41.1 |
| **Apertus + order repair + examples in the prompt (final)** | **26.2** | **21.6** | **47.9** |

The safety net changed Apertus' answer 5 times on the development set and
2 times on the held-out set.

- Development answers: [`docs/results_dev.json`](docs/results_dev.json).
  Score them with `python src/starter/evaluate.py docs/results_dev.json --data data/dataset`.
- Held-out answers (40 instructions, schema-valid):
  [`docs/results_heldout.json`](docs/results_heldout.json).

Known weaknesses: no clarification question when three identical objects are
present, "X under Y in Z" is often misread, and perception misses the black
tray, stacked objects and some grey parts.

## Challenge rules compliance

- Apertus makes every language decision, including `ask_user` and `refuse`.
  The keyword rules are only a safety net after Apertus.
- No ground truth in the predictions, no rule tied to a scene id, an
  instruction id or a pixel.
- The held-out scenes were only run through the pipeline; no code or prompt
  was written after looking at them.

## Repository layout

| Path | Content |
|---|---|
| `Makefile` | `make run`, `make perception`, `make decision`, `make test` |
| `docker-compose.yml` | the three containers: `ollama`, `apertus`, `pipeline` |
| `src/` | `run.py`, `pipeline/`, `tests/`, the official `starter/` scorer and `schema/`, the pipeline `Dockerfile` |
| `src/docker/` | Apertus 1.5 vLLM image + bitsandbytes |
| `data/` | the scenes (see Data) |
| `docs/` | results, slides |
| `technical_report.md` | the technical report |

## License

Code: Apache 2.0 (see `../LICENSE`), like the challenge. Technical report and
slides: CC-BY-4.0. Data: CC BY 4.0 (ZHAW).
