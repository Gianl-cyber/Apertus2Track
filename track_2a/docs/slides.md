---
marp: true
size: 16:9
paginate: true
footer: Aperceptia · NeuralForce · Hack Apertus Track 2A
style: |
  section {
    font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
    background: #ffffff;
    color: #14213d;
    font-size: 22px;
    padding: 52px 64px 56px;
    justify-content: flex-start;
  }
  section::after { color: #8a93a6; font-size: 15px; }
  footer { color: #8a93a6; font-size: 14px; }
  h1 {
    font-size: 36px;
    margin: 0 0 22px;
    padding-left: 16px;
    border-left: 8px solid #d52b1e;
  }
  em { color: #d52b1e; font-style: normal; font-weight: 600; }
  code { background: #eef0f4; color: #14213d; border-radius: 4px; }
  .muted, p.muted { color: #5b6475; }

  /* Title slide */
  section.title { background: #14213d; color: #ffffff; justify-content: center; }
  section.title h1 { font-size: 72px; color: #ffffff; border: none; padding: 0; margin: 0; }
  section.title h2 { font-size: 30px; font-weight: 400; color: #c9d1e3; margin: 6px 0 36px; }
  section.title p { color: #e6e9f0; font-size: 22px; }
  section.title em { color: #ff6b5e; }
---

<!-- _class: title -->
<!-- _paginate: false -->
<!-- _footer: '' -->

![bg right:42%](scene_06_perception.png)

# Aperceptia

## Qwen2.5-VL *sees*, Apertus *decides*

From a photo of a robot workcell and an English instruction: a safe pick-and-place plan, a clarifying question, or a refusal.

**Apertus-v1.5-8B makes every language decision.**

Team NeuralForce · Gian-Luca Tehrani
Hack Apertus · Track 2A · See It, Say It, Pick It (ZHAW)

---

<style scoped>
ol { display: flex; list-style: none; padding: 0; margin: 26px 0 30px; }
ol > li {
  flex: 1; position: relative; margin: 0 16px;
  background: #f4f5f7; border-top: 6px solid #14213d; border-radius: 10px;
  padding: 18px 18px 20px; font-size: 19px; line-height: 1.4;
}
ol > li:first-child { margin-left: 0; }
ol > li:last-child { margin-right: 0; }
ol > li:nth-child(2) { background: #fdeceb; border-top-color: #d52b1e; }
ol > li + li::before {
  content: "➜"; position: absolute; left: -27px; top: 42%;
  color: #8a93a6; font-size: 22px;
}
ol > li strong { display: block; font-size: 22px; margin-bottom: 8px; }
section > p { color: #5b6475; }
</style>

# Pipeline: which model does what

Input: `rgb.png` + an English instruction

1. **Qwen2.5-VL 7B** Only *sees*: lists the objects with one pixel each and their relations. Never decides.
2. **Apertus-v1.5-8B** One call, JSON schema. Decides *execute / ask / refuse* and writes the plan, the question or the reason.
3. **Safety net** Keyword rules *after* Apertus. Can only make the answer more cautious.
4. **Planner** Robot limits (one object, 85 mm gripper), target pixel, official JSON.

3 Docker containers: `ollama` · `apertus` (vLLM, 4-bit) · `pipeline` · one command: `make run`

---

<style scoped>
ol { list-style: none; padding: 0; margin: 6px 0 28px; }
ol > li {
  display: flex; justify-content: space-between;
  margin: 10px 0; padding: 12px 18px; border-radius: 8px; font-size: 20px;
}
ol > li:nth-child(1) { background: linear-gradient(90deg, #d5d9e2 46%, #f4f5f7 46%); }
ol > li:nth-child(2) { background: linear-gradient(90deg, #d5d9e2 38%, #f4f5f7 38%); }
ol > li:nth-child(3) { background: linear-gradient(90deg, #f3b5ae 63%, #f4f5f7 63%); }
ol > li:nth-child(4) { background: linear-gradient(90deg, #d52b1e 74%, #f4f5f7 74%); color: #ffffff; }
ol > li:nth-child(4) strong { color: #14213d; }
ul { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; list-style: none; padding: 0; }
ul > li { background: #f4f5f7; border-radius: 10px; padding: 14px 16px; font-size: 17px; color: #5b6475; }
ul > li strong { display: block; font-size: 34px; color: #14213d; }
section > p { color: #5b6475; }
</style>

# Results on the development set

Official scorer · 10 scenes · 50 instructions · total score / 65

1. Starting point: Qwen3, no safety rules **29.9**
2. Apertus, first prompts: free-form, several calls **24.5**
3. Apertus, one call, JSON schema, safety net **41.1**
4. Final: Apertus, one call, JSON schema, order repair, examples **47.9**

- **26.2** grounding / 35
- **21.6** safety / 30
- **24.5 → 47.9** Apertus score, first prompts to final
- **40 / 40** held-out answers in the official format

---

<style scoped>
ul { list-style: none; padding: 0; margin: 0; }
section > ul > li {
  background: #f4f5f7; border-left: 6px solid #14213d; border-radius: 8px;
  padding: 12px 18px; margin-bottom: 14px; font-size: 18px;
}
section > ul > li:first-child { border-left-color: #d52b1e; background: #fdeceb; }
section > ul > li > strong { display: block; font-size: 21px; margin-bottom: 4px; }
section > ul > li ul li::before { content: "– "; color: #8a93a6; }
section { padding-right: 40px; }
</style>

![bg right:36%](scene_06_perception.png)

# Failure cases and difficulties with Apertus

- **Apertus**
  - First prompts, without JSON schema: 24.5 / 65, worse than the Qwen3 starting point
  - Misses obvious refusals: plate, throw, pull from under, contradiction, hand (5 caught by the safety net)
  - Three identical red cubes, "pick the red cube": picks one instead of asking which one
  - "the red gear under the yellow gear in the tray": mixes up which object is on top
- **Setup**
  - Upstream vLLM does not support Apertus 1.5: swiss-ai image, 4-bit + fp8 KV cache on 12 GB, the two models run one after the other
- **Perception** (image: Qwen2.5-VL output on a dev scene)
  - The red gear *under* the yellow gear is not detected · black tray never detected · grey parts mislabelled

---

<style scoped>
section > ul { display: grid; grid-template-columns: 1fr 1fr; gap: 28px; list-style: none; padding: 0; margin-top: 8px; }
section > ul > li {
  background: #f4f5f7; border-top: 6px solid #14213d; border-radius: 10px;
  padding: 20px 24px; font-size: 19px;
}
section > ul > li:last-child { border-top-color: #d52b1e; }
section > ul > li > strong { display: block; font-size: 24px; margin-bottom: 10px; }
section > ul > li ul { padding-left: 20px; }
section > ul > li ul li { margin-bottom: 10px; }
</style>

# Lessons and next steps

- **Lessons**
  - *Constrained decoding* is the biggest lever: object ids as an enum stop invented objects and broken formats
  - One short, focused prompt per instruction beats a chain of prompts
  - Let Apertus decide; rules only as a safety net *after* it
  - Perception is now the bottleneck: a wrong object list cannot be fixed later
- **Next steps**
  - A detector or segmenter for exact masks, the black tray and stacked objects
  - Make Apertus count identical objects before deciding, to ask when needed
  - Fine-tune Apertus on refusals and clarifications, so the safety net is no longer needed
