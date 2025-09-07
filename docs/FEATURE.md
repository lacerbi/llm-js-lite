# llm-js-lite — Feature Spec (Pure Web, WebGPU)

**Version:** 0.1 (draft)
**Target:** Single-model chat UI running fully client-side in the browser using **Transformers.js** with **WebGPU**.
**Model:** `onnx-community/Qwen3-1.7B-ONNX` → subfolder **`onnx/model_q4f16.onnx`** (4‑bit).
**Inference:** Client-side only (no server inference). Static hosting is sufficient.

---

## 1) Summary

A lightweight browser app that lets a user chat with **Qwen3‑1.7B (4‑bit ONNX)** locally. The app loads the model artifacts into the browser, initializes **WebGPU** for acceleration, and performs generation **entirely on device**. It exposes a simple chat interface with streaming token output and minimal settings. First run fetches model files from the specified source (e.g., Hugging Face); subsequent runs use browser caches.

---

## 2) Goals & Non‑Goals

**Goals**

* WebGPU‑first inference (client-side) via **Transformers.js**.
* Single, fixed model: **Qwen3‑1.7B (4‑bit)** at `onnx/model_q4f16.onnx`.
* Clean chat UI with streaming tokens, Stop/Reset, and a few generation controls.
* Local persistence for settings and the active conversation.
* Clear capability checks and actionable error messages (e.g., if WebGPU unavailable).

**Non‑Goals (MVP)**

* No model switching UI, no multi-user accounts, no RAG/tools, no fine‑tuning.
* No server-side inference. WASM fallback is **off by default** (optional later).
* No multi-session management (single conversation in MVP).

---

## 3) Model Artifacts (expected)

Repository: `onnx-community/Qwen3-1.7B-ONNX`, folder: `onnx/`.

**Core files used**

* **`model_q4f16.onnx`** *(4‑bit model graph; primary artifact)*
* `config.json`
* `generation_config.json`
* Tokenizer assets: `tokenizer.json` **or** (`vocab.json` + `merges.txt`) with `tokenizer_config.json`
* `special_tokens_map.json`, `added_tokens.json` *(if present)*
* `chat_template.jinja` *(if compatible; preferred for formatting)*

> The loader will be pinned to `onnx/model_q4f16.onnx` and will prefer `tokenizer.json` when present, falling back to `vocab.json`+`merges.txt`.

---

## 4) User Stories

* As a user, I can open the app and see whether **WebGPU** is available on my device.
* I can click **“Load Qwen3‑1.7B (4‑bit)”** to fetch artifacts and initialize the model.
* I can type a message, see **streaming** output, and click **Stop** to abort generation.
* I can tweak **temperature**, **top\_p**, **top\_k**, **repetition\_penalty**, **max\_new\_tokens**.
* I can **Reset** the conversation and **Clear model cache** if needed.

---

## 5) Scope (MVP vs V1)

**MVP**

* Single conversation; one fixed model; WebGPU‑first; streaming generation.
* Sliding‑window context with a conservative token budget.
* Local persistence (settings + current transcript) via IndexedDB/localStorage.
* Basic status readouts (model loaded, backend/device name, busy/idle, errors).

**V1**

* Web Worker inference (if not already in MVP), improving UI responsiveness.
* Diagnostics (time‑to‑first‑token, tokens/sec), download progress/resume UI.
* Session management (multiple conversations), export/import transcript JSON.
* Optional WASM fallback toggle (off by default), PWA/offline caching.

---

## 6) Architecture (Browser‑Only)

**Runtime**

* **Transformers.js** in the browser using **ONNX Runtime Web** with **WebGPU** backend.
* **Main thread** renders UI; **Web Worker** (recommended) runs generation and streams tokens back via `postMessage`.
* **IndexedDB/Cache Storage** used for model artifacts and transcripts.

**Key Modules (conceptual)**

* **Model Runtime**: Initializes backend (WebGPU), loads artifacts from `onnx/`, warms up model, exposes `generate()` with streaming callbacks and `abort()`.
* **Prompt Builder**: Combines system prompt + sliding history + new user turn. Uses `chat_template.jinja` if compatible; otherwise a simple, instruction‑style fallback.
* **State Store**: Settings, system prompt, messages. Persists locally.
* **UI Components**: Model panel (Load/Unload + cache info), Settings panel, Chat (streaming), Status bar (backend & diagnostics).

---

## 7) Data & Control Flow

1. **App Start** → detect WebGPU. If missing, display guidance and keep **Load** disabled (MVP) or offer WASM toggle (V1).
2. **Load** → fetch artifacts (from HF or mirror), store/cached in browser, initialize pipeline, warm up; show **Ready** state.
3. **Send** → build prompt (system + last *K* turns under token budget), call `generate()`.
4. **Stream** → tokens arrive incrementally → render; user can **Stop** → abort ongoing generation.
5. **Persist** → append assistant/user turns to local store; obey max context window.
6. **Reset/Clear** → clear transcript and/or cached model artifacts on demand.

---

## 8) Prompting & Context Window

* **System Prompt**: A default in `prompts/system.md`. User can view/edit in Settings.
* **Chat Template**: Prefer `chat_template.jinja` when available to match model’s expected formatting. If not supported in‑runtime, use a simple role‑tagged template.
* **Context Management**: Sliding window of last *N* exchanges, limited by `max_context_tokens` (approximated when exact tokenization counts are not exposed). Oldest turns truncated first.
* **Stop Sequences**: Optional, configurable (e.g., model‑specific `eos_token_id`).

---

## 9) Configuration Surface

**User‑visible (persisted)**

* Generation: `max_new_tokens`, `temperature`, `top_p`, `top_k`, `repetition_penalty`, optional `seed`.
* Context: `max_context_tokens`, `max_history_turns`.
* Backend: `useWebGPU` (default **true**), `allowWasmFallback` (default **false**).
* Prompts: `systemPrompt`, optional `stopSequences`.

**Build‑time constants**

* Model descriptor pinned to: `repo = onnx-community/Qwen3-1.7B-ONNX`, `path = onnx/model_q4f16.onnx` plus tokenizer/config assets in `onnx/`.
* Optional mirror base URL if not pulling from HF directly.

---

## 10) UI/UX

**Layout**

* **Left Sidebar**: *Model* (Load/Unload, cache info), *Settings* (generation + context, system prompt editor).
* **Main Pane**: *Chat* (messages, streaming output, Stop, Reset).
* **Footer**: *Status Bar* (WebGPU device/adapter label if available, busy/idle, small diagnostics).

**States & Feedback**

* *Capability banner*: WebGPU OK / not available.
* *Progress*: show download/initialization progress (if available), then *Ready*.
* *Error toasts*: incompatible model, OOM, network failures with actionable suggestions.

**Accessibility**

* Keyboard-first navigation; screen‑reader friendly live region for streaming output; high‑contrast mode toggle.

---

## 11) Performance & Limits (Expectations)

* **Memory**: 1.7B @ 4‑bit ≈ \~0.85 GB for raw weights; runtime overhead (activations, KV cache) raises working set; practical footprint can exceed 1–2 GB depending on sequence lengths and backend.
* **Context**: Start conservatively (e.g., \~1–2k tokens), allow tuning. Document that large contexts may fail on devices with low VRAM.
* **Throughput**: Highly device‑dependent; we will surface TTFT and tokens/sec when feasible.
* **Storage Quotas**: Browser cache/IndexedDB quotas vary; large artifacts can exceed limits—offer *Clear cache* button and guidance.

---

## 12) Error Handling & Diagnostics

* **WebGPU Unavailable**: Clear banner with steps to enable; Load disabled (MVP). Optional WASM fallback in V1.
* **OOM / Allocation Fail**: Suggest reducing `max_new_tokens` and context, closing other GPU‑heavy tabs, or using a device with more memory.
* **Model Incompatibility**: If runtime rejects the 4‑bit ONNX graph, show a friendly explanation and link to documentation.
* **Diagnostics (V1)**: TTFT, tokens/sec, token counts, cache size.

---

## 13) Privacy & Security

* All inference is on‑device; no telemetry.
* If fetching from Hugging Face, files are pulled directly by the browser. (Provide an optional mirror for fully offline usage.)
* No user data is sent to external services.

---

## 14) Packaging & Distribution

* Built as a static site (e.g., Vite). Host locally or on any static host. No special server required.
* (V1) Optional PWA to enable offline usage after first download.

---

## 15) Milestones & Acceptance

**MVP Acceptance Criteria**

* WebGPU capability check is shown at startup.
* Clicking **Load Qwen3‑1.7B (4‑bit)** downloads artifacts from `onnx/` folder, initializes `model_q4f16.onnx`, and reaches *Ready*.
* User can send a prompt and receive **streamed** tokens; **Stop** aborts promptly; **Reset** clears context.
* Settings persist across reloads; *Clear cache* removes model artifacts and transcript.

**V1 Enhancements**

* Worker‑based inference; diagnostics panel; session management; export/import; optional WASM fallback; PWA/offline.

---

## 16) Risks & Mitigations

* **Browser/WebGPU variability** → Clear capability checks; conservative defaults; documented workarounds.
* **Storage quotas** → Make caching optional; provide *Clear cache* and warnings; consider streaming without persistent cache when needed.
* **Model graph compatibility** → Pin exact artifact names; validate at load; ship troubleshooting guide.
* **Performance on low‑end GPUs** → Document expectations; suggest smaller contexts and fewer new tokens.

---

## 17) Open Questions

1. Should we allow a manual **WASM fallback** if WebGPU is unavailable (off by default)?
2. Are we comfortable fetching directly from the HF repo, or do we want to expose a **mirror base URL** that can be swapped at build time?
3. Do we want **export/import** of conversation JSON in MVP, or defer to V1?
4. Any preferred **default system prompt** content for `prompts/system.md`?

---

## 18) Appendix — Known Filenames in `onnx/` (for reference)

From the repository listing (illustrative):

* `.gitattributes`
* `README.md`
* `added_tokens.json`
* `chat_template.jinja`
* `config.json`
* `generation_config.json`
* `merges.txt`
* `special_tokens_map.json`
* `tokenizer.json`
* `tokenizer_config.json`
* `vocab.json`
* **`model_q4f16.onnx`** *(target model file)*

> The loader will reference these names exactly, prioritizing `tokenizer.json` if present and falling back to `vocab.json`+`merges.txt` otherwise.
