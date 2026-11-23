# Irodori-TTS v4-Large Japanese Studio (Windows + CUDA)

A local, production-ready Windows desktop web application for **Japanese Text-to-Speech (TTS)** and **AI Japanese Script Generation**, built with **Gradio** and **PyTorch CUDA 12.x** for [`Aratako/Irodori-TTS-v4-Large-Quantized`](https://huggingface.co/Aratako/Irodori-TTS-v4-Large-Quantized).

Engineered specifically for **8 GB VRAM NVIDIA GPUs** (such as the **RTX 4060 Laptop GPU**, Ada Lovelace Compute Capability 8.9) using `torchao` FP8/INT4/INT8 quantization (`bfloat16` compute) paired with **CPU-offloaded DACVAE codec** decoding (`~3.9 GB` VRAM in `float8-weight-only`, leaving `~3.5 GB` free for diffusion inference and Windows WDDM).

---

## Core Features

### 1. Model & 8 GB VRAM Management
- **Quantization Selector (`torchao` 0.16.x + PyTorch 2.10.x CUDA):**
  - `float8-weight-only` *(Default — RTX 4060 Ada optimal, ~3.9 GB VRAM)*
  - `float8-dynamic` *(FP8 weights + dynamic FP8 activations)*
  - `int4-weight-only` *(Lowest footprint, ~2.8 GB VRAM)*
  - `int8-weight-only` *(Broad GPU compatibility)*
- **CPU DACVAE Codec Optimization:** Keeps the DACVAE audio codec and watermarker in system RAM by default (`codec_device="cpu"`), saving ~1+ GB of GPU VRAM with optional CUDA toggle and automatic CPU-codec fallback on CUDA OOM.
- **Explicit Lifecycle & Telemetry:** One-click **Load Model to GPU** and **Unload Model (Free VRAM)** (`gc.collect()` + `torch.cuda.empty_cache()`), plus a live VRAM monitor showing both PyTorch allocated VRAM and actual free GPU VRAM.

### 2. All 4 Core Irodori-TTS v4-Large Studio Modes
- **Tab A — Zero-Shot Voice Cloning & Long Reference Audio:** Upload (`.wav`, `.mp3`, `.flac`) or record via microphone, automatic 48 kHz mono resampling & silence preprocessing, long reference audio support (up to 120 s), and optional reference transcript notes.
- **Tab B — Text-based Voice Design:** Synthesize bespoke Japanese voices purely from natural-language descriptions (gender, age, pitch, tone, speaking style) with quick-start templates.
- **Tab C — Style-Controlled Voice Cloning:** Combine reference audio (speaker identity) with a style/emotion caption prompt simultaneously.
- **Tab D — Batch Generation (Multi-Line / TXT / CSV -> ZIP):** Synthesize multi-line scripts or `.txt`/`.csv` files sequentially with live progress tracking and export all `.wav` clips plus `manifest.csv` bundled into a single `.zip` archive.

### 3. Reference Voice Preset Manager (`./presets/`)
- Save uploaded or recorded reference voice samples with custom names (e.g., `Female_Formal`, `Male_Casual`) and optional transcripts.
- Instantly load saved presets across Tabs A, C, and D without re-uploading.

### 4. Audio Post-Processing & Persistent History (`./outputs/`)
- **Pitch-Preserving Speed Adjustment (WSOLA):** Adjust speech tempo (`0.6×`–`1.6×`) without chipmunk pitch distortion.
- **Silence Trimming & WAV Export:** Leading/trailing silence removal and export in `48 kHz` (Native Studio), `44.1 kHz`, or `24 kHz` WAV.
- **Generation History:** Automatically saves timestamped WAV outputs to `./outputs/` and displays the last 15 clips with one-click Replay & Download.

### 5. Japanese NLP Reading Helper & AI Script Generator
- **Live Furigana & Romaji Analyzer:** Context-aware morphological reading preview (`fugashi` + `unidic-lite` + `pykakasi`) displaying `<ruby>` Furigana HTML, full Hiragana, Hepburn Romaji, and a Kanji reading verification glossary before synthesis.
- **One-Click Emoji Emotion & Prosody Inserter:** Quick-insert toolbar for native Irodori-TTS expression emojis (`😆`, `😭`, `😱`, `👂`, `🤭`, `😮‍💨`, `🫶`, `😏`, `🥺`, `⏸️`, `⏩`, `🐢`, etc.).
- **AI Japanese Script & Dialogue Generator:** Generate TTS-ready scripts by topic, **JLPT level** (`N5`–`N1`, `Natural Native`), and **dialect/register** (`Standard Tokyo`, `Keigo / Formal`, `Casual / Anime-style`, `Kansai-ben`) using **Google Gemini** (`google-genai`) or **OpenAI / OpenAI-compatible** APIs (including local Ollama / LM Studio). API keys stay strictly in RAM session state.

---

## Prerequisites (Windows 10/11 64-bit)

1. **Python 3.10 – 3.12 (64-bit)** installed with *"Add Python to PATH"* enabled.
2. **NVIDIA GPU** with up-to-date CUDA 12.x drivers (RTX 4060 8 GB or compatible NVIDIA GPU).
3. **Git for Windows** on `PATH` (required to install `dacvae` and `irodori-tts` from GitHub).

---

## One-Click Setup & Launch

### First-Time Setup (`setup_windows.bat`)
Double-click **`setup_windows.bat`** (or run from Command Prompt / PowerShell):

```bat
setup_windows.bat
```

`setup_windows.bat` automatically:
1. Configures local Hugging Face, PyTorch, and pip caches inside the project directory (`models_cache/`, `.tmp_pip/`) to keep your `C:` drive clean.
2. Sets `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,garbage_collection_threshold:0.8` to minimize Windows WDDM VRAM fragmentation.
3. Creates `.venv` with `--system-site-packages`, verifies or installs CUDA-enabled PyTorch `2.10.x` (`cu128`), installs `requirements.txt` (`torchao 0.16.x`, `gradio`, `transformers`, Japanese NLP, LLM clients), installs `descript-audiotools`, `facebookresearch/dacvae`, and `Aratako/Irodori-TTS`, verifies core imports, and launches the Gradio studio at `http://127.0.0.1:7860`.

### Subsequent Launches (`run_windows.bat`)
Double-click **`run_windows.bat`**:

```bat
run_windows.bat
```

---

## Optional `.env` Configuration

API keys for the AI Script Generator can be entered directly in the UI (stored in session RAM only) or provided via a local `.env` file (excluded from Git via `.gitignore`):

```env
GEMINI_API_KEY=your_gemini_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
```

---

## Project Structure

```text
.
├── app.py                # Multi-tab Gradio Blocks studio UI & state management
├── tts_engine.py         # Irodori-TTS v4-Large torchao backend, WSOLA DSP, presets & history
├── text_processor.py     # fugashi/pykakasi Furigana/Romaji analyzer & Gemini/OpenAI script writer
├── requirements.txt      # Pinned Windows + CUDA 12.x Python dependencies
├── setup_windows.bat     # One-click Windows environment setup, dependency install & launch
├── run_windows.bat       # Quick-launch script
├── presets/              # Local user voice presets (tracked via .gitkeep; audio ignored)
├── outputs/              # Timestamped WAV outputs & batch ZIPs (tracked via .gitkeep; audio ignored)
└── .gitignore            # Excludes .venv/, models_cache/, outputs/*, presets/*, .env, caches
```

---

## Upstream Attribution & Credits

This studio builds upon and wraps the following open-source models and libraries:
- **TTS Model & Inference Runtime:** [`Aratako/Irodori-TTS-v4-Large-Quantized`](https://huggingface.co/Aratako/Irodori-TTS-v4-Large-Quantized) and [`Aratako/Irodori-TTS`](https://github.com/Aratako/Irodori-TTS) by Aratako (Chihiro Arata).
- **Audio Codec:** [`facebookresearch/dacvae`](https://github.com/facebookresearch/dacvae) by Meta FAIR & `Aratako/Semantic-DACVAE-Japanese-32dim`.
- **Quantization:** [`pytorch/ao`](https://github.com/pytorch/ao) (`torchao`).

Please review and comply with the respective upstream model and code licenses before commercial or redistribution use.
