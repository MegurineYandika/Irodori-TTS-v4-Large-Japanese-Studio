"""
app.py
------
Irodori-TTS v4-Large Japanese Studio — Local Windows Desktop Web Application
built with Gradio Blocks and PyTorch CUDA.

Target Hardware:
  Windows 10/11 (64-bit) | AMD Ryzen 7 7435HS | NVIDIA RTX 4060 Laptop GPU (8GB VRAM, Ada CC 8.9)

Features:
  1. Model & VRAM Management (torchao variant selector, Load/Unload, real-time VRAM meter)
  2. AI Japanese Text & Dialogue Generator (Gemini / OpenAI-compatible, JLPT, Dialects)
     + Live Furigana (Hiragana / Ruby) & Romaji Analyzer + One-Click Emoji Emotion Inserter
  3. All 4 Core Irodori-TTS-v4-Large Modes:
       - Tab A: Zero-Shot Voice Cloning & Long Reference Audio
       - Tab B: Text-based Voice Design
       - Tab C: Style-Controlled Voice Cloning
       - Tab D: Batch Generation (Multi-Line / TXT / CSV -> ZIP)
  4. Reference Voice Preset Manager (`./presets/`) synced across all tabs
  5. Audio Output, DSP Post-Processing (Speed, Silence Trim, 24k/44.1k/48k WAV)
     & Persistent Generation History (last 15 clips in `./outputs/`)
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr

from text_processor import (
    DIALECT_GUIDANCE,
    GEMINI_DEFAULT_MODELS,
    IRODORI_EMOJI_TAGS,
    JLPT_GUIDANCE,
    OPENAI_DEFAULT_MODELS,
    ScriptGenerationError,
    analyze_japanese_text,
    generate_japanese_script,
)
from tts_engine import (
    EXPORT_SAMPLE_RATES,
    HF_REPO_ID,
    QUANTIZATION_VARIANTS,
    IrodoriTTSEngine,
    TTSEngineError,
)

# ---------------------------------------------------------------------------
# Global Engine Singleton
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
ENGINE = IrodoriTTSEngine(base_dir=BASE_DIR)

# Quick-start voice design & style caption templates
VOICE_DESIGN_TEMPLATES: Dict[str, str] = {
    "(Select a voice template...)": "",
    "Calm Young Female Narrator": "落ち着いた優しい声の若い女性。クリアな発音で、丁寧かつ自然に話している。",
    "Energetic Anime Girl": "明るく元気な若い女性の声。少し高めのピッチで、楽しそうに弾むような話し方。",
    "Deep Calm Male Announcer": "低音で落ち着いた大人の男性アナウンサー。明瞭な滑舌で信頼感のあるトーン。",
    "Gentle Close-Mic Whisper": "耳元で囁くような、柔らかく親密な距離感の女性話者。息遣いが感じられる静かな声。",
    "Friendly Kansai Casual Male": "気さくで親しみやすい20代男性の声。リラックスしたテンポで会話するように話している。",
    "Warm News / Podcast Host": "聞き取りやすく温かみのある30代女性パーソナリティ。自然で安定したテンポ。",
}

STYLE_OVERRIDE_TEMPLATES: Dict[str, str] = {
    "(Select a style override...)": "",
    "Cheerful & Smiling": "嬉しそうに笑いながら、明るく弾むトーンで話している。",
    "Soft & Whispering": "声を落として、親密に耳元で優しく囁くように話している。",
    "Sad & Sorrowful": "少し声を震わせながら、悲しげで寂しそうにゆっくり話している。",
    "Excited & Fast-Paced": "ワクワクして興奮した様子で、少し早口かつ勢いよく話している。",
    "Formal & Composed": "礼儀正しく落ち着いたフォーマルな口調で、丁寧に明瞭に話している。",
    "Sleepy & Relaxed": "眠そうで気だるげな、力の抜けたリラックスしたテンポで話している。",
}

QUICK_EMOJIS_PRIMARY = [
    ("😊 Happy", "😊"),
    ("😆 Joyful", "😆"),
    ("😢 Sad", "😢"),
    ("😭 Sobbing", "😭"),
    ("😠 Angry", "😠"),
    ("😒 Annoyed", "😒"),
    ("😲 Surprised", "😲"),
    ("😱 Shout", "😱"),
    ("👂 Whisper", "👂"),
    ("🤭 Laugh", "🤭"),
    ("😮‍💨 Sigh", "😮‍💨"),
    ("🫶 Gentle", "🫶"),
    ("😏 Playful", "😏"),
    ("🥺 Timid", "🥺"),
    ("⏸️ Pause", "⏸️"),
    ("⏩ Fast", "⏩"),
    ("🐢 Slow", "🐢"),
]


# ---------------------------------------------------------------------------
# Custom Studio CSS
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
.studio-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 60%, #172554 100%);
    border: 1px solid rgba(148, 163, 184, 0.2);
    border-radius: 14px;
    padding: 18px 22px;
    margin-bottom: 10px;
    color: #f8fafc;
}
.studio-header h1 {
    margin: 0 0 4px 0;
    font-size: 1.55rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: #f8fafc;
}
.studio-header p {
    margin: 0;
    font-size: 0.92rem;
    color: #cbd5e1;
}
.vram-card {
    background: #0f172a;
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 12px 16px;
    color: #e2e8f0;
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}
.vram-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 12px;
    flex-wrap: wrap;
    margin-bottom: 8px;
    font-size: 0.86rem;
}
.vram-badge {
    display: inline-block;
    padding: 2px 9px;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
}
.vram-badge.loaded { background: #065f46; color: #a7f3d0; }
.vram-badge.unloaded { background: #334155; color: #cbd5e1; }
.vram-bar-bg {
    width: 100%;
    height: 11px;
    background: #1e293b;
    border-radius: 999px;
    overflow: hidden;
    border: 1px solid #334155;
}
.vram-bar-fill {
    height: 100%;
    border-radius: 999px;
    transition: width 0.3s ease;
}
.ruby-preview-box {
    background: #0f172a;
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 14px 16px;
    min-height: 68px;
    color: #f1f5f9;
    line-height: 2.35;
    font-size: 1.12rem;
}
.ruby-preview-box.empty {
    color: #94a3b8;
    font-size: 0.92rem;
    line-height: 1.5;
}
.ruby-preview-box ruby rt {
    font-size: 0.65em;
    color: #38bdf8;
    font-weight: 500;
}
.emoji-btn button {
    min-width: 0 !important;
    padding: 5px 9px !important;
    font-size: 0.82rem !important;
}
"""


# ---------------------------------------------------------------------------
# UI Helper Functions
# ---------------------------------------------------------------------------
def render_vram_html() -> str:
    stats = ENGINE.get_vram_stats()
    pct = stats.utilization_pct
    if pct < 60:
        bar_color = "linear-gradient(90deg, #10b981, #34d399)"
    elif pct < 85:
        bar_color = "linear-gradient(90deg, #f59e0b, #fbbf24)"
    else:
        bar_color = "linear-gradient(90deg, #ef4444, #f87171)"

    badge_cls = "loaded" if stats.model_loaded else "unloaded"
    badge_text = (
        f"LOADED: {html.escape(str(stats.active_variant))} (Codec: {stats.codec_device.upper()})"
        if stats.model_loaded
        else "UNLOADED (VRAM FREE)"
    )

    return f"""
    <div class="vram-card">
      <div class="vram-row">
        <div>
          <strong>GPU:</strong> {html.escape(stats.gpu_name)}
          <span style="color:#94a3b8;">(CC {html.escape(stats.compute_capability)})</span>
        </div>
        <span class="vram-badge {badge_cls}">{badge_text}</span>
      </div>
      <div class="vram-row" style="margin-bottom:6px;">
        <div>
          <strong>PyTorch VRAM:</strong> {stats.allocated_mb:.0f} MB
          <span style="color:#94a3b8;">| Free GPU: {stats.free_mb:.0f} MB / {stats.total_mb:.0f} MB</span>
        </div>
        <div><strong>{pct:.1f}% used</strong></div>
      </div>
      <div class="vram-bar-bg">
        <div class="vram-bar-fill" style="width:{pct:.1f}%; background:{bar_color};"></div>
      </div>
    </div>
    """


def get_preset_choices() -> List[str]:
    return ["(None)"] + ENGINE.presets.list_presets()


def resolve_variant_key(dropdown_value: str) -> str:
    """Extract canonical variant key from dropdown choice."""
    for key in QUANTIZATION_VARIANTS:
        if dropdown_value == key or dropdown_value.startswith(key):
            return key
    return "float8-weight-only"


def get_variant_info_md(variant_key: str) -> str:
    key = resolve_variant_key(variant_key)
    info = QUANTIZATION_VARIANTS[key]
    return (
        f"**Selected Checkpoint:** `{HF_REPO_ID}/{key}`  |  "
        f"**Est. VRAM:** ~{info['est_vram_gb']:.1f} GB  |  "
        f"**Compute Dtype:** `bfloat16`  \n"
        f"{info['description']}"
    )


def handle_load_model(variant_key: str, codec_device: str) -> Tuple[str, str]:
    key = resolve_variant_key(variant_key)
    try:
        msg = ENGINE.load_model(variant=key, codec_device=codec_device)
        gr.Info(msg)
        return f"**Status:** {msg}", render_vram_html()
    except TTSEngineError as exc:
        gr.Warning(str(exc))
        return f"**Warning:** {exc}", render_vram_html()
    except Exception as exc:
        gr.Error(f"Unexpected error while loading model: {exc}")
        return f"**Error:** {exc}", render_vram_html()


def handle_unload_model() -> Tuple[str, str]:
    try:
        msg = ENGINE.unload_model()
        gr.Info(msg)
        return f"**Status:** {msg}", render_vram_html()
    except Exception as exc:
        gr.Error(f"Failed to unload model: {exc}")
        return f"**Error:** {exc}", render_vram_html()


def handle_analyze_text(text: str) -> Tuple[str, str, str, str]:
    res = analyze_japanese_text(text)
    return res.ruby_html, res.hiragana, res.romaji, res.token_notes


def handle_append_emoji(current_text: str, emoji: str) -> str:
    base = current_text or ""
    return f"{base}{emoji}"


def handle_provider_change(provider: str) -> Tuple[Any, Any]:
    if provider == "Google Gemini":
        return (
            gr.update(value=GEMINI_DEFAULT_MODELS[0], placeholder="e.g., gemini-4-hydrogen, gemini-4-argon"),
            gr.update(visible=False, value=""),
        )
    return (
        gr.update(value=OPENAI_DEFAULT_MODELS[0], placeholder="e.g., gpt-4o-mini, llama3.1"),
        gr.update(visible=True),
    )


def handle_generate_script(
    topic: str,
    jlpt_level: str,
    dialect: str,
    provider: str,
    api_key_input: str,
    stored_api_key: str,
    model_name: str,
    base_url: str,
    script_format: str,
    num_lines: int,
    include_emojis: bool,
) -> Tuple[str, str, str, str, str, str]:
    # Store API key strictly in session state (never written to disk)
    effective_key = (api_key_input or "").strip() or (stored_api_key or "").strip()
    try:
        generated = generate_japanese_script(
            topic=topic,
            jlpt_level=jlpt_level,
            dialect=dialect,
            provider=provider,
            api_key=effective_key,
            model_name=model_name,
            base_url=base_url,
            script_format=script_format,
            num_lines=int(num_lines),
            include_emojis=include_emojis,
        )
        analysis = analyze_japanese_text(generated)
        gr.Info("Japanese script generated successfully!")
        return (
            generated,
            effective_key,
            analysis.ruby_html,
            analysis.hiragana,
            analysis.romaji,
            analysis.token_notes,
        )
    except ScriptGenerationError as exc:
        gr.Warning(str(exc))
        return gr.update(), effective_key, gr.update(), gr.update(), gr.update(), gr.update()
    except Exception as exc:
        gr.Error(f"Script generation failed: {exc}")
        return gr.update(), effective_key, gr.update(), gr.update(), gr.update(), gr.update()


# ---------------------------------------------------------------------------
# Voice Preset Callbacks
# ---------------------------------------------------------------------------
def _all_preset_dropdown_updates(selected: Optional[str] = None) -> Tuple[Any, Any, Any, Any, List[List[Any]]]:
    choices = get_preset_choices()
    val = selected if (selected and selected in choices) else "(None)"
    return (
        gr.update(choices=choices, value=val),
        gr.update(choices=choices, value=val),
        gr.update(choices=choices, value=val),
        gr.update(choices=choices, value=val),
        ENGINE.presets.list_preset_rows(),
    )


def handle_save_preset(
    preset_name: str,
    audio_path: Optional[str],
    transcript: str,
) -> Tuple[str, Any, Any, Any, Any, List[List[Any]]]:
    try:
        meta = ENGINE.presets.save_preset(
            preset_name=preset_name,
            source_audio_path=audio_path or "",
            transcript=transcript,
        )
        msg = f"Saved preset `{meta['name']}` ({meta['duration_sec']:.2f}s) to `./presets/`."
        gr.Info(msg)
        u1, u2, u3, u4, rows = _all_preset_dropdown_updates(selected=meta["name"])
        return msg, u1, u2, u3, u4, rows
    except TTSEngineError as exc:
        gr.Warning(str(exc))
        u1, u2, u3, u4, rows = _all_preset_dropdown_updates()
        return f"Warning: {exc}", u1, u2, u3, u4, rows
    except Exception as exc:
        gr.Error(f"Failed to save preset: {exc}")
        u1, u2, u3, u4, rows = _all_preset_dropdown_updates()
        return f"Error: {exc}", u1, u2, u3, u4, rows


def handle_delete_preset(
    preset_name: str,
) -> Tuple[str, Any, Any, Any, Any, List[List[Any]], Optional[str]]:
    if not preset_name or preset_name == "(None)":
        gr.Warning("Select a preset to delete first.")
        u1, u2, u3, u4, rows = _all_preset_dropdown_updates()
        return "No preset selected.", u1, u2, u3, u4, rows, None
    removed = ENGINE.presets.delete_preset(preset_name)
    if removed:
        gr.Info(f"Deleted preset `{preset_name}`.")
        msg = f"Deleted preset `{preset_name}`."
    else:
        gr.Warning(f"Preset `{preset_name}` was not found.")
        msg = f"Preset `{preset_name}` not found."
    u1, u2, u3, u4, rows = _all_preset_dropdown_updates(selected="(None)")
    return msg, u1, u2, u3, u4, rows, None


def handle_load_preset_into_tab(preset_name: str) -> Tuple[Optional[str], str]:
    info = ENGINE.presets.get_preset_info(preset_name)
    if not info:
        return None, ""
    return info["wav_path"], str(info.get("transcript", ""))


def handle_preview_preset(preset_name: str) -> Tuple[Optional[str], str]:
    info = ENGINE.presets.get_preset_info(preset_name)
    if not info:
        return None, "Select a saved preset to preview."
    return (
        info["wav_path"],
        f"**Preset:** `{info['name']}`  |  **Duration:** {info.get('duration_sec', 0):.2f}s  |  "
        f"**Transcript:** {info.get('transcript') or '(none)'}",
    )


# ---------------------------------------------------------------------------
# Synthesis Callbacks (Tabs A, B, C, D)
# ---------------------------------------------------------------------------
def _resolve_export_sr(label: str) -> int:
    return EXPORT_SAMPLE_RATES.get(label, 48000)


def handle_zero_shot_clone(
    text: str,
    ref_audio: Optional[str],
    preset_name: str,
    ref_transcript: str,
    max_ref_seconds: float,
    trim_ref_silence: bool,
    num_steps: int,
    cfg_text: float,
    cfg_speaker: float,
    seed: int,
    speed: float,
    trim_output_silence: bool,
    export_sr_label: str,
) -> Tuple[Optional[str], Optional[str], str, List[List[Any]], str]:
    effective_ref = ref_audio
    if not effective_ref and preset_name and preset_name != "(None)":
        info = ENGINE.presets.get_preset_info(preset_name)
        if info:
            effective_ref = info["wav_path"]

    try:
        res = ENGINE.synthesize(
            text=text,
            mode="zero_shot",
            ref_wav_path=effective_ref,
            ref_transcript=ref_transcript,
            max_ref_seconds=max_ref_seconds,
            trim_ref_silence=trim_ref_silence,
            num_steps=int(num_steps),
            cfg_scale_text=float(cfg_text),
            cfg_scale_speaker=float(cfg_speaker),
            seed=int(seed),
            speed=float(speed),
            trim_output_silence=bool(trim_output_silence),
            export_sample_rate=_resolve_export_sr(export_sr_label),
        )
        status = (
            f"**Done (Zero-Shot Cloning)** | File: `{res.output_path.name}` | "
            f"Duration: `{res.duration_sec:.2f}s` | SR: `{res.sample_rate} Hz` | "
            f"Seed: `{res.seed_used}` | Time: `{res.elapsed_sec:.2f}s`"
        )
        gr.Info(f"Generated {res.output_path.name} ({res.duration_sec:.2f}s)")
        return (
            str(res.output_path),
            str(res.output_path),
            status,
            ENGINE.history.get_table_rows(),
            render_vram_html(),
        )
    except TTSEngineError as exc:
        gr.Warning(str(exc))
        return None, None, f"**Warning:** {exc}", ENGINE.history.get_table_rows(), render_vram_html()
    except Exception as exc:
        gr.Error(f"Unexpected synthesis error: {exc}")
        return None, None, f"**Error:** {exc}", ENGINE.history.get_table_rows(), render_vram_html()


def handle_voice_design(
    text: str,
    voice_caption: str,
    num_steps: int,
    cfg_text: float,
    cfg_caption: float,
    seed: int,
    speed: float,
    trim_output_silence: bool,
    export_sr_label: str,
) -> Tuple[Optional[str], Optional[str], str, List[List[Any]], str]:
    try:
        res = ENGINE.synthesize(
            text=text,
            mode="voice_design",
            caption=voice_caption,
            num_steps=int(num_steps),
            cfg_scale_text=float(cfg_text),
            cfg_scale_caption=float(cfg_caption),
            seed=int(seed),
            speed=float(speed),
            trim_output_silence=bool(trim_output_silence),
            export_sample_rate=_resolve_export_sr(export_sr_label),
        )
        status = (
            f"**Done (Voice Design)** | File: `{res.output_path.name}` | "
            f"Duration: `{res.duration_sec:.2f}s` | SR: `{res.sample_rate} Hz` | "
            f"Seed: `{res.seed_used}` | Time: `{res.elapsed_sec:.2f}s`"
        )
        gr.Info(f"Generated {res.output_path.name} ({res.duration_sec:.2f}s)")
        return (
            str(res.output_path),
            str(res.output_path),
            status,
            ENGINE.history.get_table_rows(),
            render_vram_html(),
        )
    except TTSEngineError as exc:
        gr.Warning(str(exc))
        return None, None, f"**Warning:** {exc}", ENGINE.history.get_table_rows(), render_vram_html()
    except Exception as exc:
        gr.Error(f"Unexpected synthesis error: {exc}")
        return None, None, f"**Error:** {exc}", ENGINE.history.get_table_rows(), render_vram_html()


def handle_style_clone(
    text: str,
    ref_audio: Optional[str],
    preset_name: str,
    style_caption: str,
    max_ref_seconds: float,
    trim_ref_silence: bool,
    num_steps: int,
    cfg_text: float,
    cfg_speaker: float,
    cfg_caption: float,
    seed: int,
    speed: float,
    trim_output_silence: bool,
    export_sr_label: str,
) -> Tuple[Optional[str], Optional[str], str, List[List[Any]], str]:
    effective_ref = ref_audio
    if not effective_ref and preset_name and preset_name != "(None)":
        info = ENGINE.presets.get_preset_info(preset_name)
        if info:
            effective_ref = info["wav_path"]

    try:
        res = ENGINE.synthesize(
            text=text,
            mode="style_clone",
            ref_wav_path=effective_ref,
            caption=style_caption,
            max_ref_seconds=max_ref_seconds,
            trim_ref_silence=trim_ref_silence,
            num_steps=int(num_steps),
            cfg_scale_text=float(cfg_text),
            cfg_scale_speaker=float(cfg_speaker),
            cfg_scale_caption=float(cfg_caption),
            seed=int(seed),
            speed=float(speed),
            trim_output_silence=bool(trim_output_silence),
            export_sample_rate=_resolve_export_sr(export_sr_label),
        )
        status = (
            f"**Done (Style-Controlled Cloning)** | File: `{res.output_path.name}` | "
            f"Duration: `{res.duration_sec:.2f}s` | SR: `{res.sample_rate} Hz` | "
            f"Seed: `{res.seed_used}` | Time: `{res.elapsed_sec:.2f}s`"
        )
        gr.Info(f"Generated {res.output_path.name} ({res.duration_sec:.2f}s)")
        return (
            str(res.output_path),
            str(res.output_path),
            status,
            ENGINE.history.get_table_rows(),
            render_vram_html(),
        )
    except TTSEngineError as exc:
        gr.Warning(str(exc))
        return None, None, f"**Warning:** {exc}", ENGINE.history.get_table_rows(), render_vram_html()
    except Exception as exc:
        gr.Error(f"Unexpected synthesis error: {exc}")
        return None, None, f"**Error:** {exc}", ENGINE.history.get_table_rows(), render_vram_html()


def handle_batch_generate(
    multiline_text: str,
    uploaded_file: Optional[str],
    batch_voice_mode: str,
    preset_name: str,
    caption: str,
    max_ref_seconds: float,
    num_steps: int,
    cfg_text: float,
    cfg_speaker: float,
    cfg_caption: float,
    seed: int,
    speed: float,
    trim_output_silence: bool,
    export_sr_label: str,
    progress: gr.Progress = gr.Progress(),
) -> Tuple[Optional[str], Optional[str], List[List[Any]], str, List[List[Any]], str]:
    mode_map = {
        "Zero-Shot Voice Preset": "zero_shot",
        "Text-based Voice Design": "voice_design",
        "Style-Controlled Voice Preset": "style_clone",
    }
    engine_mode = mode_map.get(batch_voice_mode, "zero_shot")

    ref_wav_path: Optional[str] = None
    ref_transcript = ""
    if engine_mode in ("zero_shot", "style_clone"):
        info = ENGINE.presets.get_preset_info(preset_name)
        if not info:
            msg = "Select a saved Voice Preset in Batch Settings (or switch Batch Mode to Voice Design)."
            gr.Warning(msg)
            return None, None, [], f"**Warning:** {msg}", ENGINE.history.get_table_rows(), render_vram_html()
        ref_wav_path = info["wav_path"]
        ref_transcript = str(info.get("transcript", ""))

    try:
        items = ENGINE.parse_batch_inputs(
            multiline_text=multiline_text,
            uploaded_file_path=uploaded_file,
            default_caption=(caption or "").strip(),
        )
        zip_path, results = ENGINE.synthesize_batch(
            items=items,
            batch_mode=engine_mode,
            ref_wav_path=ref_wav_path,
            ref_transcript=ref_transcript,
            max_ref_seconds=max_ref_seconds,
            num_steps=int(num_steps),
            cfg_scale_text=float(cfg_text),
            cfg_scale_speaker=float(cfg_speaker),
            cfg_scale_caption=float(cfg_caption),
            seed=int(seed),
            speed=float(speed),
            trim_output_silence=bool(trim_output_silence),
            export_sample_rate=_resolve_export_sr(export_sr_label),
            progress_cb=lambda frac, desc: progress(frac, desc=desc),
        )

        batch_rows = [
            [
                idx,
                r.output_path.name,
                f"{r.duration_sec:.2f}s",
                f"{r.elapsed_sec:.2f}s",
                r.text[:64] + ("…" if len(r.text) > 64 else ""),
            ]
            for idx, r in enumerate(results, start=1)
        ]
        first_preview = str(results[0].output_path) if results else None
        total_dur = sum(r.duration_sec for r in results)
        status = (
            f"**Batch Complete!** Synthesized `{len(results)}` clips "
            f"(total audio `{total_dur:.1f}s`). ZIP ready: `{zip_path.name}`."
        )
        gr.Info(f"Batch complete: {len(results)} WAV files bundled into {zip_path.name}")
        return (
            str(zip_path),
            first_preview,
            batch_rows,
            status,
            ENGINE.history.get_table_rows(),
            render_vram_html(),
        )
    except TTSEngineError as exc:
        gr.Warning(str(exc))
        return None, None, [], f"**Warning:** {exc}", ENGINE.history.get_table_rows(), render_vram_html()
    except Exception as exc:
        gr.Error(f"Batch generation failed: {exc}")
        return None, None, [], f"**Error:** {exc}", ENGINE.history.get_table_rows(), render_vram_html()


def handle_history_select(evt: gr.SelectData) -> Tuple[Optional[str], Optional[str], str]:
    row_idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else int(evt.index)
    path = ENGINE.history.get_path_by_row(row_idx)
    if not path:
        gr.Warning("Selected audio file no longer exists on disk.")
        return None, None, "Audio file missing."
    entries = ENGINE.history.get_entries()
    entry = entries[row_idx]
    info = (
        f"**Replaying History #{row_idx + 1}** | `{entry['filename']}` | "
        f"Mode: `{entry['mode']}` | Duration: `{entry['duration_sec']}s` | "
        f"SR: `{entry['sample_rate']} Hz` | Seed: `{entry['seed']}`"
    )
    return path, path, info


# ---------------------------------------------------------------------------
# Reusable UI Block Helpers
# ---------------------------------------------------------------------------
def build_emoji_toolbar(target_textbox: gr.Textbox) -> None:
    """Render a compact one-click emoji emotion/prosody inserter bound to `target_textbox`."""
    with gr.Accordion("One-Click Emoji Emotion & Prosody Inserter (Irodori-TTS Native Tags)", open=False):
        with gr.Row():
            for label, emoji in QUICK_EMOJIS_PRIMARY[:9]:
                btn = gr.Button(label, size="sm", elem_classes=["emoji-btn"])
                btn.click(
                    fn=lambda txt, em=emoji: handle_append_emoji(txt, em),
                    inputs=[target_textbox],
                    outputs=[target_textbox],
                )
        with gr.Row():
            for label, emoji in QUICK_EMOJIS_PRIMARY[9:]:
                btn = gr.Button(label, size="sm", elem_classes=["emoji-btn"])
                btn.click(
                    fn=lambda txt, em=emoji: handle_append_emoji(txt, em),
                    inputs=[target_textbox],
                    outputs=[target_textbox],
                )


def build_postproc_controls() -> Tuple[gr.Slider, gr.Checkbox, gr.Dropdown]:
    with gr.Row():
        speed_slider = gr.Slider(
            label="Output Speed (Pitch-Preserving)",
            minimum=0.6,
            maximum=1.6,
            step=0.05,
            value=1.0,
            info="1.0 = natural pace (WSOLA time-stretch)",
        )
        trim_chk = gr.Checkbox(
            label="Trim Leading/Trailing Silence",
            value=True,
        )
        export_sr_dd = gr.Dropdown(
            label="Export WAV Sample Rate",
            choices=list(EXPORT_SAMPLE_RATES.keys()),
            value="48 kHz (Native Studio WAV)",
        )
    return speed_slider, trim_chk, export_sr_dd


# ---------------------------------------------------------------------------
# Gradio Blocks Application Construction
# ---------------------------------------------------------------------------
def create_app() -> Tuple[gr.Blocks, Any, str]:
    theme = gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="sky",
        neutral_hue="slate",
    )

    gradio_major = int(gr.__version__.split(".")[0])
    blocks_kwargs: Dict[str, Any] = {"title": "Irodori-TTS v4-Large Japanese Studio"}
    if gradio_major < 6:
        blocks_kwargs["theme"] = theme
        blocks_kwargs["css"] = CUSTOM_CSS

    with gr.Blocks(**blocks_kwargs) as demo:
        # Session-only state for API keys (never persisted or hardcoded)
        api_key_state = gr.State("")

        # Header Banner
        gr.HTML(
            """
            <div class="studio-header">
              <h1>🎙️ Irodori-TTS v4-Large Japanese Studio</h1>
              <p>
                Local CUDA 12.x Production Suite for <code>Aratako/Irodori-TTS-v4-Large-Quantized</code>
                (torchao BF16/FP8/INT4) • Zero-Shot Cloning • Voice Design • Style Control • Batch Export • AI Script Writer
              </p>
            </div>
            """
        )

        # =====================================================================
        # 1. Model & VRAM Management (Header / Top Accordion)
        # =====================================================================
        with gr.Accordion("⚙️ Model & VRAM Management (RTX 4060 8GB CUDA Control)", open=True):
            with gr.Row(equal_height=False):
                with gr.Column(scale=3):
                    variant_choices = [
                        (meta["label"], key) for key, meta in QUANTIZATION_VARIANTS.items()
                    ]
                    with gr.Row():
                        variant_dd = gr.Dropdown(
                            label="Quantization Variant (torchao)",
                            choices=variant_choices,
                            value="float8-weight-only",
                            interactive=True,
                            scale=3,
                        )
                        codec_device_dd = gr.Dropdown(
                            label="DACVAE Codec Device",
                            choices=[
                                ("CPU (Recommended for 8GB VRAM — Saves ~1 GB)", "cpu"),
                                ("CUDA (Slightly faster decode — Needs >6GB free)", "cuda"),
                            ],
                            value="cpu",
                            interactive=True,
                            scale=2,
                        )
                    variant_info_md = gr.Markdown(get_variant_info_md("float8-weight-only"))
                    with gr.Row():
                        load_btn = gr.Button("🚀 Load Model to GPU", variant="primary")
                        unload_btn = gr.Button("🧹 Unload Model (Free VRAM)", variant="stop")
                        refresh_vram_btn = gr.Button("🔄 Refresh VRAM", variant="secondary")
                    model_status_md = gr.Markdown(
                        "**Status:** Model not loaded yet. Choose a quantization variant and click **Load Model to GPU**."
                    )
                with gr.Column(scale=2):
                    vram_html = gr.HTML(render_vram_html())

            variant_dd.change(fn=get_variant_info_md, inputs=[variant_dd], outputs=[variant_info_md])
            load_btn.click(
                fn=handle_load_model,
                inputs=[variant_dd, codec_device_dd],
                outputs=[model_status_md, vram_html],
            )
            unload_btn.click(fn=handle_unload_model, inputs=[], outputs=[model_status_md, vram_html])
            refresh_vram_btn.click(fn=render_vram_html, inputs=[], outputs=[vram_html])

            # Periodic lightweight VRAM telemetry refresh (every 5 seconds)
            vram_timer = gr.Timer(value=5.0)
            vram_timer.tick(fn=render_vram_html, inputs=[], outputs=[vram_html])

        # =====================================================================
        # 2. Main Studio Tabs (AI Script Generator + All 4 TTS Modes + Presets)
        # =====================================================================
        with gr.Tabs() as main_tabs:
            # -----------------------------------------------------------------
            # TAB 1: AI Japanese Text & Dialogue Generator + Reading Analyzer
            # -----------------------------------------------------------------
            with gr.Tab("✍️ 1. AI Script Generator & Reading Helper"):
                with gr.Row():
                    with gr.Column(scale=2):
                        ai_topic = gr.Textbox(
                            label="Topic / Scenario Prompt",
                            placeholder="e.g., A friendly barista welcoming a tired customer on a rainy Tokyo evening...",
                            lines=3,
                        )
                        with gr.Row():
                            ai_jlpt = gr.Dropdown(
                                label="JLPT Level",
                                choices=list(JLPT_GUIDANCE.keys()),
                                value="Natural Native",
                            )
                            ai_dialect = gr.Dropdown(
                                label="Dialect / Register",
                                choices=list(DIALECT_GUIDANCE.keys()),
                                value="Standard Tokyo (Hyojungo)",
                            )
                        with gr.Row():
                            ai_format = gr.Dropdown(
                                label="Script Format",
                                choices=[
                                    "Single Monologue",
                                    "Multi-Line (Batch Ready)",
                                    "Dialogue (Unlabeled Lines)",
                                ],
                                value="Single Monologue",
                            )
                            ai_num_lines = gr.Slider(
                                label="Target Sentences / Lines",
                                minimum=1,
                                maximum=15,
                                step=1,
                                value=4,
                            )
                            ai_emojis_chk = gr.Checkbox(
                                label="Include Irodori Emojis",
                                value=True,
                            )

                        with gr.Accordion("🔑 LLM Provider & API Key (Session State Only)", open=True):
                            with gr.Row():
                                ai_provider = gr.Dropdown(
                                    label="API Provider",
                                    choices=["Google Gemini", "OpenAI / OpenAI-Compatible"],
                                    value="Google Gemini",
                                )
                                ai_model = gr.Textbox(
                                    label="Model ID",
                                    value=GEMINI_DEFAULT_MODELS[0],
                                    placeholder="e.g., gemini-4-hydrogen",
                                )
                            ai_base_url = gr.Textbox(
                                label="Custom Base URL (Optional — OpenRouter / Ollama / LM Studio)",
                                placeholder="e.g., http://localhost:11434/v1",
                                visible=False,
                            )
                            ai_api_key = gr.Textbox(
                                label="API Key (stored in RAM session only, never saved to disk)",
                                type="password",
                                placeholder="Enter API key or set GEMINI_API_KEY / OPENAI_API_KEY in .env",
                            )

                        gen_script_btn = gr.Button("✨ Generate Japanese Script", variant="primary")

                    with gr.Column(scale=3):
                        script_text = gr.Textbox(
                            label="Japanese Script Editor (Edit freely or type directly)",
                            placeholder="こんにちは！今日もいい天気ですね。😆",
                            lines=6,
                        )
                        build_emoji_toolbar(script_text)

                        with gr.Row():
                            send_to_tab_a = gr.Button("➡️ Send to Tab A (Zero-Shot)", size="sm")
                            send_to_tab_b = gr.Button("➡️ Send to Tab B (Voice Design)", size="sm")
                            send_to_tab_c = gr.Button("➡️ Send to Tab C (Style Clone)", size="sm")
                            send_to_tab_d = gr.Button("➡️ Send to Tab D (Batch)", size="sm")

                        gr.Markdown("### 🔎 Live Furigana (Hiragana) & Romaji Reading Check")
                        script_ruby_html = gr.HTML(analyze_japanese_text("").ruby_html)
                        with gr.Row():
                            script_hira = gr.Textbox(
                                label="Full Hiragana Reading",
                                lines=2,
                                interactive=False,
                            )
                            script_romaji = gr.Textbox(
                                label="Hepburn Romaji",
                                lines=2,
                                interactive=False,
                            )
                        script_kanji_notes = gr.Textbox(
                            label="Kanji Reading Verification Glossary",
                            lines=1,
                            interactive=False,
                        )

                with gr.Accordion("📖 Full Irodori-TTS v4 Emoji Reference Table", open=False):
                    emoji_table_rows = [
                        [tag.emoji, tag.label, tag.category, tag.description]
                        for tag in IRODORI_EMOJI_TAGS
                    ]
                    gr.Dataframe(
                        headers=["Emoji", "Label", "Category", "Acoustic Effect"],
                        value=emoji_table_rows,
                        interactive=False,
                    )

            # -----------------------------------------------------------------
            # TAB A: Zero-Shot Voice Cloning & Long Reference Audio
            # -----------------------------------------------------------------
            with gr.Tab("🎤 Tab A: Zero-Shot Voice Cloning"):
                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown("### 1. Reference Voice Input (Short or Long Audio)")
                        with gr.Row():
                            tab_a_preset_dd = gr.Dropdown(
                                label="Load Saved Voice Preset",
                                choices=get_preset_choices(),
                                value="(None)",
                            )
                        tab_a_ref_audio = gr.Audio(
                            label="Reference Audio (Upload .wav/.mp3/.flac or Record Microphone)",
                            sources=["upload", "microphone"],
                            type="filepath",
                        )
                        tab_a_ref_transcript = gr.Textbox(
                            label="Reference Transcript (Optional — for alignment / preset notes)",
                            placeholder="Optional transcript of what is spoken in the reference clip...",
                            lines=2,
                        )
                        with gr.Row():
                            tab_a_max_ref_sec = gr.Slider(
                                label="Long Reference Audio Max Duration (sec)",
                                minimum=5.0,
                                maximum=120.0,
                                step=5.0,
                                value=30.0,
                                info="Irodori-TTS v4 supports up to 120s reference audio (auto-resampled to 48kHz).",
                            )
                            tab_a_trim_ref = gr.Checkbox(
                                label="Preprocess: Trim Ref Silence",
                                value=True,
                            )
                        with gr.Accordion("💾 Quick-Save Current Reference as Preset", open=False):
                            with gr.Row():
                                tab_a_new_preset_name = gr.Textbox(
                                    label="Preset Name",
                                    placeholder="e.g., Female_Formal",
                                )
                                tab_a_save_preset_btn = gr.Button("Save Preset", variant="secondary")
                            tab_a_preset_msg = gr.Markdown("")

                    with gr.Column(scale=3):
                        gr.Markdown("### 2. Target Japanese Text")
                        tab_a_text = gr.Textbox(
                            label="Japanese Text to Speak",
                            placeholder="こんにちは、私はイロドリティーティーエスです。自然な日本語でお話しします。😊",
                            lines=5,
                        )
                        build_emoji_toolbar(tab_a_text)
                        with gr.Accordion("🔎 Live Furigana & Romaji Preview", open=False):
                            tab_a_ruby = gr.HTML(analyze_japanese_text("").ruby_html)
                            with gr.Row():
                                tab_a_hira = gr.Textbox(label="Hiragana", lines=2, interactive=False)
                                tab_a_romaji = gr.Textbox(label="Romaji", lines=2, interactive=False)
                            tab_a_notes = gr.Textbox(label="Kanji Glossary", lines=1, interactive=False)

                        with gr.Accordion("🎛️ Sampling & Post-Processing Controls", open=True):
                            with gr.Row():
                                tab_a_steps = gr.Slider(label="Sampling Steps", minimum=10, maximum=80, step=2, value=40)
                                tab_a_cfg_text = gr.Slider(label="Text CFG", minimum=1.0, maximum=8.0, step=0.5, value=3.0)
                                tab_a_cfg_spk = gr.Slider(label="Speaker CFG", minimum=1.0, maximum=10.0, step=0.5, value=5.0)
                                tab_a_seed = gr.Number(label="Seed (-1 = Random)", value=-1, precision=0)
                            tab_a_speed, tab_a_trim_out, tab_a_sr = build_postproc_controls()

                        tab_a_gen_btn = gr.Button("🔊 Generate Cloned Speech", variant="primary", size="lg")

            # -----------------------------------------------------------------
            # TAB B: Text-based Voice Design
            # -----------------------------------------------------------------
            with gr.Tab("🎨 Tab B: Text-based Voice Design"):
                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown("### 1. Describe the Desired Voice (No Reference Audio Needed)")
                        tab_b_template_dd = gr.Dropdown(
                            label="Quick Voice Persona Template",
                            choices=list(VOICE_DESIGN_TEMPLATES.keys()),
                            value="(Select a voice template...)",
                        )
                        tab_b_caption = gr.Textbox(
                            label="Voice Description Prompt (Gender, Age, Pitch, Tone, Speaking Style)",
                            placeholder="例: 落ち着いた優しい声の若い女性。クリアな発音で、親しみやすく自然に話している。",
                            lines=5,
                        )
                        gr.Markdown(
                            "*Tip: Japanese natural-language voice descriptions yield the highest fidelity on Irodori-TTS v4-Large.*"
                        )

                    with gr.Column(scale=3):
                        gr.Markdown("### 2. Target Japanese Text")
                        tab_b_text = gr.Textbox(
                            label="Japanese Text to Speak",
                            placeholder="おはようございます！今日も素晴らしい一日にしましょうね。😆",
                            lines=5,
                        )
                        build_emoji_toolbar(tab_b_text)
                        with gr.Accordion("🔎 Live Furigana & Romaji Preview", open=False):
                            tab_b_ruby = gr.HTML(analyze_japanese_text("").ruby_html)
                            with gr.Row():
                                tab_b_hira = gr.Textbox(label="Hiragana", lines=2, interactive=False)
                                tab_b_romaji = gr.Textbox(label="Romaji", lines=2, interactive=False)
                            tab_b_notes = gr.Textbox(label="Kanji Glossary", lines=1, interactive=False)

                        with gr.Accordion("🎛️ Sampling & Post-Processing Controls", open=True):
                            with gr.Row():
                                tab_b_steps = gr.Slider(label="Sampling Steps", minimum=10, maximum=80, step=2, value=40)
                                tab_b_cfg_text = gr.Slider(label="Text CFG", minimum=1.0, maximum=8.0, step=0.5, value=3.0)
                                tab_b_cfg_cap = gr.Slider(label="Voice Design CFG", minimum=1.0, maximum=8.0, step=0.5, value=3.5)
                                tab_b_seed = gr.Number(label="Seed (-1 = Random)", value=-1, precision=0)
                            tab_b_speed, tab_b_trim_out, tab_b_sr = build_postproc_controls()

                        tab_b_gen_btn = gr.Button("🎨 Generate Designed Voice", variant="primary", size="lg")

            # -----------------------------------------------------------------
            # TAB C: Style-Controlled Voice Cloning
            # -----------------------------------------------------------------
            with gr.Tab("🎭 Tab C: Style-Controlled Voice Cloning"):
                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown("### 1. Voice Identity (Reference Audio) + Style Override")
                        tab_c_preset_dd = gr.Dropdown(
                            label="Load Saved Voice Preset",
                            choices=get_preset_choices(),
                            value="(None)",
                        )
                        tab_c_ref_audio = gr.Audio(
                            label="Reference Audio (Voice Identity)",
                            sources=["upload", "microphone"],
                            type="filepath",
                        )
                        tab_c_style_template_dd = gr.Dropdown(
                            label="Quick Style / Emotion Template",
                            choices=list(STYLE_OVERRIDE_TEMPLATES.keys()),
                            value="(Select a style override...)",
                        )
                        tab_c_caption = gr.Textbox(
                            label="Style / Emotion Override Prompt",
                            placeholder="例: 嬉しそうに笑いながら、明るく弾むトーンで話している。",
                            lines=3,
                        )
                        with gr.Row():
                            tab_c_max_ref_sec = gr.Slider(
                                label="Max Ref Duration (sec)",
                                minimum=5.0,
                                maximum=120.0,
                                step=5.0,
                                value=30.0,
                            )
                            tab_c_trim_ref = gr.Checkbox(label="Trim Ref Silence", value=True)

                    with gr.Column(scale=3):
                        gr.Markdown("### 2. Target Japanese Text")
                        tab_c_text = gr.Textbox(
                            label="Japanese Text to Speak",
                            placeholder="えっ、本当に！？それ、すっごく嬉しいんだけど！😆",
                            lines=5,
                        )
                        build_emoji_toolbar(tab_c_text)
                        with gr.Accordion("🔎 Live Furigana & Romaji Preview", open=False):
                            tab_c_ruby = gr.HTML(analyze_japanese_text("").ruby_html)
                            with gr.Row():
                                tab_c_hira = gr.Textbox(label="Hiragana", lines=2, interactive=False)
                                tab_c_romaji = gr.Textbox(label="Romaji", lines=2, interactive=False)
                            tab_c_notes = gr.Textbox(label="Kanji Glossary", lines=1, interactive=False)

                        with gr.Accordion("🎛️ Sampling & Post-Processing Controls", open=True):
                            with gr.Row():
                                tab_c_steps = gr.Slider(label="Sampling Steps", minimum=10, maximum=80, step=2, value=40)
                                tab_c_cfg_text = gr.Slider(label="Text CFG", minimum=1.0, maximum=8.0, step=0.5, value=3.0)
                                tab_c_cfg_spk = gr.Slider(label="Speaker CFG", minimum=1.0, maximum=10.0, step=0.5, value=4.5)
                                tab_c_cfg_cap = gr.Slider(label="Style CFG", minimum=1.0, maximum=8.0, step=0.5, value=3.5)
                                tab_c_seed = gr.Number(label="Seed (-1 = Random)", value=-1, precision=0)
                            tab_c_speed, tab_c_trim_out, tab_c_sr = build_postproc_controls()

                        tab_c_gen_btn = gr.Button("🎭 Generate Style-Controlled Clone", variant="primary", size="lg")

            # -----------------------------------------------------------------
            # TAB D: Batch Generation (Multi-Line / TXT / CSV -> ZIP)
            # -----------------------------------------------------------------
            with gr.Tab("📦 Tab D: Batch Generation (Multi-Line / CSV)"):
                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown("### 1. Batch Script Input")
                        tab_d_text = gr.Textbox(
                            label="Japanese Sentences (One sentence per line)",
                            placeholder="おはようございます。\n今日の予定を確認しましょう。\n準備ができたら出発します！😆",
                            lines=8,
                        )
                        tab_d_file = gr.File(
                            label="Or Upload .txt (1 line per clip) or .csv (columns: text, optional caption, filename)",
                            file_types=[".txt", ".csv"],
                            type="filepath",
                        )
                    with gr.Column(scale=2):
                        gr.Markdown("### 2. Voice & Batch Export Settings")
                        tab_d_mode = gr.Radio(
                            label="Batch Synthesis Mode",
                            choices=[
                                "Zero-Shot Voice Preset",
                                "Text-based Voice Design",
                                "Style-Controlled Voice Preset",
                            ],
                            value="Zero-Shot Voice Preset",
                        )
                        tab_d_preset_dd = gr.Dropdown(
                            label="Voice Preset (for Zero-Shot / Style-Controlled Batch)",
                            choices=get_preset_choices(),
                            value="(None)",
                        )
                        tab_d_caption = gr.Textbox(
                            label="Default Voice Design / Style Caption (overridden by CSV `caption` column if present)",
                            placeholder="落ち着いた優しい女性の声。",
                            lines=2,
                        )
                        with gr.Row():
                            tab_d_max_ref_sec = gr.Slider(label="Max Ref Sec", minimum=5.0, maximum=120.0, step=5.0, value=30.0)
                            tab_d_steps = gr.Slider(label="Steps", minimum=10, maximum=80, step=2, value=40)
                            tab_d_seed = gr.Number(label="Start Seed (-1=Rnd)", value=-1, precision=0)
                        with gr.Row():
                            tab_d_cfg_text = gr.Slider(label="Text CFG", minimum=1.0, maximum=8.0, step=0.5, value=3.0)
                            tab_d_cfg_spk = gr.Slider(label="Speaker CFG", minimum=1.0, maximum=10.0, step=0.5, value=5.0)
                            tab_d_cfg_cap = gr.Slider(label="Caption CFG", minimum=1.0, maximum=8.0, step=0.5, value=3.0)
                        tab_d_speed, tab_d_trim_out, tab_d_sr = build_postproc_controls()
                        tab_d_gen_btn = gr.Button("📦 Run Batch & Export ZIP", variant="primary", size="lg")

                tab_d_status = gr.Markdown("")
                with gr.Row():
                    tab_d_zip_out = gr.File(label="Download Batch Archive (.zip with WAVs + manifest.csv)")
                    tab_d_preview_audio = gr.Audio(label="First Clip Preview", type="filepath")
                tab_d_results_df = gr.Dataframe(
                    headers=["#", "Filename", "Duration", "Gen Time", "Text"],
                    interactive=False,
                    label="Batch Output Manifest",
                )

            # -----------------------------------------------------------------
            # TAB E: Reference Voice Preset Manager (`./presets/`)
            # -----------------------------------------------------------------
            with gr.Tab("🗂️ Voice Preset Manager"):
                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown("### Add / Overwrite a Reference Voice Preset")
                        pm_name = gr.Textbox(
                            label="Preset Name",
                            placeholder="e.g., Female_Formal, Male_Casual, Anime_Genki",
                        )
                        pm_audio = gr.Audio(
                            label="Upload or Record Reference Audio",
                            sources=["upload", "microphone"],
                            type="filepath",
                        )
                        pm_transcript = gr.Textbox(
                            label="Optional Reference Transcript / Notes",
                            placeholder="Optional spoken transcript or voice notes...",
                            lines=2,
                        )
                        pm_save_btn = gr.Button("💾 Save to ./presets/", variant="primary")
                        pm_status = gr.Markdown("")

                    with gr.Column(scale=3):
                        gr.Markdown("### Saved Presets (`./presets/`)")
                        pm_table = gr.Dataframe(
                            headers=["Preset Name", "Duration", "Transcript / Notes", "Created"],
                            value=ENGINE.presets.list_preset_rows(),
                            interactive=False,
                        )
                        with gr.Row():
                            pm_select_dd = gr.Dropdown(
                                label="Select Preset to Preview or Delete",
                                choices=get_preset_choices(),
                                value="(None)",
                            )
                            pm_delete_btn = gr.Button("🗑️ Delete Selected Preset", variant="stop")
                        pm_preview_md = gr.Markdown("Select a saved preset to preview.")
                        pm_preview_audio = gr.Audio(label="Preset Audio Preview", type="filepath")

        # =====================================================================
        # 3. Persistent Audio Output Player & Generation History (Last 15 Clips)
        # =====================================================================
        gr.Markdown("---")
        gr.Markdown("## 🔊 Studio Output Player & Generation History (`./outputs/`)")
        with gr.Row():
            with gr.Column(scale=2):
                output_status_md = gr.Markdown("Ready. Load a model above and click Generate.")
                output_audio = gr.Audio(
                    label="Generated Speech (Waveform Preview & Playback)",
                    type="filepath",
                    waveform_options=gr.WaveformOptions(show_recording_waveform=True),
                )
                output_download = gr.File(label="Download Output WAV")

            with gr.Column(scale=3):
                with gr.Row():
                    gr.Markdown("### 📜 Recent Generations (Last 15 Clips — Click Any Row to Replay & Download)")
                    refresh_history_btn = gr.Button("🔄 Refresh History", size="sm")
                history_df = gr.Dataframe(
                    headers=["#", "Timestamp", "Mode", "Text Preview", "Duration", "Sample Rate", "Filename"],
                    value=ENGINE.history.get_table_rows(),
                    interactive=False,
                )

        # =====================================================================
        # Wire All Events Cleanly
        # =====================================================================
        # AI Generator events
        ai_provider.change(
            fn=handle_provider_change,
            inputs=[ai_provider],
            outputs=[ai_model, ai_base_url],
        )
        gen_script_btn.click(
            fn=handle_generate_script,
            inputs=[
                ai_topic,
                ai_jlpt,
                ai_dialect,
                ai_provider,
                ai_api_key,
                api_key_state,
                ai_model,
                ai_base_url,
                ai_format,
                ai_num_lines,
                ai_emojis_chk,
            ],
            outputs=[
                script_text,
                api_key_state,
                script_ruby_html,
                script_hira,
                script_romaji,
                script_kanji_notes,
            ],
        )

        # Live Furigana & Romaji updates across all text boxes
        script_text.change(
            fn=handle_analyze_text,
            inputs=[script_text],
            outputs=[script_ruby_html, script_hira, script_romaji, script_kanji_notes],
        )
        tab_a_text.change(
            fn=handle_analyze_text,
            inputs=[tab_a_text],
            outputs=[tab_a_ruby, tab_a_hira, tab_a_romaji, tab_a_notes],
        )
        tab_b_text.change(
            fn=handle_analyze_text,
            inputs=[tab_b_text],
            outputs=[tab_b_ruby, tab_b_hira, tab_b_romaji, tab_b_notes],
        )
        tab_c_text.change(
            fn=handle_analyze_text,
            inputs=[tab_c_text],
            outputs=[tab_c_ruby, tab_c_hira, tab_c_romaji, tab_c_notes],
        )

        # Send generated script directly to any studio tab
        send_to_tab_a.click(
            fn=lambda t: (t, gr.Info("Sent script to Tab A (Zero-Shot Cloning)"))[0],
            inputs=[script_text],
            outputs=[tab_a_text],
        )
        send_to_tab_b.click(
            fn=lambda t: (t, gr.Info("Sent script to Tab B (Voice Design)"))[0],
            inputs=[script_text],
            outputs=[tab_b_text],
        )
        send_to_tab_c.click(
            fn=lambda t: (t, gr.Info("Sent script to Tab C (Style-Controlled Cloning)"))[0],
            inputs=[script_text],
            outputs=[tab_c_text],
        )
        send_to_tab_d.click(
            fn=lambda t: (t, gr.Info("Sent script to Tab D (Batch Generation)"))[0],
            inputs=[script_text],
            outputs=[tab_d_text],
        )

        # Templates
        tab_b_template_dd.change(
            fn=lambda k: VOICE_DESIGN_TEMPLATES.get(k, ""),
            inputs=[tab_b_template_dd],
            outputs=[tab_b_caption],
        )
        tab_c_style_template_dd.change(
            fn=lambda k: STYLE_OVERRIDE_TEMPLATES.get(k, ""),
            inputs=[tab_c_style_template_dd],
            outputs=[tab_c_caption],
        )

        # Voice Preset Manager sync across tabs
        all_preset_outputs = [tab_a_preset_dd, tab_c_preset_dd, tab_d_preset_dd, pm_select_dd, pm_table]
        pm_save_btn.click(
            fn=handle_save_preset,
            inputs=[pm_name, pm_audio, pm_transcript],
            outputs=[pm_status] + all_preset_outputs,
        )
        tab_a_save_preset_btn.click(
            fn=handle_save_preset,
            inputs=[tab_a_new_preset_name, tab_a_ref_audio, tab_a_ref_transcript],
            outputs=[tab_a_preset_msg] + all_preset_outputs,
        )
        pm_delete_btn.click(
            fn=handle_delete_preset,
            inputs=[pm_select_dd],
            outputs=[pm_status] + all_preset_outputs + [pm_preview_audio],
        )
        pm_select_dd.change(
            fn=handle_preview_preset,
            inputs=[pm_select_dd],
            outputs=[pm_preview_audio, pm_preview_md],
        )
        tab_a_preset_dd.change(
            fn=handle_load_preset_into_tab,
            inputs=[tab_a_preset_dd],
            outputs=[tab_a_ref_audio, tab_a_ref_transcript],
        )
        tab_c_preset_dd.change(
            fn=lambda name: handle_load_preset_into_tab(name)[0],
            inputs=[tab_c_preset_dd],
            outputs=[tab_c_ref_audio],
        )

        # Core Synthesis Triggers (Tabs A, B, C, D)
        tab_a_gen_btn.click(
            fn=handle_zero_shot_clone,
            inputs=[
                tab_a_text,
                tab_a_ref_audio,
                tab_a_preset_dd,
                tab_a_ref_transcript,
                tab_a_max_ref_sec,
                tab_a_trim_ref,
                tab_a_steps,
                tab_a_cfg_text,
                tab_a_cfg_spk,
                tab_a_seed,
                tab_a_speed,
                tab_a_trim_out,
                tab_a_sr,
            ],
            outputs=[output_audio, output_download, output_status_md, history_df, vram_html],
        )

        tab_b_gen_btn.click(
            fn=handle_voice_design,
            inputs=[
                tab_b_text,
                tab_b_caption,
                tab_b_steps,
                tab_b_cfg_text,
                tab_b_cfg_cap,
                tab_b_seed,
                tab_b_speed,
                tab_b_trim_out,
                tab_b_sr,
            ],
            outputs=[output_audio, output_download, output_status_md, history_df, vram_html],
        )

        tab_c_gen_btn.click(
            fn=handle_style_clone,
            inputs=[
                tab_c_text,
                tab_c_ref_audio,
                tab_c_preset_dd,
                tab_c_caption,
                tab_c_max_ref_sec,
                tab_c_trim_ref,
                tab_c_steps,
                tab_c_cfg_text,
                tab_c_cfg_spk,
                tab_c_cfg_cap,
                tab_c_seed,
                tab_c_speed,
                tab_c_trim_out,
                tab_c_sr,
            ],
            outputs=[output_audio, output_download, output_status_md, history_df, vram_html],
        )

        tab_d_gen_btn.click(
            fn=handle_batch_generate,
            inputs=[
                tab_d_text,
                tab_d_file,
                tab_d_mode,
                tab_d_preset_dd,
                tab_d_caption,
                tab_d_max_ref_sec,
                tab_d_steps,
                tab_d_cfg_text,
                tab_d_cfg_spk,
                tab_d_cfg_cap,
                tab_d_seed,
                tab_d_speed,
                tab_d_trim_out,
                tab_d_sr,
            ],
            outputs=[
                tab_d_zip_out,
                tab_d_preview_audio,
                tab_d_results_df,
                tab_d_status,
                history_df,
                vram_html,
            ],
        )

        # History table click -> instant Replay & Download
        history_df.select(
            fn=handle_history_select,
            inputs=[],
            outputs=[output_audio, output_download, output_status_md],
        )
        refresh_history_btn.click(
            fn=lambda: ENGINE.history.get_table_rows(),
            inputs=[],
            outputs=[history_df],
        )

    return demo, theme, CUSTOM_CSS


def main() -> None:
    demo, theme, css = create_app()
    gradio_major = int(gr.__version__.split(".")[0])
    launch_kwargs: Dict[str, Any] = {
        "server_name": "127.0.0.1",
        "server_port": 7860,
        "inbrowser": True,
        "allowed_paths": [str(ENGINE.outputs_dir), str(ENGINE.presets_dir)],
    }
    if gradio_major >= 6:
        launch_kwargs["theme"] = theme
        launch_kwargs["css"] = css

    demo.queue(max_size=16).launch(**launch_kwargs)


if __name__ == "__main__":
    main()
