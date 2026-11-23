"""
text_processor.py
-----------------
Japanese NLP helper (Furigana / Hiragana / Romaji conversion & reading verification)
and AI Japanese Script / Dialogue Generator supporting Google Gemini (`google-genai`)
and OpenAI / OpenAI-compatible APIs.
"""

from __future__ import annotations

import html
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Optional NLP imports with clear runtime degradation
# ---------------------------------------------------------------------------
try:
    import pykakasi

    _KAKASI = pykakasi.kakasi()
except Exception:  # pragma: no cover - surfaced gracefully in UI
    _KAKASI = None

try:
    import fugashi

    _TAGGER = fugashi.Tagger()
except Exception:  # pragma: no cover - fallback to pykakasi if unidic unavailable
    _TAGGER = None


# ---------------------------------------------------------------------------
# Official Irodori-TTS v4 Emoji / Prosody Expression Catalog
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EmojiTag:
    emoji: str
    label: str
    category: str
    description: str


IRODORI_EMOJI_TAGS: List[EmojiTag] = [
    # Emotions
    EmojiTag("😆", "Happy / Joyful", "Emotion", "Joyfully, happily (喜びながら)"),
    EmojiTag("😊", "Warm Smile", "Emotion", "Gentle, cheerful smile tone"),
    EmojiTag("😭", "Sad / Sobbing", "Emotion", "Sobbing, crying, sorrowfully (嗚咽・泣き声・悲しみ)"),
    EmojiTag("😢", "Tearful", "Emotion", "Soft sadness / melancholy"),
    EmojiTag("😠", "Angry / Firm", "Emotion", "Irritated, sharp, firm tone"),
    EmojiTag("😒", "Tutting / Annoyed", "Emotion", "Clicking tongue, exasperated (舌打ち)"),
    EmojiTag("😱", "Surprised / Shout", "Emotion", "Scream, shout, shocked shriek (悲鳴・叫び)"),
    EmojiTag("😲", "Astonished", "Emotion", "Sudden surprise / amazement"),
    EmojiTag("🥺", "Trembling / Timid", "Emotion", "Trembling voice, uncertainly (声を震わせながら)"),
    EmojiTag("😰", "Nervous / Panicked", "Emotion", "Panicked, agitated, stuttering (慌てて・緊張)"),
    EmojiTag("🫶", "Gentle / Tender", "Emotion", "Gently, tenderly (優しく)"),
    EmojiTag("😏", "Teasing / Playful", "Emotion", "Teasing, playfully sweet (からかうように・甘えるように)"),
    # Vocal / Non-verbal & Delivery
    EmojiTag("👂", "Whisper", "Delivery", "Whisper, close to the ear (囁き・耳元の音)"),
    EmojiTag("🤭", "Chuckle / Laugh", "Delivery", "Chuckle, giggle, suppressed laugh (くすくす笑い)"),
    EmojiTag("😮‍💨", "Sigh / Breath", "Delivery", "Breath, sigh, exhale (吐息・溜息)"),
    EmojiTag("😮", "Gasp", "Delivery", "Sudden intake of breath (息をのむ)"),
    EmojiTag("😪", "Sleepy", "Delivery", "Sleepily, languidly (眠そうに・気だるげに)"),
    EmojiTag("🤧", "Cough / Sniffle", "Delivery", "Coughing, sniffling, clearing throat (咳払い・くしゃみ)"),
    # Pacing & Acoustic Effects
    EmojiTag("⏸️", "Pause", "Pacing", "Pause, brief silence (間・沈黙)"),
    EmojiTag("⏩", "Fast", "Pacing", "Fast-speaking, rapid-fire (早口・急いで)"),
    EmojiTag("🐢", "Slow", "Pacing", "Slowly, deliberate pacing (ゆっくりと)"),
    EmojiTag("💥", "Emphatic", "Pacing", "With force / momentum (勢いよく)"),
    EmojiTag("📢", "Echo / Reverb", "Acoustic", "Echo or reverb effect (エコー・リバーブ)"),
    EmojiTag("📞", "Phone / Speaker", "Acoustic", "Over the phone / speaker effect (電話越し)"),
]


# ---------------------------------------------------------------------------
# Japanese Text Analysis Helpers
# ---------------------------------------------------------------------------
_KANJI_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf々ヶ]")
_KATAKANA_RE = re.compile(r"[\u30a1-\u30f6]")


def _katakana_to_hiragana(text: str) -> str:
    """Convert Katakana characters (ァ-ヶ) to Hiragana (ぁ-ゖ)."""
    return _KATAKANA_RE.sub(lambda m: chr(ord(m.group(0)) - 0x60), text)


def _has_kanji(text: str) -> bool:
    return bool(_KANJI_RE.search(text))


@dataclass
class JapaneseAnalysisResult:
    original: str
    hiragana: str
    romaji: str
    inline_furigana: str
    ruby_html: str
    token_notes: str


def _analyze_with_fugashi_tokens(text: str) -> Optional[List[Tuple[str, str]]]:
    """
    Use fugashi + unidic-lite to obtain context-aware readings for Kanji tokens.
    Returns list of (surface, hiragana_reading) pairs, or None if unavailable.
    """
    if _TAGGER is None:
        return None
    pairs: List[Tuple[str, str]] = []
    # Preserve line breaks by analyzing line by line
    lines = text.splitlines(keepends=True)
    for line in lines:
        stripped = line.rstrip("\r\n")
        newline = line[len(stripped) :]
        if stripped:
            for word in _TAGGER(stripped):
                surface = word.surface
                reading = ""
                # unidic-lite exposes kana/pron on feature
                feat = getattr(word, "feature", None)
                if feat is not None:
                    kana = getattr(feat, "kana", None) or getattr(feat, "pron", None)
                    if kana and kana != "*":
                        reading = _katakana_to_hiragana(str(kana))
                if not reading and _KAKASI is not None:
                    converted = _KAKASI.convert(surface)
                    reading = "".join(item.get("hira", item.get("orig", "")) for item in converted)
                pairs.append((surface, reading or surface))
        if newline:
            pairs.append((newline, newline))
    return pairs


def analyze_japanese_text(text: str) -> JapaneseAnalysisResult:
    """
    Convert Japanese text into:
      - Hiragana (full reading)
      - Hepburn Romaji
      - Inline Furigana e.g. 今日(きょう)はいい天気(てんき)ですね。
      - Styled HTML `<ruby>` preview for Gradio HTML display
      - Kanji reading verification notes
    """
    if not text or not text.strip():
        empty_html = (
            "<div class='ruby-preview-box empty'>"
            "Enter or generate Japanese text to preview Furigana and Romaji readings."
            "</div>"
        )
        return JapaneseAnalysisResult(
            original="",
            hiragana="",
            romaji="",
            inline_furigana="",
            ruby_html=empty_html,
            token_notes="",
        )

    if _KAKASI is None and _TAGGER is None:
        msg = "pykakasi / fugashi not installed. Run setup_windows.bat to install NLP dependencies."
        return JapaneseAnalysisResult(
            original=text,
            hiragana=msg,
            romaji=msg,
            inline_furigana=text,
            ruby_html=f"<div class='ruby-preview-box'>{html.escape(msg)}</div>",
            token_notes=msg,
        )

    # Prefer fugashi (context-aware morphological readings) when available,
    # and combine with pykakasi for clean Hepburn romaji.
    token_pairs = _analyze_with_fugashi_tokens(text)

    hiragana_parts: List[str] = []
    inline_parts: List[str] = []
    ruby_parts: List[str] = []
    kanji_glossary: List[str] = []
    seen_kanji: set[Tuple[str, str]] = set()

    if token_pairs is not None:
        for surface, hira in token_pairs:
            if surface in ("\n", "\r\n", "\r"):
                hiragana_parts.append(surface)
                inline_parts.append(surface)
                ruby_parts.append("<br/>")
                continue

            hiragana_parts.append(hira)
            if _has_kanji(surface) and hira and hira != surface:
                inline_parts.append(f"{surface}({hira})")
                ruby_parts.append(
                    f"<ruby>{html.escape(surface)}<rt>{html.escape(hira)}</rt></ruby>"
                )
                key = (surface, hira)
                if key not in seen_kanji:
                    seen_kanji.add(key)
                    kanji_glossary.append(f"{surface} [{hira}]")
            else:
                inline_parts.append(surface)
                ruby_parts.append(html.escape(surface))
    else:
        assert _KAKASI is not None
        for line_idx, line in enumerate(text.splitlines()):
            if line_idx > 0:
                hiragana_parts.append("\n")
                inline_parts.append("\n")
                ruby_parts.append("<br/>")
            for item in _KAKASI.convert(line):
                orig = item.get("orig", "")
                hira = item.get("hira", orig)
                hiragana_parts.append(hira)
                if _has_kanji(orig) and hira and hira != orig:
                    inline_parts.append(f"{orig}({hira})")
                    ruby_parts.append(
                        f"<ruby>{html.escape(orig)}<rt>{html.escape(hira)}</rt></ruby>"
                    )
                    key = (orig, hira)
                    if key not in seen_kanji:
                        seen_kanji.add(key)
                        kanji_glossary.append(f"{orig} [{hira}]")
                else:
                    inline_parts.append(orig)
                    ruby_parts.append(html.escape(orig))

    hiragana_str = "".join(hiragana_parts)

    # Convert the context-aware Hiragana to clean Hepburn Romaji via pykakasi
    if _KAKASI is not None:
        romaji_lines: List[str] = []
        for line in hiragana_str.splitlines():
            if not line.strip():
                romaji_lines.append("")
                continue
            items = _KAKASI.convert(line)
            romaji_line = " ".join(
                it.get("hepburn", it.get("orig", "")).strip()
                for it in items
                if it.get("hepburn", it.get("orig", "")).strip()
            )
            # Clean punctuation spacing
            romaji_line = re.sub(r"\s+([,.!?、。！？…])", r"\1", romaji_line)
            romaji_lines.append(romaji_line)
        romaji_str = "\n".join(romaji_lines)
    else:
        romaji_str = "(Install pykakasi for Romaji output)"

    ruby_html = (
        "<div class='ruby-preview-box'>"
        f"<div class='ruby-text'>{''.join(ruby_parts)}</div>"
        "</div>"
    )
    token_notes = " | ".join(kanji_glossary[:24]) if kanji_glossary else "No Kanji detected."

    return JapaneseAnalysisResult(
        original=text,
        hiragana=hiragana_str,
        romaji=romaji_str,
        inline_furigana="".join(inline_parts),
        ruby_html=ruby_html,
        token_notes=token_notes,
    )


# ---------------------------------------------------------------------------
# AI Japanese Script & Dialogue Generator (Gemini / OpenAI-Compatible)
# ---------------------------------------------------------------------------
JLPT_GUIDANCE: Dict[str, str] = {
    "N5": "Use JLPT N5 beginner vocabulary and simple grammar (です/ます, basic particles). Keep sentences short and clear.",
    "N4": "Use JLPT N4 elementary Japanese (て-form, たい, と思う, simple everyday expressions).",
    "N3": "Use JLPT N3 intermediate Japanese suitable for natural everyday conversation and explanations.",
    "N2": "Use JLPT N2 upper-intermediate Japanese with rich vocabulary, natural connectors, and expressive phrasing.",
    "N1": "Use JLPT N1 advanced Japanese with nuanced expressions, idioms, and sophisticated vocabulary.",
    "Natural Native": "Use completely natural, idiomatic native Japanese as actually spoken by native speakers today.",
}

DIALECT_GUIDANCE: Dict[str, str] = {
    "Standard Tokyo (Hyojungo)": "Use natural Standard Tokyo Japanese (標準語) with balanced polite/natural phrasing suitable for the scenario.",
    "Keigo / Formal": "Use polished Keigo (敬語: 丁寧語・尊敬語・謙譲語) appropriate for business, announcements, or respectful service.",
    "Casual / Anime-style": "Use lively, expressive casual Japanese (タメ口) with natural sentence-ending particles (よ, ね, な, じゃん, かな) and expressive rhythm.",
    "Kansai-ben": "Use authentic, natural Kansai dialect (関西弁: やねん, めっちゃ, ほんま, 〜へん, 〜やろ, なんでやねん where fitting).",
}

GEMINI_DEFAULT_MODELS: List[str] = [
    "gemini-4-hydrogen",
    "gemini-4-argon",
    "gemini-4-neon",
    "gemini-3.5-flash-lite",
]

OPENAI_DEFAULT_MODELS: List[str] = [
    "gpt-4o-mini",
    "gpt-4.1-mini",
    "gpt-4.1",
]


class ScriptGenerationError(RuntimeError):
    """Raised when LLM script generation fails with an actionable message."""


def _build_script_prompts(
    topic: str,
    jlpt_level: str,
    dialect: str,
    script_format: str,
    num_lines: int,
    include_emojis: bool,
) -> Tuple[str, str]:
    jlpt_rule = JLPT_GUIDANCE.get(jlpt_level, JLPT_GUIDANCE["Natural Native"])
    dialect_rule = DIALECT_GUIDANCE.get(dialect, DIALECT_GUIDANCE["Standard Tokyo (Hyojungo)"])

    emoji_rule = (
        "You may naturally sprinkle 1–3 Irodori-TTS prosody emojis where emotionally fitting "
        "(choose ONLY from: 😆 joyful, 😭 sad, 😱 shocked, 👂 whisper, 🤭 chuckle, 😮‍💨 sigh, 🫶 gentle, 😏 playful, 🥺 timid, ⏸️ pause, ⏩ fast, 🐢 slow). "
        "Do not overuse emojis."
        if include_emojis
        else "Do NOT include any emojis or sound-effect brackets."
    )

    if script_format == "Multi-Line (Batch Ready)":
        format_rule = (
            f"Output exactly {num_lines} distinct Japanese spoken lines, ONE sentence or utterance per line, "
            "with NO numbering, NO speaker labels (no 'A:' or '話者:'), NO English translations, and NO markdown bullets. "
            "Each line must be directly ready for TTS batch synthesis."
        )
    elif script_format == "Dialogue (Unlabeled Lines)":
        format_rule = (
            f"Write a natural back-and-forth Japanese dialogue of about {num_lines} turns, ONE spoken turn per line, "
            "with NO speaker names, NO quotation marks wrapping lines, NO English translations, and NO markdown formatting "
            "so it can be read directly by TTS."
        )
    else:
        format_rule = (
            f"Write a cohesive spoken Japanese monologue/narration of approximately {num_lines} sentences. "
            "Output ONLY the Japanese text ready to be spoken aloud by a TTS engine. "
            "Do NOT include English translations, romaji, furigana parentheses, titles, or explanations."
        )

    system_instruction = (
        "You are an expert Japanese voice-acting scriptwriter specializing in natural, speakable Japanese "
        "for neural Text-to-Speech (Irodori-TTS v4-Large).\n"
        f"- Level: {jlpt_rule}\n"
        f"- Register / Dialect: {dialect_rule}\n"
        f"- Prosody Cues: {emoji_rule}\n"
        f"- Output Format: {format_rule}\n"
        "- Always use natural Japanese punctuation (、。！？…) to guide speech rhythm and pauses."
    )

    user_prompt = f"Topic / Scenario:\n{topic.strip()}"
    return system_instruction, user_prompt


def generate_japanese_script(
    topic: str,
    jlpt_level: str,
    dialect: str,
    provider: str,
    api_key: str,
    model_name: str = "",
    base_url: str = "",
    script_format: str = "Single Monologue",
    num_lines: int = 4,
    include_emojis: bool = True,
) -> str:
    """
    Generate Japanese TTS-ready script using Google Gemini (`google-genai`)
    or OpenAI / OpenAI-compatible APIs.
    API key is accepted from UI session state (or optional environment variable fallback).
    """
    if not topic or not topic.strip():
        raise ScriptGenerationError("Please enter a topic or scenario prompt first.")

    effective_key = (api_key or "").strip()
    if provider == "Google Gemini":
        effective_key = effective_key or os.getenv("GEMINI_API_KEY", "").strip()
        if not effective_key:
            raise ScriptGenerationError(
                "Missing Gemini API Key. Enter your API key in the AI Generator panel "
                "(or set GEMINI_API_KEY in a local .env file)."
            )
    else:
        effective_key = effective_key or os.getenv("OPENAI_API_KEY", "").strip()
        # Allow local OpenAI-compatible servers (e.g., Ollama / LM Studio) with placeholder key when base_url is set
        if not effective_key and not (base_url or "").strip():
            raise ScriptGenerationError(
                "Missing OpenAI API Key. Enter your API key in the AI Generator panel "
                "(or specify a local OpenAI-compatible Base URL such as http://localhost:11434/v1)."
            )
        if not effective_key:
            effective_key = "local-no-key-required"

    system_instruction, user_prompt = _build_script_prompts(
        topic=topic,
        jlpt_level=jlpt_level,
        dialect=dialect,
        script_format=script_format,
        num_lines=max(1, min(int(num_lines), 30)),
        include_emojis=include_emojis,
    )

    if provider == "Google Gemini":
        return _call_gemini(
            api_key=effective_key,
            model_name=(model_name or "").strip() or "gemini-4-hydrogen",
            system_instruction=system_instruction,
            user_prompt=user_prompt,
        )
    return _call_openai_compatible(
        api_key=effective_key,
        model_name=(model_name or "").strip() or "gpt-4o-mini",
        base_url=(base_url or "").strip() or None,
        system_instruction=system_instruction,
        user_prompt=user_prompt,
    )


def _call_gemini(
    api_key: str,
    model_name: str,
    system_instruction: str,
    user_prompt: str,
) -> str:
    try:
        from google import genai
    except ImportError as exc:
        raise ScriptGenerationError(
            "`google-genai` package is not installed. Run `pip install -U google-genai` inside .venv."
        ) from exc

    # Guard against deprecated legacy Gemini model IDs
    if model_name.startswith(("gemini-1.5-", "gemini-2.0-", "gemini-2.5-")):
        model_name = "gemini-4-hydrogen"

    try:
        client = genai.Client(api_key=api_key)
        interaction = client.interactions.create(
            model=model_name,
            system_instruction=system_instruction,
            input=user_prompt,
            generation_config={"max_output_tokens": 1024},
        )
        output_text = (getattr(interaction, "output_text", None) or "").strip()
        if not output_text:
            # Compatibility fallback for google-genai 2.x responses on Python 3.10
            for container_attr in ("outputs", "steps"):
                items = getattr(interaction, container_attr, None) or []
                chunks: List[str] = []
                for item in items:
                    txt = getattr(item, "text", None)
                    if txt:
                        chunks.append(str(txt))
                    for part in getattr(item, "content", None) or []:
                        ptxt = getattr(part, "text", None)
                        if ptxt:
                            chunks.append(str(ptxt))
                if chunks:
                    output_text = "\n".join(chunks).strip()
                    break
        if not output_text:
            raise ScriptGenerationError("Gemini returned an empty response. Try refining your prompt.")
        return output_text
    except ScriptGenerationError:
        raise
    except Exception as exc:
        raise ScriptGenerationError(f"Gemini API error ({model_name}): {exc}") from exc


def _call_openai_compatible(
    api_key: str,
    model_name: str,
    base_url: Optional[str],
    system_instruction: str,
    user_prompt: str,
) -> str:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ScriptGenerationError(
            "`openai` package is not installed. Run `pip install -U openai` inside .venv."
        ) from exc

    try:
        client_kwargs: Dict[str, object] = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        client = OpenAI(**client_kwargs)  # type: ignore[arg-type]
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=1024,
        )
        content = ""
        if response.choices and response.choices[0].message:
            content = (response.choices[0].message.content or "").strip()
        if not content:
            raise ScriptGenerationError("OpenAI-compatible API returned an empty response.")
        return content
    except ScriptGenerationError:
        raise
    except Exception as exc:
        raise ScriptGenerationError(f"OpenAI-compatible API error ({model_name}): {exc}") from exc
