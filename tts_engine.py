"""
tts_engine.py
-------------
Production backend inference wrapper for `Aratako/Irodori-TTS-v4-Large-Quantized`
(Hugging Face + `torchao` quantization) tailored for Windows 10/11 + NVIDIA CUDA 12.x
(optimized for RTX 4060 Laptop GPU 8GB VRAM, Ada Lovelace CC 8.9, bfloat16 compute).

Includes:
  - Strict VRAM lifecycle management (Load / Unload / Real-time VRAM telemetry)
  - All 4 Irodori-TTS v4-Large modes (Zero-Shot Voice Cloning + Long Reference Audio,
    Text-based Voice Design, Style-Controlled Voice Cloning, and Batch ZIP synthesis)
  - Audio preprocessing (resampling, mono conversion, long-reference clamping)
  - Pitch-preserving speed adjustment (WSOLA), silence trimming, and sample-rate export
  - Local Reference Voice Preset Manager (`./presets/`)
  - Persistent Generation History Manager (`./outputs/`, last 15 clips)
"""

from __future__ import annotations

import csv
import dataclasses
import gc
import inspect
import json
import math
import os
import re
import shutil
import tempfile
import threading
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# Reduce CUDA allocator fragmentation on 8GB Windows WDDM GPUs (must be set before CUDA init)
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "max_split_size_mb:128,garbage_collection_threshold:0.8",
)

import numpy as np
import soundfile as sf
import torch
from scipy import signal as scipy_signal

try:
    import torchaudio
except Exception:  # pragma: no cover - soundfile/scipy fallback handles audio I/O cleanly
    torchaudio = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Constants & Hardware Profile
# ---------------------------------------------------------------------------
HF_REPO_ID = "Aratako/Irodori-TTS-v4-Large-Quantized"
DEFAULT_CODEC_REPO = "Aratako/Semantic-DACVAE-Japanese-32dim"
NATIVE_SAMPLE_RATE = 48000
MAX_HISTORY_ITEMS = 15

QUANTIZATION_VARIANTS: Dict[str, Dict[str, Any]] = {
    "float8-weight-only": {
        "label": "float8-weight-only (Default — RTX 4060 Ada Optimal, ~3.6 GB VRAM)",
        "est_vram_gb": 3.6,
        "min_cc": (8, 9),
        "description": "FP8 weights + BF16 compute. Best quality-to-VRAM ratio on Ada/Hopper/Blackwell (CC 8.9+).",
    },
    "float8-dynamic": {
        "label": "float8-dynamic (FP8 Weights + Dynamic FP8 Activations, ~3.7 GB VRAM)",
        "est_vram_gb": 3.7,
        "min_cc": (8, 9),
        "description": "FP8 weights & dynamic FP8 activations. Fast matmuls on RTX 40-series (CC 8.9+).",
    },
    "int4-weight-only": {
        "label": "int4-weight-only (Lowest VRAM — ~2.8 GB VRAM)",
        "est_vram_gb": 2.8,
        "min_cc": (8, 0),
        "description": "W4A16 group-128 CUDA tinygemm kernels. Smallest VRAM footprint (Ampere CC 8.0+).",
    },
    "int8-weight-only": {
        "label": "int8-weight-only (Broad Compatibility — ~3.8 GB VRAM)",
        "est_vram_gb": 3.8,
        "min_cc": (7, 0),
        "description": "W8A16 INT8 linear weights with BF16 compute. High stability across CUDA GPUs.",
    },
}

EXPORT_SAMPLE_RATES: Dict[str, int] = {
    "48 kHz (Native Studio WAV)": 48000,
    "44.1 kHz (CD Quality WAV)": 44100,
    "24 kHz (Compact Speech WAV)": 24000,
}


class TTSEngineError(RuntimeError):
    """Actionable error surfaced cleanly to the Gradio UI without crashing."""


@dataclass
class VRAMStats:
    cuda_available: bool
    gpu_name: str
    compute_capability: str
    allocated_mb: float
    reserved_mb: float
    free_mb: float
    total_mb: float
    utilization_pct: float
    model_loaded: bool
    active_variant: Optional[str]
    codec_device: str = "cpu"


@dataclass
class SynthesisResult:
    output_path: Path
    sample_rate: int
    duration_sec: float
    elapsed_sec: float
    mode: str
    text: str
    caption: str
    seed_used: int
    timestamp: str


# ---------------------------------------------------------------------------
# Audio DSP Utilities (Windows-safe, pure NumPy/SciPy + soundfile/torchaudio)
# ---------------------------------------------------------------------------
def load_audio_mono(audio_path: Path, target_sr: int = NATIVE_SAMPLE_RATE) -> Tuple[np.ndarray, int]:
    """
    Load `.wav`, `.flac`, or `.mp3` cleanly on Windows into float32 mono numpy array
    resampled to `target_sr`.
    """
    if not audio_path.exists():
        raise TTSEngineError(f"Audio file not found: {audio_path}")

    waveform: Optional[np.ndarray] = None
    sr: int = target_sr

    # Attempt soundfile first (fast, zero FFmpeg issues for WAV/FLAC/OGG/MP3 via libsndfile>=1.1)
    try:
        data, sr = sf.read(str(audio_path), dtype="float32", always_2d=True)
        waveform = data.mean(axis=1)
    except Exception as sf_err:
        if torchaudio is not None:
            try:
                wav_tensor, sr = torchaudio.load(str(audio_path))
                if wav_tensor.ndim == 2 and wav_tensor.shape[0] > 1:
                    wav_tensor = wav_tensor.mean(dim=0)
                else:
                    wav_tensor = wav_tensor.squeeze(0)
                waveform = wav_tensor.detach().cpu().numpy().astype(np.float32)
            except Exception as ta_err:
                raise TTSEngineError(
                    f"Failed to decode audio file '{audio_path.name}'. "
                    f"soundfile error: {sf_err} | torchaudio error: {ta_err}"
                ) from ta_err
        else:
            raise TTSEngineError(
                f"Failed to decode audio file '{audio_path.name}': {sf_err}. "
                "Convert the file to .wav or ensure torchaudio is installed."
            ) from sf_err

    assert waveform is not None
    if waveform.size == 0:
        raise TTSEngineError(f"Reference audio '{audio_path.name}' is empty.")

    if sr != target_sr:
        waveform = resample_audio(waveform, sr, target_sr)
        sr = target_sr

    # Peak normalize gently if clipping
    peak = float(np.max(np.abs(waveform)))
    if peak > 0.99:
        waveform = (waveform / peak) * 0.98

    return waveform.astype(np.float32), sr


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """High-quality polyphase resampling via scipy.signal.resample_poly."""
    if orig_sr == target_sr or audio.size == 0:
        return audio.astype(np.float32, copy=False)
    gcd = math.gcd(int(orig_sr), int(target_sr))
    up = int(target_sr) // gcd
    down = int(orig_sr) // gcd
    resampled = scipy_signal.resample_poly(audio.astype(np.float64), up, down)
    return np.clip(resampled, -1.0, 1.0).astype(np.float32)


def trim_silence(
    audio: np.ndarray,
    sr: int,
    threshold_db: float = -42.0,
    pad_ms: float = 80.0,
) -> np.ndarray:
    """Trim leading and trailing silence using short-time RMS energy with safe padding."""
    if audio.size < int(sr * 0.05):
        return audio

    frame_len = max(256, int(sr * 0.02))
    hop = max(128, frame_len // 2)
    threshold_amp = 10.0 ** (threshold_db / 20.0)

    # Frame RMS
    num_frames = 1 + max(0, (audio.size - frame_len) // hop)
    if num_frames <= 1:
        return audio

    active_indices: List[int] = []
    for idx in range(num_frames):
        start = idx * hop
        frame = audio[start : start + frame_len]
        rms = float(np.sqrt(np.mean(frame * frame) + 1e-12))
        if rms >= threshold_amp:
            active_indices.append(idx)

    if not active_indices:
        return audio

    pad_samples = int(sr * (pad_ms / 1000.0))
    start_sample = max(0, active_indices[0] * hop - pad_samples)
    end_sample = min(audio.size, active_indices[-1] * hop + frame_len + pad_samples)
    trimmed = audio[start_sample:end_sample]
    return trimmed if trimmed.size > 0 else audio


def adjust_speed_wsola(audio: np.ndarray, sr: int, speed: float) -> np.ndarray:
    """
    Pitch-preserving time-scale modification using Waveform Similarity Overlap-Add (WSOLA).
    `speed > 1.0` makes speech faster; `speed < 1.0` makes speech slower.
    """
    speed = float(np.clip(speed, 0.5, 2.0))
    if abs(speed - 1.0) < 0.01 or audio.size < int(sr * 0.1):
        return audio.astype(np.float32, copy=False)

    win_len = int(sr * 0.032)  # 32 ms window for speech formants
    if win_len % 2 == 1:
        win_len += 1
    syn_hop = win_len // 2
    ana_hop = max(1, int(round(syn_hop * speed)))
    search_radius = max(16, syn_hop // 2)

    window = np.hanning(win_len).astype(np.float32)
    n_in = audio.size
    est_out_len = int(math.ceil(n_in / speed)) + win_len * 2
    out = np.zeros(est_out_len, dtype=np.float32)
    norm = np.zeros(est_out_len, dtype=np.float32)

    # First frame
    first_frame = audio[: min(win_len, n_in)]
    if first_frame.size < win_len:
        first_frame = np.pad(first_frame, (0, win_len - first_frame.size))
    out[:win_len] += first_frame * window
    norm[:win_len] += window

    in_pos = ana_hop
    out_pos = syn_hop
    prev_in_pos = 0

    while in_pos + win_len < n_in and out_pos + win_len < est_out_len:
        # Target continuation after previous analysis frame
        nat_pos = prev_in_pos + syn_hop
        if nat_pos + win_len < n_in:
            target = audio[nat_pos : nat_pos + syn_hop]
            lo = max(0, in_pos - search_radius)
            hi = min(n_in - win_len, in_pos + search_radius)
            best_pos = in_pos
            if hi > lo:
                # Coarse stride=2 cross-correlation for speed + responsiveness
                best_score = -1e18
                for cand in range(lo, hi + 1, 2):
                    seg = audio[cand : cand + syn_hop]
                    score = float(np.dot(seg, target))
                    if score > best_score:
                        best_score = score
                        best_pos = cand
        else:
            best_pos = in_pos

        frame = audio[best_pos : best_pos + win_len]
        out[out_pos : out_pos + win_len] += frame * window
        norm[out_pos : out_pos + win_len] += window

        prev_in_pos = best_pos
        in_pos += ana_hop
        out_pos += syn_hop

    valid_len = min(out_pos + win_len, est_out_len)
    out = out[:valid_len]
    norm = norm[:valid_len]
    mask = norm > 1e-4
    out[mask] /= norm[mask]
    return np.clip(out, -1.0, 1.0).astype(np.float32)


def preprocess_reference_audio(
    audio_path: Path,
    max_seconds: float = 30.0,
    trim_ref_silence: bool = True,
    target_sr: int = NATIVE_SAMPLE_RATE,
    work_dir: Optional[Path] = None,
) -> Tuple[Path, float]:
    """
    Preprocess a reference audio clip (including long reference audio up to 120s):
    - Decodes WAV/MP3/FLAC safely on Windows
    - Converts to mono and resamples to `target_sr`
    - Optionally trims dead leading/trailing silence
    - Clamps duration cleanly to `max_seconds`
    Returns (processed_wav_path, duration_seconds).
    """
    wav, sr = load_audio_mono(audio_path, target_sr=target_sr)
    if trim_ref_silence:
        wav = trim_silence(wav, sr, threshold_db=-45.0, pad_ms=100.0)

    max_samples = int(max(1.0, min(float(max_seconds), 120.0)) * sr)
    if wav.size > max_samples:
        wav = wav[:max_samples]

    if wav.size < int(sr * 0.4):
        raise TTSEngineError(
            "Reference audio is shorter than 0.4 seconds after silence trimming. "
            "Please provide at least 2–10 seconds of clear speech."
        )

    out_dir = work_dir or Path(tempfile.gettempdir()) / "irodori_tts_ref_cache"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"ref_proc_{int(time.time() * 1000)}.wav"
    sf.write(str(out_path), wav, sr, subtype="PCM_16")
    return out_path, float(wav.size) / float(sr)


# ---------------------------------------------------------------------------
# Reference Voice Preset Manager (`./presets/`)
# ---------------------------------------------------------------------------
class VoicePresetManager:
    """Manages persistent local reference audio presets in `./presets/`."""

    def __init__(self, presets_dir: Path) -> None:
        self.presets_dir = presets_dir.resolve()
        self.presets_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _sanitize_name(name: str) -> str:
        cleaned = re.sub(r"[^\w\-\u3040-\u30ff\u4e00-\u9fff ]+", "", name.strip())
        cleaned = re.sub(r"\s+", "_", cleaned).strip("_")
        return cleaned[:64]

    def list_presets(self) -> List[str]:
        names = sorted(p.stem for p in self.presets_dir.glob("*.wav"))
        return names

    def get_preset_info(self, name: str) -> Optional[Dict[str, Any]]:
        if not name or name == "(None)":
            return None
        wav_path = self.presets_dir / f"{name}.wav"
        if not wav_path.exists():
            return None
        meta_path = self.presets_dir / f"{name}.json"
        meta: Dict[str, Any] = {"name": name, "wav_path": str(wav_path), "transcript": "", "duration_sec": 0.0}
        if meta_path.exists():
            try:
                loaded = json.loads(meta_path.read_text(encoding="utf-8"))
                meta.update(loaded)
            except Exception:
                pass
        meta["wav_path"] = str(wav_path)
        return meta

    def list_preset_rows(self) -> List[List[Any]]:
        rows: List[List[Any]] = []
        for name in self.list_presets():
            info = self.get_preset_info(name)
            if info:
                rows.append(
                    [
                        name,
                        f"{float(info.get('duration_sec', 0.0)):.2f}s",
                        str(info.get("transcript", ""))[:80],
                        str(info.get("created_at", "")),
                    ]
                )
        return rows

    def save_preset(
        self,
        preset_name: str,
        source_audio_path: str,
        transcript: str = "",
        max_seconds: float = 120.0,
    ) -> Dict[str, Any]:
        safe_name = self._sanitize_name(preset_name)
        if not safe_name:
            raise TTSEngineError("Please enter a valid preset name (e.g., Female_Formal, Male_Casual).")
        if not source_audio_path:
            raise TTSEngineError("Please upload or record a reference audio clip before saving a preset.")

        src = Path(source_audio_path)
        wav, sr = load_audio_mono(src, target_sr=NATIVE_SAMPLE_RATE)
        max_samples = int(max(1.0, min(float(max_seconds), 120.0)) * sr)
        if wav.size > max_samples:
            wav = wav[:max_samples]
        duration_sec = float(wav.size) / float(sr)

        dest_wav = self.presets_dir / f"{safe_name}.wav"
        sf.write(str(dest_wav), wav, sr, subtype="PCM_16")

        meta = {
            "name": safe_name,
            "wav_path": str(dest_wav),
            "transcript": (transcript or "").strip(),
            "duration_sec": round(duration_sec, 2),
            "sample_rate": sr,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        meta_path = self.presets_dir / f"{safe_name}.json"
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        return meta

    def delete_preset(self, preset_name: str) -> bool:
        if not preset_name or preset_name == "(None)":
            return False
        wav_path = self.presets_dir / f"{preset_name}.wav"
        meta_path = self.presets_dir / f"{preset_name}.json"
        removed = False
        if wav_path.exists():
            wav_path.unlink()
            removed = True
        if meta_path.exists():
            meta_path.unlink()
        return removed


# ---------------------------------------------------------------------------
# Persistent Generation History Manager (`./outputs/`)
# ---------------------------------------------------------------------------
class HistoryManager:
    """Persists and serves the last 15 generated audio clips in `./outputs/`."""

    def __init__(self, outputs_dir: Path, max_items: int = MAX_HISTORY_ITEMS) -> None:
        self.outputs_dir = outputs_dir.resolve()
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.outputs_dir / "history_index.json"
        self.max_items = max_items
        self._lock = threading.Lock()

    def _read_entries(self) -> List[Dict[str, Any]]:
        if not self.index_path.exists():
            return []
        try:
            data = json.loads(self.index_path.read_text(encoding="utf-8"))
            if not isinstance(data, list):
                return []
            # Keep only items whose WAV file still exists on disk
            valid = [item for item in data if Path(item.get("path", "")).exists()]
            return valid[: self.max_items]
        except Exception:
            return []

    def _write_entries(self, entries: List[Dict[str, Any]]) -> None:
        self.index_path.write_text(
            json.dumps(entries[: self.max_items], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def add_entry(self, result: SynthesisResult) -> None:
        with self._lock:
            entries = self._read_entries()
            entries.insert(
                0,
                {
                    "timestamp": result.timestamp,
                    "mode": result.mode,
                    "text": result.text,
                    "caption": result.caption,
                    "duration_sec": round(result.duration_sec, 2),
                    "sample_rate": result.sample_rate,
                    "elapsed_sec": round(result.elapsed_sec, 2),
                    "seed": result.seed_used,
                    "filename": result.output_path.name,
                    "path": str(result.output_path),
                },
            )
            self._write_entries(entries[: self.max_items])

    def get_entries(self) -> List[Dict[str, Any]]:
        with self._lock:
            return self._read_entries()

    def get_table_rows(self) -> List[List[Any]]:
        rows: List[List[Any]] = []
        for idx, item in enumerate(self.get_entries(), start=1):
            snippet = str(item.get("text", "")).replace("\n", " ")
            if len(snippet) > 48:
                snippet = snippet[:48] + "…"
            rows.append(
                [
                    idx,
                    item.get("timestamp", ""),
                    item.get("mode", ""),
                    snippet,
                    f"{float(item.get('duration_sec', 0.0)):.2f}s",
                    f"{int(item.get('sample_rate', 48000)) // 1000} kHz",
                    item.get("filename", ""),
                ]
            )
        return rows

    def get_path_by_row(self, row_index: int) -> Optional[str]:
        entries = self.get_entries()
        if 0 <= row_index < len(entries):
            path = Path(entries[row_index]["path"])
            if path.exists():
                return str(path)
        return None


# ---------------------------------------------------------------------------
# Core Irodori-TTS v4-Large Quantized Engine
# ---------------------------------------------------------------------------
class IrodoriTTSEngine:
    """
    Thread-safe wrapper around `Aratako/Irodori-TTS-v4-Large-Quantized` using the
    official `irodori_tts.inference_runtime` (`InferenceRuntime`, `RuntimeKey`,
    `SamplingRequest`) with `torchao` quantization and strict 8GB VRAM hygiene.
    """

    def __init__(self, base_dir: Optional[Path] = None) -> None:
        self.base_dir = (base_dir or Path(__file__).resolve().parent).resolve()
        self.outputs_dir = self.base_dir / "outputs"
        self.presets_dir = self.base_dir / "presets"
        self.cache_dir = self.base_dir / "models_cache"
        self.temp_dir = self.base_dir / ".tmp_audio"

        for d in (self.outputs_dir, self.presets_dir, self.cache_dir, self.temp_dir):
            d.mkdir(parents=True, exist_ok=True)

        # Ensure Hugging Face and PyTorch caches stay on the workspace drive (e.g., D:) rather than C:
        hf_home = self.cache_dir / "huggingface"
        torch_home = self.cache_dir / "torch"
        hf_home.mkdir(parents=True, exist_ok=True)
        torch_home.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HOME", str(hf_home))
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(hf_home / "hub"))
        os.environ.setdefault("TORCH_HOME", str(torch_home))

        self.presets = VoicePresetManager(self.presets_dir)
        self.history = HistoryManager(self.outputs_dir)

        self._lock = threading.Lock()
        self._runtime: Any = None
        self._active_variant: Optional[str] = None
        self._active_codec_device: str = "cpu"
        self._runtime_cls: Any = None

    # ------------------------------------------------------------------
    # Hardware & VRAM Telemetry
    # ------------------------------------------------------------------
    def get_vram_stats(self) -> VRAMStats:
        cuda_ok = torch.cuda.is_available()
        if cuda_ok:
            try:
                props = torch.cuda.get_device_properties(0)
                gpu_name = props.name
                cc = f"{props.major}.{props.minor}"
                total_mb = float(props.total_memory) / (1024.0 * 1024.0)
                allocated_mb = float(torch.cuda.memory_allocated(0)) / (1024.0 * 1024.0)
                reserved_mb = float(torch.cuda.memory_reserved(0)) / (1024.0 * 1024.0)
                try:
                    free_bytes, _ = torch.cuda.mem_get_info(0)
                    free_mb = float(free_bytes) / (1024.0 * 1024.0)
                except Exception:
                    free_mb = max(0.0, total_mb - allocated_mb)
                used_total_mb = max(allocated_mb, total_mb - free_mb)
                pct = (used_total_mb / total_mb * 100.0) if total_mb > 0 else 0.0
            except Exception:
                gpu_name = "NVIDIA CUDA GPU"
                cc = "Unknown"
                total_mb = 8192.0
                allocated_mb = 0.0
                reserved_mb = 0.0
                free_mb = 8192.0
                pct = 0.0
        else:
            gpu_name = "No CUDA GPU Detected (CPU Fallback)"
            cc = "N/A"
            total_mb = 8192.0
            allocated_mb = 0.0
            reserved_mb = 0.0
            free_mb = 0.0
            pct = 0.0

        return VRAMStats(
            cuda_available=cuda_ok,
            gpu_name=gpu_name,
            compute_capability=cc,
            allocated_mb=allocated_mb,
            reserved_mb=reserved_mb,
            free_mb=free_mb,
            total_mb=total_mb,
            utilization_pct=min(100.0, max(0.0, pct)),
            model_loaded=self._runtime is not None,
            active_variant=self._active_variant,
            codec_device=self._active_codec_device,
        )

    @property
    def is_loaded(self) -> bool:
        return self._runtime is not None

    @property
    def active_variant(self) -> Optional[str]:
        return self._active_variant

    # ------------------------------------------------------------------
    # Model Lifecycle (Load / Unload with Strict VRAM Cleanup)
    # ------------------------------------------------------------------
    def _clear_cuda_memory(self) -> None:
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                try:
                    torch.cuda.ipc_collect()
                except Exception:
                    pass
        gc.collect()

    def unload_model(self) -> str:
        with self._lock:
            was_loaded = self._runtime is not None
            prev_variant = self._active_variant

            if self._runtime is not None:
                # Move submodules off CUDA if exposed before releasing reference
                for attr in ("model", "codec", "text_encoder", "speaker_encoder"):
                    mod = getattr(self._runtime, attr, None)
                    if mod is not None and hasattr(mod, "to"):
                        try:
                            mod.to("cpu")
                        except Exception:
                            pass
                self._runtime = None

            # Clear any class-level runtime cache inside irodori_tts if present
            if self._runtime_cls is not None:
                for cache_attr in ("_cache", "_instances", "cache"):
                    cache_obj = getattr(self._runtime_cls, cache_attr, None)
                    if isinstance(cache_obj, dict):
                        cache_obj.clear()
                if hasattr(self._runtime_cls, "clear_cache"):
                    try:
                        self._runtime_cls.clear_cache()
                    except Exception:
                        pass

            self._active_variant = None
            self._clear_cuda_memory()

        stats = self.get_vram_stats()
        if was_loaded:
            return (
                f"Unloaded `{prev_variant}` and cleared CUDA cache. "
                f"Current VRAM: {stats.allocated_mb:.0f} MB / {stats.total_mb:.0f} MB."
            )
        return (
            f"No model was loaded. CUDA cache swept. "
            f"Current VRAM: {stats.allocated_mb:.0f} MB / {stats.total_mb:.0f} MB."
        )

    def _download_quantized_variant(self, variant: str) -> Path:
        """
        Download the requested quantized subfolder from `Aratako/Irodori-TTS-v4-Large-Quantized`
        using the official `irodori_tts.inference_runtime.download_hf_checkpoint` helper
        (which also fetches any bundled tokenizer assets), falling back to `huggingface_hub`.
        """
        hf_subpath = f"{HF_REPO_ID}/{variant}"
        try:
            from irodori_tts.inference_runtime import download_hf_checkpoint

            resolved = Path(download_hf_checkpoint(hf_subpath))
            if resolved.exists():
                return resolved
        except Exception:
            pass

        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise TTSEngineError(
                "`huggingface-hub` is not installed. Run setup_windows.bat to install dependencies."
            ) from exc

        local_root = self.cache_dir / "Irodori-TTS-v4-Large-Quantized"
        local_root.mkdir(parents=True, exist_ok=True)

        try:
            snapshot_path = Path(
                snapshot_download(
                    repo_id=HF_REPO_ID,
                    allow_patterns=[f"{variant}/*", "config.json", "*.json"],
                    local_dir=str(local_root),
                )
            )
        except Exception as exc:
            raise TTSEngineError(
                f"Failed to download `{hf_subpath}` from Hugging Face: {exc}\n"
                "Check your internet connection and verify the repo is accessible."
            ) from exc

        variant_dir = snapshot_path / variant
        for candidate_name in ("model.safetensors", "checkpoint.safetensors", "model.pt"):
            cand = variant_dir / candidate_name
            if cand.exists():
                return cand

        safetensor_files = sorted(variant_dir.glob("*.safetensors"))
        if safetensor_files:
            return safetensor_files[0]

        if variant_dir.exists():
            return variant_dir

        raise TTSEngineError(
            f"Downloaded `{HF_REPO_ID}` but could not locate checkpoint files in `{variant_dir}`."
        )

    @staticmethod
    def _build_filtered_kwargs(target_cls: Any, candidate_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Inspect dataclass fields or `__init__` signature so our wrapper stays 100%
        compatible across minor `irodori_tts` updates on GitHub.
        """
        valid_names: Optional[set[str]] = None
        if dataclasses.is_dataclass(target_cls):
            valid_names = {f.name for f in dataclasses.fields(target_cls)}
        else:
            try:
                sig = inspect.signature(target_cls)
                if not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
                    valid_names = set(sig.parameters.keys())
            except Exception:
                valid_names = None

        if valid_names is None:
            return {k: v for k, v in candidate_kwargs.items() if v is not None}
        return {k: v for k, v in candidate_kwargs.items() if k in valid_names and v is not None}

    def load_model(self, variant: str = "float8-weight-only", codec_device: str = "cpu") -> str:
        """
        Load `Aratako/Irodori-TTS-v4-Large-Quantized/<variant>` onto CUDA with BF16 compute.
        Defaults `codec_device="cpu"` (matching upstream Irodori-TTS default) so the DACVAE
        codec and watermarker stay in 32GB system RAM, saving ~1+ GB of 8GB GPU VRAM.
        Unloads any existing model first to protect 8GB VRAM.
        """
        if variant not in QUANTIZATION_VARIANTS:
            raise TTSEngineError(
                f"Unknown quantization variant '{variant}'. "
                f"Valid choices: {', '.join(QUANTIZATION_VARIANTS.keys())}"
            )

        codec_dev = "cuda" if str(codec_device).strip().lower() == "cuda" else "cpu"

        if not torch.cuda.is_available():
            raise TTSEngineError(
                "CUDA GPU is not available in PyTorch (`torch.cuda.is_available() == False`).\n"
                "Run `setup_windows.bat` to install CUDA-enabled PyTorch for your RTX 4060."
            )

        # Verify torchao is importable before loading quantized weights
        try:
            import torchao  # noqa: F401
        except ImportError as exc:
            raise TTSEngineError(
                "`torchao` is not installed. Run `setup_windows.bat` to install torchao 0.16.x."
            ) from exc

        # Verify compute capability requirement for selected quantization scheme
        props = torch.cuda.get_device_properties(0)
        cc_tuple = (props.major, props.minor)
        min_cc = QUANTIZATION_VARIANTS[variant]["min_cc"]
        if cc_tuple < min_cc:
            raise TTSEngineError(
                f"Variant `{variant}` requires NVIDIA Compute Capability >= {min_cc[0]}.{min_cc[1]}, "
                f"but detected {props.name} (CC {cc_tuple[0]}.{cc_tuple[1]}).\n"
                "Please select `int8-weight-only` instead."
            )

        # Import official Irodori-TTS runtime
        try:
            from irodori_tts.inference_runtime import InferenceRuntime, RuntimeKey
        except ImportError as exc:
            raise TTSEngineError(
                "The official `irodori_tts` package is not installed.\n"
                "Run `setup_windows.bat` to install it into `.venv`."
            ) from exc

        # Already loaded with the same variant and codec device?
        if (
            self._runtime is not None
            and self._active_variant == variant
            and self._active_codec_device == codec_dev
        ):
            stats = self.get_vram_stats()
            return (
                f"Model `{HF_REPO_ID}/{variant}` is already loaded on {stats.gpu_name} "
                f"(codec on `{codec_dev}`). VRAM: {stats.allocated_mb:.0f} MB / {stats.total_mb:.0f} MB."
            )

        # Unload any previous variant first to avoid OOM on 8GB VRAM
        if self._runtime is not None:
            self.unload_model()

        with self._lock:
            self._runtime_cls = InferenceRuntime
            self._clear_cuda_memory()

            # Enable TF32 for Ada Lovelace performance on non-quantized layers
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

            hf_subpath = f"{HF_REPO_ID}/{variant}"
            local_ckpt_path = self._download_quantized_variant(variant)

            candidate_rk: Dict[str, Any] = {
                "checkpoint": str(local_ckpt_path),
                "model_device": "cuda",
                "codec_device": codec_dev,
                "model_precision": "bf16",
                "codec_precision": "fp32",
                "codec_repo": DEFAULT_CODEC_REPO,
            }

            t0 = time.perf_counter()
            try:
                rk_kwargs = self._build_filtered_kwargs(RuntimeKey, candidate_rk)
                runtime_key = RuntimeKey(**rk_kwargs)
                self._runtime = InferenceRuntime.from_key(runtime_key)
                self._active_variant = variant
                self._active_codec_device = codec_dev
            except torch.cuda.OutOfMemoryError as oom_err:
                self._runtime = None
                self._active_variant = None
                self._clear_cuda_memory()

                # If user selected codec on CUDA and OOM'd, automatically retry with codec on CPU
                if codec_dev == "cuda":
                    try:
                        candidate_rk["codec_device"] = "cpu"
                        rk_kwargs = self._build_filtered_kwargs(RuntimeKey, candidate_rk)
                        runtime_key = RuntimeKey(**rk_kwargs)
                        self._runtime = InferenceRuntime.from_key(runtime_key)
                        self._active_variant = variant
                        self._active_codec_device = "cpu"
                        codec_dev = "cpu"
                    except Exception as retry_err:
                        self._runtime = None
                        self._active_variant = None
                        self._clear_cuda_memory()
                        stats_now = self.get_vram_stats()
                        raise TTSEngineError(
                            f"CUDA Out of Memory loading `{hf_subpath}` "
                            f"(Free GPU VRAM right now: {stats_now.free_mb:.0f} MB / {stats_now.total_mb:.0f} MB).\n"
                            f"Error: {retry_err}\n"
                            "Actionable fixes:\n"
                            "  1. Switch quantization to `int4-weight-only` (~2.8 GB VRAM).\n"
                            "  2. Keep Codec Device on `CPU (Recommended for 8GB VRAM)`.\n"
                            "  3. Close GPU-heavy apps (games, extra Chrome/Edge tabs, local LLMs) and retry."
                        ) from retry_err
                else:
                    stats_now = self.get_vram_stats()
                    raise TTSEngineError(
                        f"CUDA Out of Memory loading `{hf_subpath}` "
                        f"(Free GPU VRAM right now: {stats_now.free_mb:.0f} MB / {stats_now.total_mb:.0f} MB).\n"
                        f"Error: {oom_err}\n"
                        "Actionable fixes:\n"
                        "  1. Switch quantization to `int4-weight-only` (~2.8 GB VRAM).\n"
                        "  2. Close other GPU-heavy Windows apps (games, browser hardware video, local LLMs) to free VRAM.\n"
                        "  3. Restart `run_windows.bat` so Windows WDDM clears fragmented GPU memory."
                    ) from oom_err
            except Exception as err:
                self._runtime = None
                self._active_variant = None
                self._clear_cuda_memory()
                raise TTSEngineError(
                    f"Failed to load `{hf_subpath}` with torchao BF16 on CUDA: {err}\n"
                    "Tip: Keep Codec Device on `CPU (Recommended for 8GB VRAM)` or try `int4-weight-only`."
                ) from err

            load_secs = time.perf_counter() - t0
            self._clear_cuda_memory()

        stats = self.get_vram_stats()
        return (
            f"Loaded `{HF_REPO_ID}/{variant}` (BF16 on CUDA, DACVAE codec on `{codec_dev.upper()}`) "
            f"in {load_secs:.1f}s on {stats.gpu_name}. "
            f"PyTorch VRAM: {stats.allocated_mb:.0f} MB | Free GPU VRAM: {stats.free_mb:.0f} MB / {stats.total_mb:.0f} MB."
        )

    # ------------------------------------------------------------------
    # Synthesis Core (Covers All 4 Irodori-TTS v4-Large Modes)
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_audio_and_sr(raw_result: Any) -> Tuple[np.ndarray, int]:
        """Extract 1-D float32 numpy waveform and sample rate from `InferenceRuntime.synthesize()` output."""
        sr = int(getattr(raw_result, "sample_rate", NATIVE_SAMPLE_RATE))

        audio_obj = getattr(raw_result, "audio", None)
        if audio_obj is None:
            audios_list = getattr(raw_result, "audios", None)
            if audios_list and len(audios_list) > 0:
                audio_obj = audios_list[0]
        if audio_obj is None and isinstance(raw_result, tuple) and len(raw_result) >= 2:
            audio_obj, sr = raw_result[0], int(raw_result[1])

        if audio_obj is None:
            raise TTSEngineError("TTS runtime returned an empty audio result.")

        if isinstance(audio_obj, torch.Tensor):
            arr = audio_obj.detach().float().cpu().numpy()
        else:
            arr = np.asarray(audio_obj, dtype=np.float32)

        arr = np.squeeze(arr)
        if arr.ndim == 2:
            # Handle (channels, samples) or (samples, channels)
            arr = arr.mean(axis=0) if arr.shape[0] <= 4 else arr.mean(axis=1)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        peak = float(np.max(np.abs(arr))) if arr.size > 0 else 0.0
        if peak > 0.99:
            arr = (arr / peak) * 0.98
        return arr, sr

    def synthesize(
        self,
        text: str,
        mode: str,
        ref_wav_path: Optional[str] = None,
        ref_transcript: str = "",
        caption: str = "",
        max_ref_seconds: float = 30.0,
        trim_ref_silence: bool = True,
        num_steps: int = 40,
        cfg_scale_text: float = 3.0,
        cfg_scale_speaker: float = 5.0,
        cfg_scale_caption: float = 3.0,
        seed: int = -1,
        speed: float = 1.0,
        trim_output_silence: bool = False,
        export_sample_rate: int = 48000,
        save_to_history: bool = True,
        output_filename_prefix: Optional[str] = None,
    ) -> SynthesisResult:
        """
        Unified synthesis entry point for:
          - `zero_shot`: Zero-Shot Voice Cloning & Long Reference Audio
          - `voice_design`: Text-based Voice Design (no reference audio)
          - `style_clone`: Style-Controlled Voice Cloning (reference audio + caption)
          - `batch`: Single line within Batch Generation
        """
        clean_text = (text or "").strip()
        if not clean_text:
            raise TTSEngineError("Please enter Japanese text to synthesize.")

        if self._runtime is None:
            raise TTSEngineError(
                "Model is not loaded on GPU yet. Click 'Load Model to GPU' in the top Model & VRAM panel first."
            )

        try:
            from irodori_tts.inference_runtime import SamplingRequest
        except ImportError as exc:
            raise TTSEngineError("`irodori_tts` package is unavailable.") from exc

        clean_caption = (caption or "").strip()
        clean_ref_transcript = (ref_transcript or "").strip()
        use_no_ref = mode == "voice_design"

        if mode in ("zero_shot", "style_clone") and not ref_wav_path:
            raise TTSEngineError(
                "Reference audio is required for Voice Cloning. "
                "Upload an audio file, record via microphone, or select a saved Voice Preset."
            )
        if mode == "voice_design" and not clean_caption:
            raise TTSEngineError(
                "Please provide a Voice Design description (e.g., gender, age, pitch, tone, speaking style)."
            )
        if mode == "style_clone" and not clean_caption:
            raise TTSEngineError(
                "Please enter a Style / Emotion Prompt to steer the cloned voice."
            )

        processed_ref: Optional[Path] = None
        if not use_no_ref and ref_wav_path:
            processed_ref, _ = preprocess_reference_audio(
                Path(ref_wav_path),
                max_seconds=max_ref_seconds,
                trim_ref_silence=trim_ref_silence,
                target_sr=NATIVE_SAMPLE_RATE,
                work_dir=self.temp_dir,
            )

        seed_used = int(seed) if int(seed) >= 0 else int(torch.randint(0, 2**31 - 1, (1,)).item())

        candidate_req: Dict[str, Any] = {
            "text": clean_text,
            "ref_wav": str(processed_ref) if processed_ref is not None else None,
            "no_ref": use_no_ref,
            "caption": clean_caption if clean_caption else None,
            "style_prompt": clean_caption if clean_caption else None,
            "ref_text": clean_ref_transcript if clean_ref_transcript else None,
            "ref_transcript": clean_ref_transcript if clean_ref_transcript else None,
            "max_ref_seconds": float(max_ref_seconds),
            "num_steps": int(num_steps),
            "cfg_scale_text": float(cfg_scale_text),
            "cfg_scale_speaker": 0.0 if use_no_ref else float(cfg_scale_speaker),
            "cfg_scale_caption": float(cfg_scale_caption) if clean_caption else 0.0,
            "seed": seed_used,
            "num_candidates": 1,
        }

        req_kwargs = self._build_filtered_kwargs(SamplingRequest, candidate_req)
        request = SamplingRequest(**req_kwargs)

        t0 = time.perf_counter()
        try:
            with self._lock:
                with torch.inference_mode():
                    raw_result = self._runtime.synthesize(request)
                # Keep 8GB VRAM clean between generations
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError as oom_err:
            self._clear_cuda_memory()
            raise TTSEngineError(
                "CUDA Out of Memory (8GB VRAM limit reached).\n"
                "Actionable fixes:\n"
                "  1. Shorten the target text or Long Reference Audio duration.\n"
                "  2. Switch to `int4-weight-only` (~2.8 GB VRAM) in Model & VRAM Management.\n"
                "  3. Close other GPU-heavy apps (games, browsers with WebGL, local LLMs)."
            ) from oom_err
        except Exception as exc:
            raise TTSEngineError(f"Synthesis failed: {exc}") from exc
        finally:
            if processed_ref is not None and processed_ref.exists():
                try:
                    processed_ref.unlink()
                except Exception:
                    pass

        elapsed_sec = time.perf_counter() - t0
        audio, native_sr = self._extract_audio_and_sr(raw_result)

        # Post-processing pipeline
        if trim_output_silence:
            audio = trim_silence(audio, native_sr, threshold_db=-42.0, pad_ms=80.0)
        if abs(float(speed) - 1.0) >= 0.01:
            audio = adjust_speed_wsola(audio, native_sr, float(speed))

        target_sr = int(export_sample_rate) if int(export_sample_rate) in (24000, 44100, 48000) else native_sr
        if target_sr != native_sr:
            audio = resample_audio(audio, native_sr, target_sr)

        duration_sec = float(audio.size) / float(target_sr)
        now = datetime.now()
        ts_file = now.strftime("%Y%m%d_%H%M%S")
        ts_display = now.strftime("%Y-%m-%d %H:%M:%S")

        prefix = output_filename_prefix or f"{ts_file}_{mode}"
        safe_prefix = re.sub(r"[^\w\-]+", "_", prefix).strip("_")
        out_path = self.outputs_dir / f"{safe_prefix}.wav"
        counter = 1
        while out_path.exists():
            out_path = self.outputs_dir / f"{safe_prefix}_{counter:02d}.wav"
            counter += 1

        sf.write(str(out_path), audio, target_sr, subtype="PCM_16")

        result = SynthesisResult(
            output_path=out_path,
            sample_rate=target_sr,
            duration_sec=duration_sec,
            elapsed_sec=elapsed_sec,
            mode=mode,
            text=clean_text,
            caption=clean_caption,
            seed_used=seed_used,
            timestamp=ts_display,
        )
        if save_to_history:
            self.history.add_entry(result)
        return result

    # ------------------------------------------------------------------
    # Tab D: Batch Synthesis (Multi-line / TXT / CSV -> Single ZIP)
    # ------------------------------------------------------------------
    @staticmethod
    def parse_batch_inputs(
        multiline_text: str,
        uploaded_file_path: Optional[str],
        default_caption: str = "",
    ) -> List[Dict[str, str]]:
        """
        Parse batch rows from an uploaded `.csv` / `.txt` file or the multi-line text area.
        Each item has keys: `text`, `caption`, `filename`.
        """
        items: List[Dict[str, str]] = []

        if uploaded_file_path:
            src = Path(uploaded_file_path)
            if not src.exists():
                raise TTSEngineError(f"Uploaded batch file not found: {src}")
            suffix = src.suffix.lower()
            raw_content = src.read_text(encoding="utf-8-sig")

            if suffix == ".csv":
                reader = csv.DictReader(raw_content.splitlines())
                if reader.fieldnames:
                    lower_map = {name.strip().lower(): name for name in reader.fieldnames if name}
                    text_col = lower_map.get("text") or lower_map.get("sentence") or lower_map.get("japanese")
                    caption_col = lower_map.get("caption") or lower_map.get("style")
                    name_col = lower_map.get("filename") or lower_map.get("id") or lower_map.get("name")

                    if text_col:
                        for idx, row in enumerate(reader, start=1):
                            line_text = (row.get(text_col) or "").strip()
                            if not line_text:
                                continue
                            line_caption = (row.get(caption_col) or "").strip() if caption_col else ""
                            line_name = (row.get(name_col) or "").strip() if name_col else f"line_{idx:03d}"
                            items.append(
                                {
                                    "text": line_text,
                                    "caption": line_caption or default_caption,
                                    "filename": line_name,
                                }
                            )
                    else:
                        # Headerless CSV fallback: use first column as text, optional second as caption
                        plain_reader = csv.reader(raw_content.splitlines())
                        for idx, row in enumerate(plain_reader, start=1):
                            if not row or not row[0].strip():
                                continue
                            items.append(
                                {
                                    "text": row[0].strip(),
                                    "caption": (row[1].strip() if len(row) > 1 and row[1].strip() else default_caption),
                                    "filename": f"line_{idx:03d}",
                                }
                            )
            else:
                for idx, line in enumerate(raw_content.splitlines(), start=1):
                    clean = line.strip()
                    if clean:
                        items.append(
                            {"text": clean, "caption": default_caption, "filename": f"line_{idx:03d}"}
                        )

        # Also accept multi-line text area if no file uploaded (or empty file)
        if not items and multiline_text and multiline_text.strip():
            for idx, line in enumerate(multiline_text.splitlines(), start=1):
                clean = line.strip()
                if clean:
                    items.append(
                        {"text": clean, "caption": default_caption, "filename": f"line_{idx:03d}"}
                    )

        if not items:
            raise TTSEngineError(
                "No Japanese sentences found for batch generation. "
                "Enter lines in the text area (one sentence per line) or upload a .txt / .csv file."
            )
        return items

    def synthesize_batch(
        self,
        items: List[Dict[str, str]],
        batch_mode: str,
        ref_wav_path: Optional[str] = None,
        ref_transcript: str = "",
        max_ref_seconds: float = 30.0,
        num_steps: int = 40,
        cfg_scale_text: float = 3.0,
        cfg_scale_speaker: float = 5.0,
        cfg_scale_caption: float = 3.0,
        seed: int = -1,
        speed: float = 1.0,
        trim_output_silence: bool = False,
        export_sample_rate: int = 48000,
        progress_cb: Optional[Callable[[float, str], None]] = None,
    ) -> Tuple[Path, List[SynthesisResult]]:
        """
        Sequentially synthesize all batch sentences and bundle every `.wav` + `manifest.csv`
        into a single downloadable `.zip` archive in `./outputs/`.
        """
        if self._runtime is None:
            raise TTSEngineError("Model is not loaded on GPU yet. Load the model first.")

        ts_batch = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_work_dir = self.temp_dir / f"batch_{ts_batch}"
        batch_work_dir.mkdir(parents=True, exist_ok=True)

        results: List[SynthesisResult] = []
        total = len(items)

        try:
            for idx, item in enumerate(items, start=1):
                if progress_cb is not None:
                    progress_cb((idx - 1) / total, f"Synthesizing line {idx}/{total}...")

                item_seed = (int(seed) + idx - 1) if int(seed) >= 0 else -1
                res = self.synthesize(
                    text=item["text"],
                    mode=batch_mode,
                    ref_wav_path=ref_wav_path,
                    ref_transcript=ref_transcript,
                    caption=item.get("caption", ""),
                    max_ref_seconds=max_ref_seconds,
                    num_steps=num_steps,
                    cfg_scale_text=cfg_scale_text,
                    cfg_scale_speaker=cfg_scale_speaker,
                    cfg_scale_caption=cfg_scale_caption,
                    seed=item_seed,
                    speed=speed,
                    trim_output_silence=trim_output_silence,
                    export_sample_rate=export_sample_rate,
                    save_to_history=(idx == 1),  # Record batch sample in history without flooding all 15 slots
                    output_filename_prefix=f"batch_{ts_batch}_{idx:03d}_{item['filename']}",
                )
                results.append(res)

            if progress_cb is not None:
                progress_cb(0.95, "Packaging WAV files into ZIP archive...")

            zip_path = self.outputs_dir / f"irodori_batch_{ts_batch}.zip"
            manifest_path = batch_work_dir / "manifest.csv"
            with manifest_path.open("w", encoding="utf-8-sig", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["index", "filename", "duration_sec", "sample_rate", "seed", "caption", "text"])
                for idx, res in enumerate(results, start=1):
                    writer.writerow(
                        [
                            idx,
                            res.output_path.name,
                            f"{res.duration_sec:.2f}",
                            res.sample_rate,
                            res.seed_used,
                            res.caption,
                            res.text,
                        ]
                    )

            with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                zf.write(manifest_path, arcname="manifest.csv")
                for res in results:
                    zf.write(res.output_path, arcname=res.output_path.name)

            if progress_cb is not None:
                progress_cb(1.0, f"Completed {total} clips!")

            return zip_path, results
        finally:
            shutil.rmtree(batch_work_dir, ignore_errors=True)
