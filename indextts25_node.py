"""IndexTTS 2.5 source plugin: complete text requests to mono SDK audio."""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import math
import sys
from collections.abc import AsyncGenerator, Callable
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np
from remotemedia.core.multiprocessing import (
    MultiprocessNode,
    NodeConfig,
    python_requires,
    register_node,
)
from remotemedia.core.multiprocessing.data import RuntimeData

UPSTREAM_REVISION = "d9e41aac89fd00b3d71497fddb287b7f24613712"
EMOTIONS = (
    "happy",
    "angry",
    "sad",
    "afraid",
    "disgusted",
    "melancholic",
    "surprised",
    "calm",
)
LANGUAGES = frozenset({"ZH", "EN", "JA", "ES", "AR"})
REQUEST_FIELDS = frozenset(
    {"text", "spk_audio_prompt", "lang", "emo_vector", "duration_factor", "output_path"}
)
AUDIO_END = "<|audio_end|>"
logger = logging.getLogger(__name__)


def _number(value: Any, field: str, minimum: float, maximum: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not minimum <= value <= maximum
        or not math.isfinite(value)
    ):
        raise ValueError(f"{field} must be a finite number in [{minimum}, {maximum}]")
    return float(value)


def _controls(request: dict[str, Any]) -> None:
    """Validate and normalize a private copy of the synthesis controls."""
    lang = request["lang"]
    if not isinstance(lang, str) or lang.upper() not in LANGUAGES:
        raise ValueError("lang must be ZH, EN, JA, ES, or AR")
    request["lang"] = lang.upper()
    request["duration_factor"] = _number(
        request["duration_factor"], "duration_factor", 0.5, 2.0
    )
    vector = request["emo_vector"]
    if vector is not None:
        if not isinstance(vector, (list, tuple)) or len(vector) != len(EMOTIONS):
            raise ValueError("emo_vector must contain exactly eight numbers")
        request["emo_vector"] = [
            _number(value, f"emo_vector[{index}] ({EMOTIONS[index]})", 0, 1)
            for index, value in enumerate(vector)
        ]


def _path(value: Any, field: str) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError(f"{field} must be a nonempty worker-local file path")
    return Path(value).expanduser().resolve()


async def _finish_thread(function: Callable[..., Any], *args: Any) -> Any:
    """Do not release the model lock until a cancelled worker has finished."""
    task = asyncio.create_task(asyncio.to_thread(function, *args))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        # Retrieve any failure so it cannot become an unhandled-task warning.
        if not task.cancelled():
            task.exception()
        raise


def _decode_pcm(result: Any) -> tuple[int, np.ndarray]:
    if not isinstance(result, tuple) or len(result) != 2:
        raise ValueError("IndexTTS returned no (sample_rate, waveform) result")
    sample_rate, pcm = result
    if (
        isinstance(sample_rate, bool)
        or not isinstance(sample_rate, (int, np.integer))
        or sample_rate <= 0
    ):
        raise ValueError("IndexTTS returned an invalid sample rate")
    pcm = np.asarray(pcm)
    if pcm.ndim == 2 and pcm.shape[1] == 1:
        pcm = pcm[:, 0]
    if pcm.ndim != 1 or pcm.size == 0 or pcm.dtype != np.int16:
        raise ValueError("IndexTTS must return nonempty mono PCM16 audio")
    return int(sample_rate), np.ascontiguousarray(pcm, dtype=np.float32) / 32768.0


@register_node("IndexTTS25Node")
@python_requires(
    [
        "accelerate==1.8.1",
        "cn2an==0.5.22",
        "cython==3.0.7",
        "einops>=0.8.1",
        "ffmpeg-python==0.2.0",
        "fugashi>=1.2.0",
        "unidic-lite>=1.0.0",
        "g2p-en==2.1.0",
        "jieba==0.42.1",
        "json5==0.10.0",
        "keras==2.9.0",
        "librosa==0.10.2.post1",
        "matplotlib==3.10.0",
        "modelscope==1.27.0",
        "munch==4.0.0",
        "numba==0.63.0",
        "numpy==2.2.6",
        "omegaconf>=2.3.0",
        "opencv-python==4.9.0.80",
        "pandas==2.3.2",
        "safetensors==0.5.2",
        "sentencepiece>=0.2.1",
        "tensorboard==2.20.0",
        "textstat>=0.7.10",
        "tokenizers==0.21.0",
        "requests>=2.28",
        "tqdm>=4.67.1",
        "transformers==4.52.1",
        "openai-whisper>=20231117",
        "fugashi[unidic-lite]",
        "wetext>=0.0.9; sys_platform != 'linux'",
        "WeTextProcessing; sys_platform == 'linux'",
        "torch @ https://download.pytorch.org/whl/cu128/torch-2.8.0%2Bcu128-cp311-cp311-win_amd64.whl ; python_version == '3.11' and sys_platform == 'win32'",
        "torch @ https://download.pytorch.org/whl/cu128/torch-2.8.0%2Bcu128-cp311-cp311-manylinux_2_28_x86_64.whl ; python_version == '3.11' and sys_platform == 'linux' and platform_machine == 'x86_64'",
        "torch==2.8.0 ; python_version != '3.11' or (sys_platform != 'win32' and (sys_platform != 'linux' or platform_machine != 'x86_64'))",
        "torchaudio @ https://download.pytorch.org/whl/cu128/torchaudio-2.8.0%2Bcu128-cp311-cp311-win_amd64.whl ; python_version == '3.11' and sys_platform == 'win32'",
        "torchaudio @ https://download.pytorch.org/whl/cu128/torchaudio-2.8.0%2Bcu128-cp311-cp311-manylinux_2_28_x86_64.whl ; python_version == '3.11' and sys_platform == 'linux' and platform_machine == 'x86_64'",
        "torchaudio==2.8.0 ; python_version != '3.11' or (sys_platform != 'win32' and (sys_platform != 'linux' or platform_machine != 'x86_64'))",
        "soundfile>=0.13.1,<1",
        "argbind",
        "pyloudnorm",
        "importlib-resources",
        "scipy",
        "julius",
        "ffmpy",
        "ipython",
        "rich",
        "pystoi",
        "torch_stoi",
        "flatten-dict",
        "markdown2",
        "randomname",
        "protobuf>=6.31.1",
    ]
)
class IndexTTS25Node(MultiprocessNode):
    """Clone a reference voice with language, emotion and duration controls.

    Args:
        config: Full runner configuration, or a mapping with a ``params`` member.
        **kwargs: Additional base-node configuration.
    """

    def __init__(self, config: NodeConfig | dict | None = None, **kwargs: Any) -> None:
        if isinstance(config, str):
            raise TypeError("IndexTTS25Node requires NodeConfig with manifest params")
        if config is None:
            config = NodeConfig("indextts25", "IndexTTS25Node", {})
        elif isinstance(config, dict):
            config = NodeConfig(
                config.get("node_id", "indextts25"),
                config.get("node_type", "IndexTTS25Node"),
                config.get("params", {}),
                session_id=config.get("session_id"),
            )
        params = copy.deepcopy(config.params)
        super().__init__(config, **kwargs)
        self.is_streaming = True
        self.model_dir = _path(params.get("model_dir", "checkpoints"), "model_dir")
        self.cfg_path = _path(
            params.get("cfg_path", self.model_dir / "config.yaml"), "cfg_path"
        )
        self.device = params.get("device")
        if self.device is not None and (
            not isinstance(self.device, str) or not self.device.strip()
        ):
            raise ValueError("device must be a nonempty string or null")
        self.use_bf16 = params.get("use_bf16", True)
        if not isinstance(self.use_bf16, bool):
            raise ValueError("use_bf16 must be a boolean")
        self.chunk_size = params.get("chunk_size", 4096)
        if type(self.chunk_size) is not int or self.chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer")
        self._defaults = {
            "spk_audio_prompt": params.get("spk_audio_prompt"),
            "lang": params.get("lang", "EN"),
            "emo_vector": params.get("emo_vector"),
            "duration_factor": params.get("duration_factor", 1.0),
        }
        _controls(self._defaults)
        self._model: Any = None
        self._model_lock = asyncio.Lock()
        self._closed = False

    def _load_model(self) -> None:
        if not (3, 10) <= sys.version_info[:2] < (3, 12):
            raise RuntimeError(
                "IndexTTS 2.5 requires Python 3.10-3.11; set "
                "params.__python_version__ to '3.11' in the plugin manifest"
            )
        if not self.model_dir.is_dir() or not self.cfg_path.is_file():
            raise FileNotFoundError(
                f"IndexTTS 2.5 checkpoints missing: model_dir={self.model_dir}, "
                f"cfg_path={self.cfg_path}; download IndexTeam/IndexTTS-2.5 first"
            )
        try:
            vendor_root = str(Path(__file__).resolve().parent / "_vendor")
            if vendor_root not in sys.path:
                sys.path.insert(0, vendor_root)
            from indextts.infer_v2_5 import IndexTTS2
        except ImportError as exc:
            raise RuntimeError(
                "The managed plugin environment is missing an IndexTTS dependency; "
                "recreate it from plugin.toml using Python 3.11"
            ) from exc
        self._model = IndexTTS2(
            cfg_path=str(self.cfg_path),
            model_dir=str(self.model_dir),
            device=self.device,
            use_bf16=self.use_bf16,
            use_cuda_kernel=False,
            use_deepspeed=False,
            use_accel=False,
            use_torch_compile=False,
            use_qwen_emo=False,
        )

    async def _initialize_locked(self) -> None:
        if self._closed:
            raise RuntimeError("IndexTTS25Node has been cleaned up")
        if self._model is None:
            await _finish_thread(self._load_model)

    async def initialize(self) -> None:
        """Load the model once, outside the event loop."""
        async with self._model_lock:
            await self._initialize_locked()

    async def cleanup(self) -> None:
        """Wait for in-flight inference before releasing the model."""
        async with self._model_lock:
            self._model = None
            self._closed = True

    def _request(self, data: Any) -> dict[str, Any]:
        if isinstance(data, RuntimeData):
            if data.is_text():
                data = data.as_text()
            elif data.is_json():
                data = data.payload
        if isinstance(data, str):
            if data.lstrip().startswith("{"):
                try:
                    data = json.loads(data)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON request: {exc.msg}") from exc
            else:
                data = {"text": data}
        if not isinstance(data, dict):
            raise ValueError("input must be text or a JSON request object")
        unknown = data.keys() - REQUEST_FIELDS
        if unknown:
            raise ValueError(f"unknown request fields: {', '.join(map(str, unknown))}")
        request = copy.deepcopy(self._defaults)
        request.update(copy.deepcopy(data))
        text = request.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a nonblank string")
        _controls(request)
        reference = _path(request["spk_audio_prompt"], "spk_audio_prompt")
        if not reference.is_file():
            raise ValueError(f"spk_audio_prompt is not a readable file: {reference}")
        request["spk_audio_prompt"] = str(reference)
        if "output_path" in request:
            output = _path(request["output_path"], "output_path")
            if output == reference or output == self.cfg_path:
                raise ValueError(
                    "output_path must not overwrite the reference or config"
                )
            request["output_path"] = str(output)
        return request

    def _synthesize(self, request: dict[str, Any]) -> tuple[int, np.ndarray]:
        import soundfile as sf

        try:
            info = sf.info(request["spk_audio_prompt"])
            if info.frames == 0:
                raise ValueError("reference audio is empty")
        except (OSError, RuntimeError, ValueError) as exc:
            raise ValueError(f"spk_audio_prompt is not readable audio: {exc}") from exc
        args = dict(request)
        output = args.pop("output_path", None)
        result = self._model.infer(**args, output_path=None, stream_return=False)
        sample_rate, audio = _decode_pcm(result)
        if output is not None:
            Path(output).parent.mkdir(parents=True, exist_ok=True)
            sf.write(output, audio, sample_rate, format="WAV", subtype="PCM_16")
        return sample_rate, audio

    def _error(self, exc: Exception, data: Any) -> RuntimeData:
        if isinstance(data, RuntimeData):
            data = data.payload
        logger.warning("[%s] IndexTTS request failed: %s", self.node_id, exc)
        return RuntimeData.text(
            json.dumps(
                {"error": str(exc), "input": data, "processed_by": self.node_id},
                ensure_ascii=False,
                default=str,
            )
        )

    async def process(self, data: Any) -> AsyncGenerator[RuntimeData, None]:
        """Synthesize one utterance, then yield bounded audio frames and a marker."""
        if isinstance(data, RuntimeData) and data.is_text():
            if getattr(data.metadata, "channel", "tts") == "ui":
                yield data
                return
        try:
            request = self._request(data)
        except (ValueError, TypeError, OSError) as exc:
            yield self._error(exc, data)
            return
        async with self._model_lock:
            # Initialization failures must reach the runner, not become request errors.
            await self._initialize_locked()
            try:
                sample_rate, audio = await _finish_thread(self._synthesize, request)
            except Exception as exc:
                if isinstance(exc, RuntimeError) and "cuda" in str(exc).lower():
                    raise
                yield self._error(exc, data)
                return
            for start in range(0, audio.size, self.chunk_size):
                await asyncio.sleep(0)
                yield RuntimeData.audio(
                    audio[start : start + self.chunk_size], sample_rate, channels=1
                )
            await asyncio.sleep(0)
            yield RuntimeData.text(AUDIO_END)


register_node(f"{IndexTTS25Node.__module__}.{IndexTTS25Node.__name__}")(IndexTTS25Node)
