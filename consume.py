"""Run IndexTTS through the manifest resolver and save received pipeline audio."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import tempfile
import time
import wave
from pathlib import Path
from typing import Any

import numpy as np

PLUGIN = Path(__file__).resolve().parent
CASES = {
    "clone": {"text": "Hello, this is a voice cloning demo.", "lang": "EN"},
    "emotion": {
        "text": "快躲起来！是他要来了！",
        "lang": "ZH",
        "emo_vector": [0, 0, 0.8, 0, 0, 0, 0, 0],
    },
    "pinyin": {"text": "他在银<行|XING2>里<行|HANG2>走了半天。", "lang": "ZH"},
    "cmu": {"text": "I am <going|G OW1 . IH0 NG> home.", "lang": "EN"},
    "kana": {"text": "<日本|ニホン>です。", "lang": "JA"},
    "spanish": {"text": "Hola, bienvenidos a esta demostración.", "lang": "ES"},
    "arabic": {"text": "مرحبا بكم في هذا العرض.", "lang": "AR"},
    "fast": {
        "text": "大家好，欢迎来到IndexTTS。",
        "lang": "ZH",
        "duration_factor": 0.5,
    },
    "normal": {
        "text": "大家好，欢迎来到IndexTTS。",
        "lang": "ZH",
        "duration_factor": 1.0,
    },
    "slow": {
        "text": "大家好，欢迎来到IndexTTS。",
        "lang": "ZH",
        "duration_factor": 1.2,
    },
}


def configured_manifest(model_dir: Path, reference: Path, device: str | None) -> dict:
    """Use absolute plugin and worker-local asset paths, independent of cwd."""
    manifest = json.loads((PLUGIN / "manifest.json").read_text(encoding="utf-8"))
    manifest["plugins"] = [{"path": str(PLUGIN)}]
    params = manifest["nodes"][0]["params"]
    params.update(
        {
            "model_dir": str(model_dir.expanduser().resolve()),
            "spk_audio_prompt": str(reference.expanduser().resolve()),
            "device": device,
        }
    )
    return manifest


async def receive_utterance(session: Any, source_wav: Path) -> tuple[int, np.ndarray]:
    """Drain until the marker and every exported sample have crossed IPC.

    The runtime's audio/data queues can reorder relative to each other and stay
    open after input completion. The explicitly requested worker-local WAV gives
    this local smoke consumer an exact count, without guessing from silence.
    """
    chunks = []
    sample_rate = None
    markers = 0
    received = 0
    expected_pcm = None
    while True:
        # Polling avoids retaining a native receive lock across cancellation.
        for _ in range(64):
            payload = session.try_recv_audio()
            if payload is None:
                break
            if not isinstance(payload, dict) or payload.get("channels") != 1:
                raise RuntimeError("Expected mono SDK audio output")
            rate = payload["sample_rate"]
            if sample_rate is not None and rate != sample_rate:
                raise RuntimeError("Sample rate changed within an utterance")
            sample_rate = rate
            chunk = np.asarray(payload["samples"], dtype=np.float32)
            chunks.append(chunk)
            received += chunk.size
        payload = session.try_recv_data()
        if payload == "<|audio_end|>":
            markers += 1
            if markers != 1:
                raise RuntimeError("Duplicate audio completion marker")
            with wave.open(str(source_wav), "rb") as wav:
                expected_rate = wav.getframerate()
                if wav.getnchannels() != 1 or wav.getsampwidth() != 2:
                    raise RuntimeError("Expected a mono PCM16 worker export")
                expected_pcm = np.frombuffer(
                    wav.readframes(wav.getnframes()), dtype="<i2"
                )
        elif payload is not None:
            raise RuntimeError(
                f"Synthesis failed or returned unexpected output: {payload}"
            )
        if expected_pcm is not None and received >= expected_pcm.size:
            if (
                not chunks
                or received != expected_pcm.size
                or sample_rate != expected_rate
            ):
                raise RuntimeError("Pipeline audio does not match the worker export")
            audio = np.concatenate(chunks)
            if not np.array_equal(audio, expected_pcm.astype(np.float32) / 32768):
                raise RuntimeError("Pipeline waveform differs from the worker export")
            return sample_rate, audio
        await asyncio.sleep(0.01)


def save_wave(path: Path, sample_rate: int, audio: np.ndarray) -> None:
    """Write received float32 samples as PCM16 without an extra host dependency."""
    if audio.size == 0 or not np.isfinite(audio).all():
        raise RuntimeError("Received empty or nonfinite audio")
    pcm = np.clip(np.rint(audio * 32768), -32768, 32767).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(pcm.tobytes())


async def run_case(
    runtime: Any, manifest: dict, request: dict, timeout: float
) -> tuple:
    # One input per session keeps this local smoke's export and output associated.
    # Model reuse within a node is tested separately.
    session = await runtime.create_streaming_session(json.dumps(manifest))
    try:
        with tempfile.TemporaryDirectory(prefix="indextts25-export-") as directory:
            payload = dict(request)
            source_wav = Path(
                payload.get("output_path", Path(directory) / "source.wav")
            )
            payload["output_path"] = str(source_wav.resolve())
            await session.send_input(
                {"type": "text", "data": json.dumps(payload, ensure_ascii=False)}
            )
            await session.signal_input_complete()
            return await asyncio.wait_for(
                receive_utterance(session, source_wav), timeout
            )
    finally:
        await session.close()


async def main_async(args: argparse.Namespace) -> None:
    import remotemedia.runtime as runtime

    if not hasattr(runtime, "create_streaming_session"):
        raise RuntimeError(
            "A built RemoteMedia FFI runtime with python-source-plugin support is required"
        )
    manifest = configured_manifest(args.model_dir, args.reference, args.device)
    cases = CASES if args.suite else {args.case: CASES[args.case]}
    if args.request:
        cases = {"custom": json.loads(args.request.read_text(encoding="utf-8"))}
    results = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, request in cases.items():
        started = time.monotonic()
        record = {"case": name, "request": request, "status": "running"}
        results.append(record)
        try:
            rate, audio = await run_case(runtime, manifest, request, args.timeout)
            path = args.output_dir / f"{name}.wav"
            save_wave(path, rate, audio)
            record.update(
                status="passed",
                sample_rate=rate,
                samples=len(audio),
                duration_seconds=len(audio) / rate,
                wav=str(path.resolve()),
            )
            print(
                f"{name}: {len(audio) / rate:.2f}s at {rate} Hz -> {path}", flush=True
            )
        except Exception as exc:
            record.update(status="failed", error=str(exc))
            raise
        finally:
            record["elapsed_seconds"] = time.monotonic() - started
            report = {
                "manifest": manifest,
                "results": results,
                "evidence": "source-plugin pipeline; output has not been auditioned",
            }
            (args.output_dir / "results.json").write_text(
                json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--case", choices=list(CASES), default="clone")
    parser.add_argument(
        "--suite", action="store_true", help="Run all documented feature cases"
    )
    parser.add_argument(
        "--request", type=Path, help="Read a custom JSON synthesis request"
    )
    parser.add_argument("--output-dir", type=Path, default=PLUGIN / "outputs")
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    os.environ.setdefault("REMOTEMEDIA_NODE_TIMEOUT_MS", "600000")
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
