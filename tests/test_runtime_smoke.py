"""Opt-in manifest resolution, automatic provisioning, runner, and IPC check."""

import asyncio
import ctypes
import os
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from consume import run_case


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("INDEXTTS25_RUNTIME_TEST") != "1",
    reason="set INDEXTTS25_RUNTIME_TEST=1 with a built source-plugin runtime",
)
async def test_managed_source_plugin_round_trip(tmp_path, monkeypatch):
    # Optional DLL search paths are for a local Windows FFI development build.
    handles = []
    if os.name == "nt":
        handles = [
            os.add_dll_directory(p)
            for p in os.environ.get("REMOTEMEDIA_TEST_DLL_DIR", "").split(os.pathsep)
            if p
        ]
    try:
        import remotemedia.runtime as runtime

        assert hasattr(runtime, "create_streaming_session"), "FFI runtime is missing"
        reference = tmp_path / "reference.wav"
        pid_file = tmp_path / "worker.pid"
        monkeypatch.setenv("INDEXTTS25_TEST_PID_FILE", str(pid_file))
        sf.write(reference, np.zeros(2205), 22050)
        manifest = {
            "version": "v1",
            "metadata": {"name": "indextts25-managed-contract"},
            "plugins": [{"path": str(Path(__file__).parent / "fixtures")}],
            "nodes": [
                {
                    "id": "tts",
                    "node_type": "IndexTTS25Node",
                    "params": {
                        "__python_version__": "3.11",
                        "spk_audio_prompt": str(reference),
                        "lang": "JA",
                        "duration_factor": 1.2,
                        "chunk_size": 2,
                        "emo_vector": [0, 0, 0.8, 0, 0, 0, 0, 0],
                    },
                }
            ],
            "connections": [],
        }
        rate, samples = await run_case(
            runtime, manifest, {"text": "<日本|ニホン>です。"}, 120
        )
        assert rate == 22050
        np.testing.assert_array_equal(samples, [-1, 0, 0.5, 32767 / 32768])
        pid = int(pid_file.read_text(encoding="ascii"))
        if os.name == "nt":
            # Check the actual interpreter, not just the venv launcher PID.
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.OpenProcess.restype = ctypes.c_void_p
            kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
            kernel.CloseHandle.argtypes = [ctypes.c_void_p]
            process = kernel.OpenProcess(0x00100000, False, pid)
            if process:
                try:
                    status = await asyncio.to_thread(
                        kernel.WaitForSingleObject, process, 10000
                    )
                    assert status == 0, "session.close() left the model worker running"
                finally:
                    kernel.CloseHandle(process)
            else:
                assert ctypes.get_last_error() == 87, "Unable to inspect worker exit"
    finally:
        for handle in handles:
            handle.close()
