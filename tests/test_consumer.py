import asyncio
import wave

import numpy as np
import pytest

from consume import configured_manifest, receive_utterance, save_wave


class Session:
    def __init__(self, outputs):
        self.audio = iter(payload for kind, payload in outputs if kind == "audio")
        self.data = iter(payload for kind, payload in outputs if kind == "data")
        self.marker_read = False

    def try_recv_audio(self):
        return next(self.audio, None) if self.marker_read else None

    def try_recv_data(self):
        self.marker_read = True
        return next(self.data, None)


async def test_consumer_drains_audio_even_when_marker_arrives_first(tmp_path):
    session = Session(
        [
            ("data", "<|audio_end|>"),
            ("audio", {"samples": [-1, 0, 0.5], "sample_rate": 22050, "channels": 1}),
            (
                "audio",
                {"samples": [32767 / 32768], "sample_rate": 22050, "channels": 1},
            ),
        ]
    )
    path = tmp_path / "received.wav"
    source = tmp_path / "source.wav"
    save_wave(source, 22050, np.array([-1, 0, 0.5, 32767 / 32768]))
    rate, samples = await receive_utterance(session, source)
    save_wave(path, rate, samples)
    with wave.open(str(path)) as wav:
        assert wav.getframerate() == 22050
        assert wav.getnchannels() == 1
        np.testing.assert_array_equal(
            np.frombuffer(wav.readframes(4), dtype="<i2"), [-32768, 0, 16384, 32767]
        )


async def test_consumer_rejects_error_and_missing_completion(tmp_path):
    with pytest.raises(RuntimeError, match="Synthesis failed"):
        await receive_utterance(
            Session([("data", '{"error":"bad input"}')]), tmp_path / "missing.wav"
        )
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(
            receive_utterance(Session([]), tmp_path / "missing.wav"), 0.02
        )


def test_manifest_paths_are_absolute(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    manifest = configured_manifest(
        tmp_path / "weights", tmp_path / "voice.wav", "cuda:1"
    )
    params = manifest["nodes"][0]["params"]
    assert params["model_dir"] == str(tmp_path / "weights")
    assert params["spk_audio_prompt"] == str(tmp_path / "voice.wav")
    assert params["__python_version__"] == "3.11"
    assert params["device"] == "cuda:1"
