import asyncio
import copy
import json
import threading
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from remotemedia.core.multiprocessing import NodeConfig
from remotemedia.core.multiprocessing.data import RuntimeData

from indextts25_node import AUDIO_END, IndexTTS25Node


class FakeModel:
    def __init__(self):
        self.calls = []
        self.pcm = np.array([-32768, -16384, 0, 16384, 32767], dtype=np.int16)
        self.result = (22050, self.pcm[:, None])
        self.error = None

    def infer(self, **kwargs):
        self.calls.append(copy.deepcopy(kwargs))
        if self.error:
            raise self.error
        return self.result


@pytest.fixture
def node(tmp_path, monkeypatch):
    reference = tmp_path / "prompt.wav"
    sf.write(reference, np.zeros(2205), 22050)
    instance = IndexTTS25Node(
        NodeConfig(
            "tts-test",
            "IndexTTS25Node",
            {
                "spk_audio_prompt": str(reference),
                "chunk_size": 2,
            },
        )
    )
    instance.fake = FakeModel()
    instance.load_count = 0

    def load():
        instance.load_count += 1
        instance._model = instance.fake

    monkeypatch.setattr(instance, "_load_model", load)
    return instance


async def collect(node, request="Hello"):
    return [frame async for frame in node.process(request)]


def error(frames):
    assert len(frames) == 1
    assert frames[0].is_text()
    result = json.loads(frames[0].as_text())
    assert set(result) == {"error", "input", "processed_by"}
    return result["error"]


async def test_signed_pcm_frames_reconstruct_complete_audio(node):
    frames = await collect(node)
    assert [len(frame.payload) for frame in frames[:-1]] == [2, 2, 1]
    assert frames[-1].as_text() == AUDIO_END
    for frame in frames[:-1]:
        assert frame.is_audio()
        assert frame.metadata.sample_rate == 22050
        assert frame.metadata.channels == 1
        assert frame.payload.dtype == np.float32
        assert frame.payload.flags.c_contiguous
    samples = np.concatenate([frame.payload for frame in frames[:-1]])
    np.testing.assert_array_equal(samples, node.fake.pcm.astype(np.float32) / 32768)
    call = node.fake.calls[0]
    assert call["output_path"] is None
    assert call["stream_return"] is False
    assert call["lang"] == "EN"


async def test_request_shapes_and_no_mutation(node):
    request = {
        "text": "  Hello 世界!  ",
        "lang": "ja",
        "duration_factor": 1.2,
        "emo_vector": [0, 0, 0.8, 0, 0, 0, 0, 0],
    }
    original = copy.deepcopy(request)
    for data in (
        request,
        RuntimeData.json(request),
        RuntimeData.text(json.dumps(request)),
        json.dumps(request),
    ):
        frames = await collect(node, data)
        assert frames[-1].as_text() == AUDIO_END
        call = node.fake.calls[-1]
        assert call["text"] == request["text"]
        assert call["emo_vector"] == request["emo_vector"]
        assert call["lang"] == "JA"
        assert call["duration_factor"] == 1.2
    assert request == original
    await collect(node, RuntimeData.text("Next"))
    assert node.fake.calls[-1]["lang"] == "EN"
    assert node.fake.calls[-1]["duration_factor"] == 1.0
    assert node.load_count == 1


async def test_emotion_default_null_and_zero_are_distinct(node):
    vector = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    node._defaults["emo_vector"] = vector
    await collect(node, {"text": "Null", "emo_vector": None})
    assert node.fake.calls[-1]["emo_vector"] is None
    await collect(node, {"text": "Zeros", "emo_vector": [0] * 8})
    assert node.fake.calls[-1]["emo_vector"] == [0] * 8
    await collect(node)
    assert node.fake.calls[-1]["emo_vector"] == vector
    assert node._defaults["emo_vector"] == vector


@pytest.mark.parametrize(
    "text,lang",
    [
        ("他在银<行|XING2>里<行|HANG2>走了半天。", "ZH"),
        ("I am <going|G OW1 . IH0 NG> home.", "EN"),
        ("<日本|ニホン>です。", "JA"),
        ("Hola, bienvenidos.", "ES"),
        ("مرحبا بكم", "AR"),
    ],
)
async def test_annotations_and_all_languages_are_forwarded_exactly(node, text, lang):
    await collect(node, {"text": text, "lang": lang})
    assert node.fake.calls[-1]["text"] == text
    assert node.fake.calls[-1]["lang"] == lang


@pytest.mark.parametrize("factor", [0.5, 1, 1.2, 2.0])
async def test_duration_endpoints_and_direction(node, factor):
    await collect(node, {"text": "Duration", "duration_factor": factor})
    assert node.fake.calls[-1]["duration_factor"] == factor


@pytest.mark.parametrize(
    "payload,field",
    [
        ("", "text"),
        ("  ", "text"),
        ({}, "text"),
        ({"text": 7}, "text"),
        (42, "input"),
        ([], "input"),
        ('{"text":', "JSON"),
        ({"text": "Hi", "unknown": True}, "unknown"),
        ({"text": "Hi", "lang": "DE"}, "lang"),
        ({"text": "Hi", "lang": None}, "lang"),
        ({"text": "Hi", "duration_factor": 0.49}, "duration_factor"),
        ({"text": "Hi", "duration_factor": 2.01}, "duration_factor"),
        ({"text": "Hi", "duration_factor": True}, "duration_factor"),
        ({"text": "Hi", "duration_factor": "1.2"}, "duration_factor"),
        ({"text": "Hi", "duration_factor": float("nan")}, "duration_factor"),
        ({"text": "Hi", "duration_factor": float("inf")}, "duration_factor"),
        ({"text": "Hi", "duration_factor": 10**400}, "duration_factor"),
        ({"text": "Hi", "emo_vector": [0] * 7}, "emo_vector"),
        ({"text": "Hi", "emo_vector": [0] * 9}, "emo_vector"),
        ({"text": "Hi", "emo_vector": [True] * 8}, "emo_vector"),
        ({"text": "Hi", "emo_vector": ["0"] * 8}, "emo_vector"),
        ({"text": "Hi", "emo_vector": [-0.1] * 8}, "emo_vector"),
        ({"text": "Hi", "emo_vector": [1.1] * 8}, "emo_vector"),
        ({"text": "Hi", "emo_vector": [float("inf")] * 8}, "emo_vector"),
        ({"text": "Hi", "emo_vector": [float("nan")] * 8}, "emo_vector"),
        ({"text": "Hi", "spk_audio_prompt": None}, "spk_audio_prompt"),
        ({"text": "Hi", "spk_audio_prompt": "missing.wav"}, "spk_audio_prompt"),
        ({"text": "Hi", "output_path": None}, "output_path"),
    ],
)
async def test_invalid_requests_never_invoke_model(node, payload, field):
    assert field in error(await collect(node, payload))
    assert not node.fake.calls
    assert node.load_count == 0


async def test_unreadable_reference_is_rejected_before_inference(node, tmp_path):
    path = tmp_path / "bad.wav"
    path.write_text("not audio")
    frames = await collect(node, {"text": "Hi", "spk_audio_prompt": str(path)})
    assert "spk_audio_prompt" in error(frames)
    assert not node.fake.calls


async def test_voice_override_is_request_local(node, tmp_path):
    reference = tmp_path / "other.wav"
    sf.write(reference, np.zeros(500), 16000)
    await collect(node, {"text": "Other", "spk_audio_prompt": str(reference)})
    assert node.fake.calls[-1]["spk_audio_prompt"] == str(reference.resolve())
    await collect(node)
    assert node.fake.calls[-1]["spk_audio_prompt"] == node._defaults["spk_audio_prompt"]


async def test_output_wav_matches_audio_and_can_be_replaced(node, tmp_path):
    path = tmp_path / "new" / "output.wav"
    for _ in range(2):
        frames = await collect(node, {"text": "Export", "output_path": str(path)})
        saved, rate = sf.read(path, dtype="int16")
        np.testing.assert_array_equal(saved, node.fake.pcm)
        assert rate == 22050
        assert frames[-1].as_text() == AUDIO_END
    assert sf.info(path).subtype == "PCM_16"


async def test_default_does_not_write_output(node, tmp_path):
    before = set(tmp_path.rglob("*"))
    await collect(node)
    assert set(tmp_path.rglob("*")) == before


async def test_output_failure_has_no_audio_or_success(node, tmp_path):
    frames = await collect(node, {"text": "Export", "output_path": str(tmp_path)})
    assert error(frames)


@pytest.mark.parametrize(
    "result",
    [
        None,
        (22050, np.empty(0, dtype=np.int16)),
        (22050, np.zeros((3, 2), dtype=np.int16)),
        (22050, np.zeros(3, dtype=np.float32)),
        (0, np.zeros(3, dtype=np.int16)),
    ],
)
async def test_invalid_upstream_output(node, result):
    node.fake.result = result
    assert "IndexTTS" in error(await collect(node))


async def test_model_failure_is_recoverable_but_cuda_failure_propagates(node):
    node.fake.error = ValueError("inference failed")
    assert "inference failed" in error(await collect(node))
    node.fake.error = None
    assert (await collect(node))[-1].as_text() == AUDIO_END
    node.fake.error = RuntimeError("CUDA error: device-side assert triggered")
    with pytest.raises(RuntimeError, match="CUDA"):
        await collect(node)


async def test_initialization_failure_propagates(node, monkeypatch):
    def fail():
        raise FileNotFoundError("checkpoint missing")

    monkeypatch.setattr(node, "_load_model", fail)
    with pytest.raises(FileNotFoundError, match="checkpoint"):
        await collect(node)


async def test_concurrent_calls_serialize_and_leave_loop_responsive(node, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    original = node.fake.infer

    def infer(**kwargs):
        entered.set()
        assert release.wait(5)
        return original(**kwargs)

    monkeypatch.setattr(node.fake, "infer", infer)
    first = asyncio.create_task(collect(node, "First"))
    try:
        assert await asyncio.to_thread(entered.wait, 3)
        second = asyncio.create_task(collect(node, "Second"))
        await asyncio.sleep(0.01)
        assert not second.done()
        assert node.load_count == 1
    finally:
        release.set()
    await asyncio.gather(first, second)
    assert [call["text"] for call in node.fake.calls] == ["First", "Second"]


async def test_repeated_cancellation_drains_worker_before_cleanup(node, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    original = node.fake.infer

    def infer(**kwargs):
        entered.set()
        assert release.wait(5)
        return original(**kwargs)

    monkeypatch.setattr(node.fake, "infer", infer)
    task = asyncio.create_task(collect(node))
    try:
        assert await asyncio.to_thread(entered.wait, 3)
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        cleanup = asyncio.create_task(node.cleanup())
        await asyncio.sleep(0.01)
        assert not task.done()
        assert not cleanup.done()
        assert node._model is node.fake
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    await cleanup
    await node.cleanup()
    assert node._model is None
    with pytest.raises(RuntimeError, match="cleaned up"):
        await collect(node)


async def test_ui_channel_text_passes_through_without_synthesis(node):
    frame = RuntimeData.text("Display this markdown")
    frame.metadata.channel = "ui"
    assert await collect(node, frame) == [frame]
    assert not node.fake.calls


@pytest.mark.parametrize(
    "params,field",
    [
        ({"chunk_size": 0}, "chunk_size"),
        ({"chunk_size": True}, "chunk_size"),
        ({"use_bf16": "true"}, "use_bf16"),
        ({"device": 5}, "device"),
    ],
)
def test_invalid_configuration(params, field):
    with pytest.raises(ValueError, match=field):
        IndexTTS25Node({"params": params})


def test_constructor_and_config_paths(tmp_path):
    with pytest.raises(TypeError, match="NodeConfig"):
        IndexTTS25Node("runner-id")
    config = NodeConfig(
        "configured",
        "IndexTTS25Node",
        {
            "model_dir": str(tmp_path),
            "lang": "ar",
            "use_bf16": False,
            "duration_factor": 1.2,
        },
    )
    node = IndexTTS25Node(config)
    assert node.node_id == "configured"
    assert node.cfg_path == Path(tmp_path) / "config.yaml"
    assert node.use_bf16 is False
    assert node._defaults["lang"] == "AR"
