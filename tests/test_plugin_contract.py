import ast
import hashlib
import json
import subprocess
import sys
import tomllib
from pathlib import Path

from remotemedia.core.multiprocessing import NodeConfig, get_node_class

from indextts25_node import UPSTREAM_REVISION, IndexTTS25Node

PLUGIN = Path(__file__).resolve().parents[1]


def test_metadata_decorator_and_manifest_agree():
    metadata = tomllib.loads((PLUGIN / "plugin.toml").read_text())
    assert metadata["plugin"]["language"] == "python"
    assert metadata["python"]["entry_module"] == "indextts25_node"
    assert metadata["python"]["node_types"] == ["IndexTTS25Node"]
    assert metadata["python"]["requires"] == IndexTTS25Node.__python_requires__
    tree = ast.parse((PLUGIN / "indextts25_node.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    decorator = next(
        d
        for d in cls.decorator_list
        if isinstance(d, ast.Call) and d.func.id == "python_requires"
    )
    assert ast.literal_eval(decorator.args[0]) == metadata["python"]["requires"]
    manifest = json.loads((PLUGIN / "manifest.json").read_text())
    assert manifest["nodes"][0]["params"]["__python_version__"] == "3.11"
    assert get_node_class("IndexTTS25Node") is IndexTTS25Node
    assert get_node_class("indextts25_node.IndexTTS25Node") is IndexTTS25Node


def test_vendored_source_matches_recorded_revision():
    vendor = PLUGIN / "_vendor"
    record = json.loads((vendor / "manifest.json").read_text())
    assert record["revision"] == UPSTREAM_REVISION
    for name, digest in record["sha256"].items():
        assert hashlib.sha256((vendor / name).read_bytes()).hexdigest() == digest, name
    assert "indextts/infer_v2_5.py" in record["sha256"]
    assert (vendor / "LICENSE").is_file()
    audio_record = json.loads((vendor / "audiotools-manifest.json").read_text())
    assert audio_record["revision"] == "348ebf2034ce24e2a91a553e3171cb00c0c71678"
    for name, digest in audio_record["sha256"].items():
        assert hashlib.sha256((vendor / name).read_bytes()).hexdigest() == digest, name
    assert (vendor / "AUDIOTOOLS_LICENSE").is_file()


def test_discovery_does_not_import_torch_or_model():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import remotemedia.core.multiprocessing; "
            "torch_before = sys.modules.get('torch'); import indextts25_node; "
            "assert sys.modules.get('torch') is torch_before; "
            "assert 'indextts.infer_v2_5' not in sys.modules",
        ],
        cwd=PLUGIN,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


async def test_actual_runner_fallback_preserves_manifest_params(monkeypatch, tmp_path):
    from remotemedia.core.multiprocessing import runner

    monkeypatch.setattr(runner, "setup_signal_handlers", lambda node: None)
    config = NodeConfig(
        "from-runner",
        "IndexTTS25Node",
        {
            "model_dir": str(tmp_path),
            "lang": "ES",
            "duration_factor": 1.7,
        },
    )
    host = runner.NodeRunner(config)
    await host.initialize()
    assert host.node.node_id == "from-runner"
    assert host.node.model_dir == tmp_path
    assert host.node._defaults["lang"] == "ES"
    assert host.node._defaults["duration_factor"] == 1.7
    assert host.node._model is None
