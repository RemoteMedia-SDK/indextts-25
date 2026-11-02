"""Test-only backend, loaded by the real subprocess runner and IPC machinery."""

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from indextts25_node import IndexTTS25Node  # noqa: E402


class FakeIndexTTS:
    def infer(self, **kwargs):
        assert kwargs["text"] == "<日本|ニホン>です。"
        assert kwargs["lang"] == "JA", "runner dropped manifest language"
        assert kwargs["duration_factor"] == 1.2, "runner dropped manifest duration"
        assert kwargs["emo_vector"] == [0, 0, 0.8, 0, 0, 0, 0, 0]
        assert kwargs["output_path"] is None
        assert kwargs["stream_return"] is False
        return 22050, np.array([-32768, 0, 16384, 32767], dtype=np.int16)[:, None]


def load_model(self):
    pid_file = os.environ.get("INDEXTTS25_TEST_PID_FILE")
    if pid_file:
        Path(pid_file).write_text(str(os.getpid()), encoding="ascii")
    self._model = FakeIndexTTS()


IndexTTS25Node._load_model = load_model
