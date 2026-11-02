# IndexTTS 2.5 source plugin

`IndexTTS25Node` clones a reference voice and supports an eight-value emotion
vector, Pinyin/CMU/Kana pronunciation annotations, and duration control. It uses
the existing Python source-plugin resolver and emits SDK audio.

## Run

Use a RemoteMedia runtime built with Python source-plugin support. The managed
environment **automatically installs this node's dependencies** from `plugin.toml`.
The example selects Python 3.11 and matching PyTorch 2.8 CUDA 12.8 wheels on
Windows and Linux x86_64. No manual torch or IndexTTS installation is needed.
The first environment provision downloads several GB; allow time and disk space.
Git is needed for any SDK development dependencies that use Git URLs.

Provide an existing reference audio clip and the
[IndexTeam/IndexTTS-2.5 checkpoints](https://huggingface.co/IndexTeam/IndexTTS-2.5).
The model directory must include `config.yaml` and its referenced weights.
For example, the Hugging Face CLI can download the model with
`hf download IndexTeam/IndexTTS-2.5 --local-dir checkpoints`.
Qwen emotion-text inference is disabled, so its optional checkpoint is unused.
The upstream loader may download auxiliary models on first initialization.

Clone the plugin and run from its root:

```sh
git clone https://github.com/RemoteMedia-SDK/indextts-25.git
cd indextts-25
```

On Windows:

```powershell
.\run.cmd --model-dir C:\models\IndexTTS-2.5 --reference C:\voices\prompt.wav --case clone
```

On Linux:

```sh
bash run.sh --model-dir /models/IndexTTS-2.5 --reference /voices/prompt.wav --case clone
```

For SDK development, set `REMOTEMEDIA_PYTHON_SRC` to the SDK's `clients/python`
directory before launching. The worker environment will install that SDK source
alongside the node dependencies. An installed SDK can be used without this
variable. The host interpreter must provide the built `remotemedia.runtime`.

To load the published source directly in a pipeline, use:

```json
{"plugins": ["RemoteMedia-SDK/indextts-25@v0.1.0"]}
```

Use `IndexTTS25Node` in the pipeline's nodes, with the parameters illustrated in
[manifest.json](manifest.json). That example uses `.` for a local checkout.

Use `--case emotion`, `pinyin`, `cmu`, `kana`, `spanish`, `arabic`, `fast`,
`normal`, or `slow`; `--suite` runs all cases. `--device cuda:1` selects another
GPU. `--request request.json` sends a custom request. The consumer writes received
audio and a `results.json` report to `outputs/` (or `--output-dir`). It runs each
case in its own session and verifies that the received audio matches an explicitly
requested worker WAV before finishing. Temporary verification exports are removed.
Within any node instance, successive requests reuse the same model.

The launchers set source paths and a ten-minute node timeout without changing
the caller's working directory. A custom manifest can reference the plugin by
absolute directory. `model_dir`, `cfg_path`, and all audio paths refer to files
on the **worker host**; they do not upload client-local files to a remote worker.

## Request API

Plain text uses node defaults. A dict, SDK JSON value, or JSON object in a text
frame can override synthesis settings for one request:

```json
{
  "text": "Hello, this is a voice cloning demo.",
  "spk_audio_prompt": "C:/voices/prompt.wav",
  "lang": "EN",
  "emo_vector": [0, 0, 0.8, 0, 0, 0, 0, 0],
  "duration_factor": 1.2,
  "output_path": "C:/outputs/output.wav"
}
```

| Field | Meaning |
|---|---|
| `text` | Required nonblank text, forwarded unchanged |
| `spk_audio_prompt` | Readable reference audio; required in defaults or request |
| `lang` | `ZH`, `EN`, `JA`, `ES`, `AR`; case-insensitive; default `EN` |
| `emo_vector` | Eight finite numbers in [0, 1]; default null |
| `duration_factor` | Finite number in [0.5, 2.0]; default 1.0 |
| `output_path` | Optional request-only PCM16 WAV export; replaces an existing target |

Emotion order is **happy, angry, sad, afraid, disgusted, melancholic, surprised,
calm**. Explicit `emo_vector: null` clears a configured vector for one request;
omission inherits the default. The wrapper does not reorder or normalize values.
Values above 1.0 for `duration_factor` slow speech; below 1.0 speed it up.

Pronunciation examples, handled by the model's parser:

```json
{"text":"他在银<行|XING2>里<行|HANG2>走了半天。","lang":"ZH"}
{"text":"I am <going|G OW1 . IH0 NG> home.","lang":"EN"}
{"text":"<日本|ニホン>です。","lang":"JA"}
```

Unknown request keys, booleans in numeric controls, invalid numbers, unsupported
languages, unreadable references, and blank text are errors. Text starting with
`{` is parsed as a JSON request; malformed JSON is not spoken as literal text.

## Node configuration and output

The manifest supports `model_dir` (default `checkpoints`), `cfg_path` (default
`<model_dir>/config.yaml`), `device` (default upstream automatic selection),
`use_bf16` (default true), and `chunk_size` (positive sample count, default 4096).
The speaker, language, emotion vector, and duration fields also serve as node
defaults. `__python_version__: "3.11"` selects the managed interpreter.

Each request completes synthesis before emitting mono float32 audio, normally
at 22,050 Hz. Frames reconstruct the original waveform in order; frame size
does not represent incremental model generation. Exactly one text frame
`<|audio_end|>` follows successful output. No WAV is written unless requested.
An output-file failure is reported before any audio is emitted.

Recoverable errors are JSON text frames with `error`, `input`, and `processed_by`.
They have no success marker. Initialization failures and fatal CUDA errors reach
the runner. Display-only SDK text (`channel="ui"`) passes through unchanged.
Inference runs off the event loop and is serialized per model. Cancellation
waits for the active inference thread before model reuse or cleanup.

## Validation

From this plugin's root, with the SDK's usual development/test tools:

```powershell
$env:REMOTEMEDIA_PYTHON_SRC = 'C:/path/to/remotemedia-sdk/clients/python'
python -m pytest -q tests
python -m ruff check .
```

Omit the environment variable when testing against an installed SDK.

The ordinary tests replace only inference with a fake backend, using actual SDK
node/config/audio types. They verify contracts, signed PCM16 conversion, WAV
output, cancellation, and runner constructor fallback. They do not establish
voice quality or real GPU support. `--suite` exercises the actual pipeline with
real weights; inspect its result record and audition its WAVs separately.

Set `INDEXTTS25_RUNTIME_TEST=1` to enable the additional fake-backend integration
test. It uses the real resolver, managed provisioning, subprocess runner, and IPC;
on Windows it also verifies that the worker exits when the session closes. For
a development FFI build, `REMOTEMEDIA_TEST_DLL_DIR` may specify DLL search paths.
The source runtime must include the `SourcePythonNode` worker-cleanup fix in
[patches/source-python-worker-cleanup.patch](patches/source-python-worker-cleanup.patch);
older runtimes can leave Python workers alive after session close. Apply that
patch in a `remotemedia-sdk-base` checkout and rebuild the host runtime if needed.

See [VALIDATION.md](VALIDATION.md) for tested versions and platform limitations.

## Source and dependency provenance

The RemoteMedia wrapper, examples, and tests retain the SDK's
[Community License 1.0](LICENSE.md). Vendored upstream code keeps its own
licenses listed below.

The unchanged upstream package source is pinned at
[`d9e41aac89fd00b3d71497fddb287b7f24613712`](https://github.com/index-tts/index-tts/tree/d9e41aac89fd00b3d71497fddb287b7f24613712).
`_vendor/manifest.json` records hashes; original license/disclaimer files are
preserved there. The training-only JDC checkpoint is excluded. Main and
auxiliary inference weights are external.

Installing upstream's entire package conflicts with the SDK: its
`descript-audiotools==0.7.2` dependency requires protobuf below 3.20, and the
newer 0.7.4 package still caps it below 5. The SDK's generated code needs protobuf
6.31.1 or newer. This plugin also vendors unchanged audiotools 0.7.4 source at
[`348ebf2`](https://github.com/descriptinc/audiotools/tree/348ebf2034ce24e2a91a553e3171cb00c0c71678),
records its hashes in `_vendor/audiotools-manifest.json`, and declares both
libraries' dependencies with the compatible protobuf bound. The retained
`AUDIOTOOLS_LICENSE` covers that source. Real inference verifies this combination.
The managed environment installs SDK and node dependencies together; no manual
installation or dependency-resolver bypass is needed.
Optional acceleration engines and Qwen text-emotion inference are disabled.

Use of upstream code and models is governed by their included licenses and the
[model license](https://huggingface.co/IndexTeam/IndexTTS-2.5/blob/main/LICENSE).
The [upstream model card](https://huggingface.co/IndexTeam/IndexTTS-2.5) documents
the supported model features and reference-voice usage responsibilities.
