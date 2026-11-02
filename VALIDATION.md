# Validation record

Validated on Windows with Python 3.11.15 and an NVIDIA GeForce RTX 4090
(`cuda:1`) on 2026-11-02.

- 67 contract tests passed; the opt-in integration test is skipped by default.
- The separate resolver/managed provisioning/runner/IPC test passed, including
  an assertion that the actual Windows Python worker exits after session close.
- A fresh managed environment automatically installed 133 resolved packages
  from the declared dependencies and the SDK. No manual node dependency install
  targeted that environment.
- All ten real CUDA pipeline cases passed: voice cloning, emotion vector,
  Pinyin, CMU, Kana, Spanish, Arabic, and three duration settings. Every received
  waveform matched its worker WAV sample for sample. No workers remained alive
  after the suite.
- The user listened to the direct-inference clone, emotion, Pinyin, CMU, and Kana
  samples and reported that they sounded acceptable. This is a listening review,
  not a quantitative quality benchmark.
- Ruff, Windows launcher checks, shell syntax checks, and the host FFI build
  passed. Linux and other GPUs have not been runtime-verified.

## Versions

| Component | Version or revision |
|---|---|
| IndexTTS source | `d9e41aac89fd00b3d71497fddb287b7f24613712` |
| Audiotools source | `348ebf2034ce24e2a91a553e3171cb00c0c71678` |
| IndexTeam/IndexTTS-2.5 model | `c39ce5ba981572cb187443877ff559dfb246ce63` |
| torch / torchaudio | `2.8.0+cu128` |
| transformers | `4.52.1` |
| numpy | `2.2.6` |
| protobuf | `7.36.2` |
| soundfile in managed worker | `0.14.0` |

Vendor manifests retain source hashes. Weights and local recordings are not
included in the repository. The reference was the upstream demo's
`examples/voice_01.wav` from `IndexTeam/IndexTTS-2-Demo`.

## Controlled duration comparison

Direct production-node inference used the same text and seed 7, resetting Python,
NumPy, Torch, and CUDA RNGs before each call:

| Factor | Output duration |
|---|---|
| 0.5 | 1.556 seconds |
| 1.0 | 3.111 seconds |
| 1.2 | 3.727 seconds |

## Runtime prerequisite

The validated host includes the accompanying
[`SourcePythonNode` cleanup patch](patches/source-python-worker-cleanup.patch).
Apply it in `remotemedia-sdk-base` and rebuild the host if it is not already
present. It fixes an existing worker leak, including the Windows venv launcher's
child interpreter. The integration test catches that leak on Windows.
Iceoryx2 can still log a Windows node-resource removal warning even after the
worker exits; this was observed during validation.

Run the tests and the `--suite` consumer as documented in [README.md](README.md).
The consumer retains WAVs and a JSON result record for each new run.
