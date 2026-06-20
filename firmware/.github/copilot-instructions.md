# Copilot project instructions

This repository is an ESP-IDF firmware for ESP32-S3 that embeds a TensorFlow Lite for Microcontrollers (TFLM) model. Use these notes to keep AI coding agents productive and avoid common pitfalls in this codebase.

## Overview

- Platform: ESP32-S3, ESP-IDF 5.x (idf.py, CMake, Ninja)
- Language: C/C++ (main entry in `main/main.cpp`)
- ML Inference: TensorFlow Lite Micro under `components/tflite-micro` (TFLM source vendored-in)
- Model assets: `main/model_data.{h,cc}`

Key component files:
- `components/tflite-micro/CMakeLists.txt`: assembles TFLM sources via glob and excludes host-only/platform-specific code.
- `main/CMakeLists.txt` + `CMakeLists.txt` (root): standard ESP-IDF build setup.

## Build and run

Prereqs:
- ESP-IDF installed and exported in your shell.
- Toolchain for ESP32-S3 available (xtensa-esp32s3-elf-gcc).

Build steps (developer runs these locally):
- idf.py reconfigure
- idf.py build
- idf.py -p <PORT> flash monitor

Notes:
- Target is ESP32-S3. If needed, set: idf.py set-target esp32s3
- The first build may be long due to TFLM sources.

## Third-party deps inside TFLM

TFLM expects some third_party trees to exist under `components/tflite-micro/tensorflow/third_party` and microfrontend’s `third_party`:

- Flatbuffers:
  - Required: headers under `tensorflow/third_party/flatbuffers/include`
  - Compatibility: TFLM in this repo expects Flatbuffers v23.x (e.g., v23.5.26). Newer 25.x headers will fail version checks in generated headers.

- gemmlowp:
  - Required for `fixedpoint` headers used by several TFLM kernels
  - Location: `tensorflow/third_party/gemmlowp`

- KISS FFT (for microfrontend):
  - Location: `tensorflow/lite/experimental/microfrontend/third_party/kissfft`
  - Ensure `tools/kiss_fftr.h` and `tools/kiss_fftr.c` are present. We exclude kissfft tools/tests in CMake so host-only utilities (png, fftw) won’t be pulled in.

These are vendored and not fetched automatically—make sure they exist when setting up the repo on a fresh machine.

## CMake source curation for TFLM

TFLM sources are added via a broad GLOB in `components/tflite-micro/CMakeLists.txt`, then pruned to fit embedded builds. Current exclusions include:
- Host-only utilities/tests/examples:
  - `tensorflow/lite/tools/*`
  - `tensorflow/lite/micro/tools/*`
  - `tensorflow/lite/micro/examples/*`
  - `tensorflow/lite/micro/benchmarks/*`
  - `tensorflow/lite/micro/integration_tests/*`
  - Any `*/test/*` folders and `*_test.(cc|c)` files
  - `tensorflow/lite/experimental/microfrontend/lib/frontend_memmap_main.c`
  - `tensorflow/lite/experimental/microfrontend/lib/frontend_io.c`
  - `tensorflow/lite/experimental/microfrontend/third_party/kissfft/tools/*`
- Platform-specific ports not used on ESP32-S3:
  - `tensorflow/lite/micro/arc_custom/*`
  - `tensorflow/lite/micro/arc_emsdp/*`
  - `tensorflow/lite/micro/bluepill/*`
  - `tensorflow/lite/micro/chre/*`
  - `tensorflow/lite/micro/hexagon/*`
  - `tensorflow/lite/micro/ceva/*`
  - `tensorflow/lite/micro/cortex_m_corstone_300/*`
  - `tensorflow/lite/micro/cortex_m_generic/*`
  - `tensorflow/lite/micro/kernels/arc_mli/*`
- Performance library not needed here:
  - `tensorflow/lite/micro/kernels/ruy/*`

Includes configured:
- `.` (component root)
- `tensorflow/third_party/flatbuffers/include`
- `tensorflow/third_party/gemmlowp`
- `tensorflow/lite/experimental/microfrontend/third_party/kissfft`

If new platform-specific compile errors arise, add another exclusion list(FILTER ... EXCLUDE REGEX ...) matching the offending path.

## Common errors and fixes

- Flatbuffers major version mismatch (e.g., 25 vs 23):
  - Symptom: compile-time asserts in schema-generated headers
  - Fix: use Flatbuffers v23.x headers in `tensorflow/third_party/flatbuffers/include`

- Missing `kiss_fftr.h` or `kiss_fftr.c` under microfrontend `tools/`:
  - Fix: place `kiss_fftr.h` and `kiss_fftr.c` into `tensorflow/lite/experimental/microfrontend/third_party/kissfft/tools/`

- Host-only microfrontend memmap utilities pulling `memmap.h`:
  - Fix: already excluded via CMake; do not try to generate `memmap.h` for embedded builds

- Platform-specific ports (ARC/Hexagon/CEVA/Corstone) pulling missing headers:
  - Fix: already excluded via CMake

- Platform-specific debug_log implementations (bluepill/arc_emsdp/chre) requiring eyalroz_printf or chre.h:
  - Fix: already excluded via CMake

- Cortex-M generic micro_time using ARM DWT/DCB peripherals not available on Xtensa:
  - Fix: already excluded via CMake

- Integration tests requiring generated golden test data:
  - Fix: already excluded via CMake

- Benchmarks requiring example/model data:
  - Fix: already excluded via CMake

- Ruy kernels causing extra deps or duplicates:
  - Fix: excluded via CMake

## Adding/updating a model

- Convert your `.tflite` model to a C array (e.g., `xxd -i` or `xxd -i -n model_data`), replace `main/model_data.cc` contents and keep `model_data.h` declarations.
- Ensure generated schema/flatbuffer compatibility is maintained (Flatbuffers v23.x)
- Rebuild and verify memory usage fits flash/PSRAM constraints on ESP32-S3.

## Conventions

- Keep `components/tflite-micro/CMakeLists.txt` as the single source of truth for all TFLM source filtering.
- Prefer minimal, self-contained changes in `main/` for application logic and model wiring.
- Avoid introducing host-only TFLM utilities; if you must include a new directory from upstream, review and exclude tests/tools/examples.

## PR checklist for agents

- Confirm third_party trees exist (flatbuffers v23.x, gemmlowp, kissfft with `tools/kiss_fftr.*`).
- Build with `idf.py build`; attach first failing error if it fails.
- If failure points to a non-ESP directory, exclude it in `components/tflite-micro/CMakeLists.txt`.
- Keep include paths limited to the minimal needed third_party dirs.
- Do not add external dependencies that conflict with ESP-IDF unless strictly necessary.

## Notes for maintainers

- If upstreaming TFLM updates, expect to re-validate exclusions. Run a clean build and scan the first compile errors for paths to exclude.
- Consider adding a small CI check that verifies presence of `flatbuffers/include`, `gemmlowp`, and `kissfft/tools/kiss_fftr.*` to avoid setup regressions.
