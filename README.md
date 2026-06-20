# Dosypator

An embedded ML device that listens for a crying baby and responds with a soothing sound — built on an ESP32-S3, running a custom-trained cry-detection model entirely on-device.

## Status: working end-to-end

The full pipeline — dual-mic audio capture, MFCC feature extraction, on-device CNN inference, and a triggered audio response — runs on real hardware and correctly distinguishes background noise from a loud cry-like sound. From a live test run:

```
Background room noise:  P(cry) = 0.0000 – 0.0117
Loud cry-like sound:    P(cry) = 0.7344 – 0.8555  →  CRY DETECTED, response triggered
```

This isn't a simulation or a bench test — it's the device's serial output, captured while actually running, reacting to actual sound in real time.

## What it does

A parent records up to 5 short audio clips (their own voice, a lullaby, anything soothing). The device continuously listens; when its on-device model detects the acoustic signature of a baby crying, it plays one of those clips — so a child wakes briefly, hears a familiar voice, and goes back to sleep, without a parent needing to get up.

## How it works

```
Dual I2S microphones (ES7210 codec)
  → 3-second audio capture (16kHz)
  → Windowed FFT (custom KissFFT integration) → Mel filterbank → MFCC features
  → Quantized INT8 CNN (TensorFlow Lite Micro)
  → P(cry) > threshold → triggers playback (ES8311 codec → speaker)
```

Everything — feature extraction and inference — runs on the ESP32-S3 itself. No cloud, no network dependency, no latency beyond the 3-second capture window.

### Hardware

[Waveshare ESP32-S3 Audio Board](https://www.waveshare.com/esp32-s3-audio-board.htm): ES7210 4-channel mic ADC (2 channels wired), ES8311 speaker DAC, TCA9555 GPIO expander, all communicating over I2C with audio over I2S.

### The model

A CNN trained on a labeled dataset of crying vs. non-crying audio, achieving ~98% discrimination in offline evaluation. The training pipeline (feature extraction → TFLite conversion → quantization) is in `ml-training/`. The deployed model is INT8-quantized to fit ESP32-S3 memory and run inference fast enough for real-time use.

### A deliberate engineering choice: a separate KissFFT

The firmware vendors its own copy of KissFFT (`local_kiss_fft.c/h`) rather than using the one bundled inside TensorFlow Lite Micro, which uses 16-bit integer FFT internally. The MFCC pipeline needed float-precision FFT to match the Python training pipeline's output exactly — using TFLM's int16 version would have introduced a type/precision mismatch between training and inference. Keeping them separate, rather than trying to coerce one implementation to serve both purposes, was the simpler and more correct fix.

## Known limitations (next steps, not blockers)

- **Detection threshold needs real-world calibration.** It's currently set to 0.50 in the shipped code — intentionally low, set during the verification test above to confirm the full pipeline reacts correctly. The model clearly discriminates (near-zero on silence, 0.73–0.86 on a loud test sound), but a production threshold should be tuned against a real corpus of crying audio, not a single test session, before this goes anywhere near an actual sleeping child.
- **Response sound is a placeholder.** The current "shush" response is a flat 2-second, 440 Hz sine wave — a deliberate placeholder to verify the audio output path independently of the actual content (see the `// TODO` in `main.cpp`). It works, but it isn't pleasant, and isn't what a parent would want a child to hear. Swapping it for actual recorded audio (a parent's voice, a lullaby) is a known next step, not a technical blocker.
- **No mobile app / cloud sync.** The audio response is currently hardcoded on-device. The original design included a companion app for uploading custom clips remotely — not built yet.

## Stack

C++ (ESP-IDF / FreeRTOS), TensorFlow Lite Micro, KissFFT, I2C/I2S peripheral drivers · Python (TensorFlow/Keras for training, librosa-style MFCC feature extraction for the offline pipeline)

## Repository structure

```
firmware/
├── main/
│   ├── main.cpp                 # Full pipeline: I2C/I2S init, MFCC, inference, response
│   ├── dosypator_model.tflite   # Quantized INT8 model, embedded into the firmware binary
│   └── local_kiss_fft*          # Float-precision FFT, kept separate from TFLM's int16 version
├── idf_component.yml            # Pulls in TensorFlow Lite Micro via ESP-IDF Component Manager
└── CMakeLists.txt / sdkconfig / partitions.csv

ml-training/
├── extract_features.py          # Audio → MFCC feature extraction
├── convert_to_tflite.py         # Keras model → quantized TFLite
├── predict.py / test_audio.py   # Offline model testing
└── mfcc_stats.npy               # Normalization stats (mean/std), must match firmware exactly
```

The training dataset and the full-precision `.h5` Keras model aren't included here (large binary files, not useful without the original audio data) — what's here is the complete, runnable conversion and firmware pipeline.

## Setup

Requires the [ESP-IDF toolchain](https://docs.espressif.com/projects/esp-idf/en/stable/esp32s3/get-started/) (v5.2+) and a Waveshare ESP32-S3 Audio Board (or compatible ES7210/ES8311 hardware).

```bash
source $IDF_PATH/export.sh
cd firmware
idf.py build
idf.py -p /dev/cu.usbmodemXXXX flash monitor
```
