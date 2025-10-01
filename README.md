# AudioUtils

A Swift package for audio processing on Apple Silicon, built on MLX.

## Features

- **Core**: Load and save audio files
- **DSP**: STFT, ISTFT, Mel spectrograms, window functions
- **Filters**: Audio filtering and bandwidth operations
- **Utils**: Resampling and normalization

## Requirements

- iOS 16.0+ / macOS 13.3+
- Swift 5.9+

## Installation

Add to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/starkdmi/SwiftAudio", from: "1.0.0")
]
```

## Usage

### Load and Save Audio with Resampling

```swift
import AudioUtils

// Load audio with automatic resampling
let config = AudioLoader.Configuration(
    targetSampleRate: 16000,
    resamplingMethod: .cubic
)
let loader = AudioLoader(config: config)
let audio = try loader.load(from: "input.wav")

// Save processed audio
let saver = AudioSaver(config: .init(sampleRate: 16000))
try saver.save(audio, to: "output.wav")
```

### STFT and ISTFT

```swift
// Compute STFT
let (real, imag) = stft(audio, nFFT: 2048, hopLength: 512)

// Process in frequency domain...

// Reconstruct audio
let reconstructed = istft(real: real, imag: imag, nFFT: 2048, hopLength: 512)
```

### Mel Spectrogram

```swift
let melSpec = try melSpectrogram(
    audio,
    nFFT: 2048,
    numMels: 128,
    samplingRate: 16000,
    hopSize: 512,
    winSize: 2048,
    fmin: 0,
    fmax: 8000
)
```

### Audio Filtering

```swift
// Apply lowpass filter
let filtered = try lowpassFilter(audio, cutoff: 3000, fs: 16000, order: 4)

// Apply highpass filter
let highpassed = try highpassFilter(audio, cutoff: 100, fs: 16000)

// Apply bandpass filter
let bandpassed = try bandpassFilter(
    audio,
    lowCutoff: 300,
    highCutoff: 3400,
    fs: 16000
)
```

## License

Apache 2.0
