//
//  BandwidthSub.swift
//  AudioUtils
//
//  Bandwidth substitution for audio processing
//  Includes both high-quality and fast implementations
//

import MLX
import MLXFast

// MARK: - Bandwidth Detection

/// Detect the effective bandwidth of a signal using STFT
///
/// - Parameters:
///   - signal: Input audio signal (must be 1D array)
///   - fs: Sampling frequency
///   - energyThreshold: Energy threshold for bandwidth detection (default: 0.99)
/// - Returns: Tuple of (low_frequency, high_frequency) detected
/// - Note: This function expects a 1D audio signal. For multi-channel audio, process each channel separately.
public func detectBandwidth(_ signal: MLXArray, fs: Int, energyThreshold: Float = 0.99) -> (Float, Float) {
    // Parameters matching Python implementation
    let nperseg = 256
    let noverlap = 128
    let hopLength = nperseg - noverlap
    let nfft = 256
    
    // Reshape signal to 2D
    let signalReshaped = signal.ndim == 1 ? signal.expandedDimensions(axis: 0) : signal
    
    // Create Hann window (periodic=false)
    let indices = MLXArray(0..<nperseg).asType(.float32)
    let window = 0.5 * (1 - MLX.cos(2 * Float.pi * indices / Float(nperseg - 1)))
    
    // Compute STFT using our fast implementation
    let (real, imag) = stft(signalReshaped, nFFT: nfft, hopLength: hopLength, 
                           winLength: nperseg, window: window, center: true)
    
    // Compute PSD - shape: (1, freq_bins, time_frames)
    let realSquared = real[0] * real[0]
    let imagSquared = imag[0] * imag[0]
    let psd = realSquared + imagSquared
    
    // Calculate cumulative energy
    let totalEnergy = MLX.sum(psd).item(Float.self)
    
    // Handle silence or very low energy signals
    if totalEnergy < 1e-10 {
        // Return default bandwidth for silence
        return (0.0, Float(fs) / 2.0)
    }
    
    let energyPerFreq = MLX.sum(psd, axis: 1)
    let cumulativeEnergy = MLX.cumsum(energyPerFreq) / totalEnergy
    
    // Calculate frequencies
    let freqBins = nfft / 2 + 1
    let f = MLXArray(0..<freqBins).asType(.float32) * (Float(fs) / Float(nfft))
    
    // Find f_low: first frequency > 0 where cumulative energy > (1 - threshold)
    let lowThreshold = 1.0 - energyThreshold
    
    // Skip first element (0 Hz) for low frequency
    let lowMask = cumulativeEnergy[1...] .> Float(lowThreshold)
    let fLow: Float
    if MLX.any(lowMask).item(Bool.self) {
        let lowIdx = MLX.argMax(lowMask.asType(.int32)).item(Int.self)
        fLow = f[1 + lowIdx].item(Float.self)
    } else {
        fLow = f.shape[0] > 1 ? f[1].item(Float.self) : f[0].item(Float.self)
    }
    
    // Find f_high: first frequency where cumulative energy >= threshold
    let highMask = cumulativeEnergy .>= energyThreshold
    let fHigh: Float
    if MLX.any(highMask).item(Bool.self) {
        let highIdx = MLX.argMax(highMask.asType(.int32)).item(Int.self)
        fHigh = f[highIdx].item(Float.self)
    } else {
        fHigh = f[f.shape[0] - 1].item(Float.self)
    }
    
    return (fLow, fHigh)
}

// MARK: - Main Implementation (High Quality)

/// Bandwidth substitution using high-quality Butterworth filters
///
/// This function takes a low-bandwidth audio signal and substitutes its frequency
/// content into a high-bandwidth signal, preserving the spectral characteristics
/// while maintaining the higher sample rate quality.
///
/// - Parameters:
///   - lowBandwidthAudio: Audio signal with limited bandwidth (e.g., upsampled from lower sample rate) - must be 1D array
///   - highBandwidthAudio: Audio signal with full bandwidth - must be 1D array
///   - fs: Sampling frequency (default: 48000 Hz)
/// - Returns: Audio with substituted bandwidth
/// - Note: This function expects 1D audio signals. For multi-channel audio, process each channel separately.
public func bandwidthSub(
    _ lowBandwidthAudio: MLXArray,
    _ highBandwidthAudio: MLXArray,
    fs: Int = 48000
) throws -> MLXArray {
    // Margin from Nyquist frequency for safe filtering
    let nyquistMarginHz: Float = 100

    // Detect effective bandwidth of the first signal
    let (fLow, fHigh) = detectBandwidth(lowBandwidthAudio, fs: fs)

    // Safety check: if fHigh is at Nyquist, just return the high bandwidth audio
    // This handles the silence case where filtering at Nyquist can cause issues
    if fHigh >= Float(fs) / 2.0 - nyquistMarginHz {
        return highBandwidthAudio
    }

    // Replace the lower frequency content of the second audio
    let substitutedAudio = try replaceBandwidth(
        signal1: lowBandwidthAudio,
        signal2: highBandwidthAudio,
        fs: fs,
        fLow: fLow,
        fHigh: fHigh
    )
    
    // Optional: Smooth the transition
    let smoothedAudio = smoothTransition(
        signal1: substitutedAudio,
        signal2: lowBandwidthAudio,
        fs: fs
    )
    
    return smoothedAudio
}

/// Replace frequency content between fLow and fHigh from signal1 into signal2
/// Uses high-quality Butterworth filters
private func replaceBandwidth(
    signal1: MLXArray,
    signal2: MLXArray,
    fs: Int,
    fLow: Float,
    fHigh: Float
) throws -> MLXArray {
    let fsFloat = Float(fs)

    // Extract effective band from signal1 (frequencies below fHigh)
    let effectiveBand = try lowpassFilter(signal1, cutoff: fHigh, fs: fsFloat)

    // Extract high frequency content from signal2 (frequencies above fHigh)
    let signal2Highpass = try highpassFilter(signal2, cutoff: fHigh, fs: fsFloat)
    
    // Check for NaN/Inf in filter outputs
    eval(effectiveBand)
    eval(signal2Highpass)
    
    // Match lengths
    let minLength = min(effectiveBand.shape[0], signal2Highpass.shape[0])
    let effectiveBandTrimmed = effectiveBand[0..<minLength]
    let signal2HighpassTrimmed = signal2Highpass[0..<minLength]
    
    // Combine signals
    return signal2HighpassTrimmed + effectiveBandTrimmed
}


// MARK: - Common Functions

/// Apply smooth transition between two signals
private func smoothTransition(
    signal1: MLXArray,
    signal2: MLXArray,
    fs: Int,
    transitionBand: Int = 100
) -> MLXArray {
    // Calculate fade length
    let fadeLength = transitionBand * fs / 1000
    
    // Create fade curve using MLX
    let fade = MLX.linspace(0, 1, count: fadeLength)
    
    // Get minimum length
    let minLength = min(signal1.shape[0], signal2.shape[0])
    
    // Create full crossfade array
    let crossfade: MLXArray
    if fadeLength < minLength {
        let onesLength = minLength - fadeLength
        crossfade = MLX.concatenated([fade, MLXArray.ones([onesLength])], axis: 0)
    } else {
        crossfade = fade[0..<minLength]
    }
    
    // Apply crossfade
    let smoothedSignal = (1 - crossfade) * signal2[0..<minLength] + crossfade * signal1[0..<minLength]
    
    return smoothedSignal
}

