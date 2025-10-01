//
//  SimpleFilters.swift
//  AudioUtils
//
//  Simple filters for audio processing
//

import MLX
import Foundation

// MARK: - Simple Lowpass Filter

/// Simple lowpass filter using MLX vectorized operations
///
/// - Parameters:
///   - samples: Input audio samples (must be 1D array)
///   - cutoffRatio: Normalized cutoff frequency (0 to 1)
/// - Returns: Filtered audio samples
/// - Note: This function expects a 1D audio signal. For multi-channel audio, process each channel separately.
public func simpleLowpassFilter(
    _ samples: MLXArray,
    cutoffRatio: Float
) -> MLXArray {
    // Minimum window size for stable filtering
    let minWindowSize = 3
    // Window size scaling factor
    let windowSizeFactor: Float = 5.0

    let windowSize = max(minWindowSize, Int(windowSizeFactor / cutoffRatio))
    
    // Create Hann window for smoother filtering
    let n = MLXArray(0..<windowSize).asType(.float32)
    let hannWindow = 0.5 * (1.0 - MLX.cos(2.0 * Float.pi * n / Float(windowSize - 1)))
    let normalizedWindow = hannWindow / MLX.sum(hannWindow)
    
    // Use MLX convolution for efficient filtering
    let filtered = MLX.convolve(samples, normalizedWindow, mode: .same)
    
    return filtered
}

/// Simple lowpass filter with frequency specification
/// - Note: This function expects a 1D audio signal. For multi-channel audio, process each channel separately.
public func simpleLowpassFilter(
    _ samples: MLXArray,
    cutoff: Float,
    sampleRate: Float
) -> MLXArray {
    let cutoffRatio = cutoff / (sampleRate / 2.0)
    return simpleLowpassFilter(samples, cutoffRatio: cutoffRatio)
}

// MARK: - Simple Highpass Filter

/// Simple highpass filter (complementary to lowpass)
/// - Note: This function expects a 1D audio signal. For multi-channel audio, process each channel separately.
public func simpleHighpassFilter(
    _ samples: MLXArray,
    cutoffRatio: Float
) -> MLXArray {
    // Highpass = Original - Lowpass
    let lowpassed = simpleLowpassFilter(samples, cutoffRatio: cutoffRatio)
    return samples - lowpassed
}

/// Simple highpass filter with frequency specification
/// - Note: This function expects a 1D audio signal. For multi-channel audio, process each channel separately.
public func simpleHighpassFilter(
    _ samples: MLXArray,
    cutoff: Float,
    sampleRate: Float
) -> MLXArray {
    let cutoffRatio = cutoff / (sampleRate / 2.0)
    return simpleHighpassFilter(samples, cutoffRatio: cutoffRatio)
}

