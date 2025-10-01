//
//  AudioNormalization.swift
//  AudioUtils
//
//  High-performance audio normalization utilities with MLX optimization
//

import MLX
import Foundation

// MARK: - Automatic Normalization

/// Vectorized automatic normalization using MLX operations
/// Significantly faster than element-wise scalar operations
///
/// - Parameter samples: Input audio samples
/// - Returns: Normalized audio with peak at 1.0
public func normalizeAutomatic(_ samples: MLXArray) -> MLXArray {
    // Prevent division by zero
    let epsilon: Float = 1e-10

    // Handle empty arrays
    if samples.size == 0 {
        return samples
    }

    // Use MLX vectorized operations for maximum performance
    let absoluteValues = MLX.abs(samples)
    let maxVal = MLX.max(absoluteValues)

    let scale = MLXArray(Float(1.0)) / MLX.maximum(maxVal, MLXArray(epsilon))

    return samples * scale
}

/// Normalize with target peak level
public func normalizeToPeak(_ samples: MLXArray, targetPeak: Float = 1.0) -> MLXArray {
    // Handle empty arrays
    if samples.size == 0 {
        return samples
    }
    
    let absoluteValues = MLX.abs(samples)
    let maxVal = MLX.max(absoluteValues)
    
    let epsilon = MLXArray(Float(1e-10))
    let scale = MLXArray(targetPeak) / MLX.maximum(maxVal, epsilon)
    
    return samples * scale
}

// MARK: - RMS Normalization

/// Normalize based on RMS (Root Mean Square) level
public func normalizeRMS(_ samples: MLXArray, targetRMS: Float = 0.7) -> MLXArray {
    let squared = samples * samples
    let meanSquared = MLX.mean(squared)
    let rms = MLX.sqrt(meanSquared)
    
    let epsilon = MLXArray(Float(1e-10))
    let scale = MLXArray(targetRMS) / MLX.maximum(rms, epsilon)
    
    return samples * scale
}

// MARK: - LUFS Normalization

/// Normalize to LUFS (Loudness Units relative to Full Scale)
/// Simplified implementation - for full EBU R128 compliance, use specialized libraries
public func normalizeLUFS(_ samples: MLXArray, targetLUFS: Float = -14.0, sampleRate: Float = 48000) -> MLXArray {
    // LUFS reference offset for full scale
    let lufsReferenceOffset: Float = -0.691
    // Prevent log of zero
    let epsilon: Float = 1e-10

    // Apply K-weighting filter (simplified)
    let preFiltered = applyKWeighting(samples, sampleRate: sampleRate)

    // Calculate mean square
    let meanSquare = MLX.mean(preFiltered * preFiltered)

    // Convert to LUFS (simplified)
    let currentLUFS = lufsReferenceOffset + 10 * log10(meanSquare.item(Float.self) + epsilon)

    // Calculate gain
    let gainDB = targetLUFS - currentLUFS
    let gain = Float(pow(10.0, gainDB / 20.0))

    return samples * gain
}

/// Apply K-weighting filter for LUFS measurement (simplified)
private func applyKWeighting(_ samples: MLXArray, sampleRate: Float) -> MLXArray {
    // High shelf at 1.5kHz, +4dB
    let shelfFreq: Float = 1500
    let shelfGain: Float = 1.585  // ~4dB
    
    // Apply simple shelf filter
    let cutoffRatio = shelfFreq / (sampleRate / 2.0)
    let highFreq = samples - simpleLowpass(samples, cutoffRatio: cutoffRatio)
    
    return samples + highFreq * (shelfGain - 1.0)
}

// MARK: - Limiting and Clipping

/// Apply soft limiting to prevent clipping
public func softLimit(_ samples: MLXArray, threshold: Float = 0.95, knee: Float = 0.1) -> MLXArray {
    let absSignal = MLX.abs(samples)
    let sign = MLX.sign(samples)
    
    // Soft knee limiting
    let lowerThreshold = threshold - knee
    
    // Three regions: linear, knee, limiting
    let linear = absSignal .<= lowerThreshold
    let kneeRegion = (absSignal .> lowerThreshold) & (absSignal .< threshold)
    // let limiting = absSignal .>= threshold
    
    // Calculate gain reduction in knee region
    let kneeRatio = (absSignal - lowerThreshold) / (2 * knee)
    let kneeGain = 1.0 - kneeRatio * kneeRatio
    
    // Apply appropriate processing to each region
    let processedAbs = MLX.where(
        linear,
        absSignal,
        MLX.where(
            kneeRegion,
            lowerThreshold + (absSignal - lowerThreshold) * kneeGain,
            threshold + (absSignal - threshold) * 0.1  // 10:1 ratio above threshold
        )
    )
    
    return sign * processedAbs
}

/// Hard clip to prevent values exceeding threshold
public func hardClip(_ samples: MLXArray, threshold: Float = 1.0) -> MLXArray {
    return MLX.clip(samples, min: -threshold, max: threshold)
}

/// Soft clip with tanh saturation
public func softClip(_ samples: MLXArray, threshold: Float = 0.8) -> MLXArray {
    let thresholdMLX = MLXArray(threshold)
    let sign = MLX.sign(samples)
    let abs = MLX.abs(samples)
    
    // Soft clipping function: tanh-based for smooth saturation
    let clipped = MLX.where(
        abs .<= thresholdMLX,
        samples,
        sign * (thresholdMLX + (1.0 - thresholdMLX) * MLX.tanh((abs - thresholdMLX) / (1.0 - thresholdMLX)))
    )
    
    return clipped
}

// MARK: - DC Offset Removal

/// Remove DC offset from signal
public func removeDC(_ samples: MLXArray) -> MLXArray {
    let dcOffset = MLX.mean(samples)
    return samples - dcOffset
}

/// Remove DC offset with running average (for streaming)
public func removeDCOffsetAdaptive(_ samples: MLXArray, alpha: Float = 0.995) -> MLXArray {
    let n = samples.shape[0]
    var dcEstimate = MLXArray(Float(0.0))
    let output = MLXArray.zeros([n], dtype: samples.dtype)
    
    // Exponential moving average for DC estimation
    for i in 0..<n {
        dcEstimate = alpha * dcEstimate + (1 - alpha) * samples[i]
        output[i] = samples[i] - dcEstimate
    }
    
    return output
}

// MARK: - Batch Normalization

/// Process multiple audio samples in batch for improved efficiency
public func batchNormalize(_ samplesBatch: [MLXArray], targetPeak: Float = 1.0) -> [MLXArray] {
    // Find global maximum across all samples
    var globalMax: Float = 0.0
    
    for samples in samplesBatch {
        let maxVal = MLX.max(MLX.abs(samples)).item(Float.self)
        globalMax = max(globalMax, maxVal)
    }
    
    // Apply same scale to all samples
    let scale = targetPeak / (globalMax + 1e-10)
    
    return samplesBatch.map { $0 * scale }
}

/// Batch normalize with independent scaling
public func batchNormalizeIndependent(_ samplesBatch: [MLXArray]) -> [MLXArray] {
    return samplesBatch.map { normalizeAutomatic($0) }
}

// MARK: - True Peak Detection

/// Detect true peak using oversampling
public func detectTruePeak(_ samples: MLXArray, oversampleFactor: Int = 4) -> Float {
    // Handle multi-dimensional arrays
    if samples.ndim > 1 {
        var maxPeak: Float = 0.0
        for i in 0..<samples.shape[0] {
            let channelPeak = detectTruePeak(samples[i], oversampleFactor: oversampleFactor)
            maxPeak = max(maxPeak, channelPeak)
        }
        return maxPeak
    }
    
    // Oversample to detect inter-sample peaks
    let oversampled = resampleCubic(samples, fromRate: 1.0, toRate: Float(oversampleFactor))
    return MLX.max(MLX.abs(oversampled)).item(Float.self)
}

/// Normalize with true peak limiting
public func normalizeTruePeak(_ samples: MLXArray, targetPeak: Float = 0.95) -> MLXArray {
    let truePeak = detectTruePeak(samples)
    let scale = targetPeak / (truePeak + 1e-10)
    return samples * scale
}

// MARK: - Loudness Matching

/// Match loudness between two signals
public func matchLoudness(target: MLXArray, source: MLXArray) -> MLXArray {
    // Calculate RMS of both signals
    let targetRMS = MLX.sqrt(MLX.mean(target * target))
    let sourceRMS = MLX.sqrt(MLX.mean(source * source))
    
    // Calculate scale factor
    let scale = targetRMS / (sourceRMS + MLXArray(Float(1e-10)))
    
    return source * scale
}

/// Match loudness of batch to reference
public func matchLoudnessBatch(_ batch: [MLXArray], reference: MLXArray) -> [MLXArray] {
    let referenceRMS = MLX.sqrt(MLX.mean(reference * reference)).item(Float.self)
    
    return batch.map { samples in
        let samplesRMS = MLX.sqrt(MLX.mean(samples * samples)).item(Float.self)
        let scale = referenceRMS / (samplesRMS + 1e-10)
        return samples * scale
    }
}

// MARK: - Utility Functions

/// Calculate crest factor (peak to RMS ratio)
public func crestFactor(_ samples: MLXArray) -> Float {
    let peak = MLX.max(MLX.abs(samples)).item(Float.self)
    let rms = MLX.sqrt(MLX.mean(samples * samples)).item(Float.self)
    return 20 * log10(peak / (rms + 1e-10))
}

/// Calculate dynamic range
public func dynamicRange(_ samples: MLXArray, percentile: Float = 0.95) -> Float {
    let sorted = MLX.sorted(MLX.abs(samples))
    let index = Int(Float(sorted.shape[0]) * percentile)
    let topPercentile = sorted[index].item(Float.self)
    let noise = sorted[Int(Float(sorted.shape[0]) * 0.1)].item(Float.self)
    return 20 * log10(topPercentile / (noise + 1e-10))
}

// Import simple lowpass filter from SimpleFilters
private func simpleLowpass(_ samples: MLXArray, cutoffRatio: Float) -> MLXArray {
    // Handle multi-dimensional arrays
    if samples.ndim > 1 {
        // Process each channel separately
        let result = MLXArray.zeros(samples.shape)
        for i in 0..<samples.shape[0] {
            result[i] = simpleLowpass(samples[i], cutoffRatio: cutoffRatio)
        }
        return result
    }
    
    let windowSize = max(3, Int(5.0 / cutoffRatio))
    let n = MLXArray(0..<windowSize).asType(.float32)
    let hannWindow = 0.5 * (1.0 - MLX.cos(2.0 * Float.pi * n / Float(windowSize - 1)))
    let normalizedWindow = hannWindow / MLX.sum(hannWindow)
    let result = MLX.convolve(samples, normalizedWindow, mode: .same)
    
    // Ensure output has same size as input
    if result.shape[0] != samples.shape[0] {
        // Trim or pad to match original size
        if result.shape[0] > samples.shape[0] {
            return result[0..<samples.shape[0]]
        } else {
            // This shouldn't happen with mode .same, but handle it just in case
            return MLX.padded(result, widths: [IntOrPair([0, samples.shape[0] - result.shape[0]])])
        }
    }
    return result
}
