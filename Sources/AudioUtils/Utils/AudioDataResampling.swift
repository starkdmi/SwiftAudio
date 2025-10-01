//
//  AudioDataResampling.swift
//  AudioUtils
//
//  Resampling utilities for AudioData
//

import Foundation
import MLX

// MARK: - AudioData Resampling

public extension AudioData {
    /// Resample AudioData to a new sample rate
    func resampled(to targetSampleRate: Int, quality: ResamplingQuality = .balanced) throws -> AudioData {
        guard sampleRate != targetSampleRate else { return self }

        // Convert to MLXArray
        let mlxArray = MLXArray(samples)

        // Resample using AudioUtils resampling
        let resampled = try resample(
            mlxArray,
            fromRate: Float(sampleRate),
            toRate: Float(targetSampleRate),
            quality: quality
        )
        
        // Convert back to Float array
        let resampledSamples = resampled.asArray(Float.self)
        
        // Calculate new duration
        let newDuration = Double(resampledSamples.count) / Double(targetSampleRate)
        
        return AudioData(
            samples: resampledSamples,
            sampleRate: targetSampleRate,
            duration: newDuration
        )
    }
    
    /// Resample AudioData asynchronously (for large files)
    func resampledAsync(to targetSampleRate: Int, quality: ResamplingQuality = .balanced) async throws -> AudioData {
        return try await Task.detached(priority: .userInitiated) {
            try self.resampled(to: targetSampleRate, quality: quality)
        }.value
    }
}

// MARK: - Convenience Methods

/// Resample AudioData (standalone function)
public func resampleBuffer(
    _ buffer: AudioData,
    to targetSampleRate: Int,
    quality: ResamplingQuality = .balanced
) throws -> AudioData {
    return try buffer.resampled(to: targetSampleRate, quality: quality)
}

/// Resample AudioData asynchronously (standalone function)
public func resampleBufferAsync(
    _ buffer: AudioData,
    to targetSampleRate: Int,
    quality: ResamplingQuality = .balanced
) async throws -> AudioData {
    return try await buffer.resampledAsync(to: targetSampleRate, quality: quality)
}
