//
//  ISTFT.swift
//  AudioUtils
//
//  High-performance Inverse Short-Time Fourier Transform
//

import Foundation
import MLX
import MLXFast
import MLXNN

// MARK: - Constants

/// Numerical stability epsilon for normalization
private let numericalStabilityEpsilon: Float = 1e-8

// MARK: - MLX-Optimized iSTFT Implementation

/// Thread-safe cache for normalization buffers
private final class NormBufferCache: @unchecked Sendable {
    private var cache: [String: MLXArray] = [:]
    private let lock = NSLock()

    func get(_ key: String) -> MLXArray? {
        lock.lock()
        defer { lock.unlock() }
        return cache[key]
    }

    func set(_ key: String, value: MLXArray) {
        lock.lock()
        defer { lock.unlock() }
        cache[key] = value
    }
}

private let normBufferCache = NormBufferCache()

/// MLX-optimized iSTFT using scatter-add operations like Python
public func istft(
    real: MLXArray,
    imag: MLXArray,
    nFFT: Int = 2048,
    hopLength: Int = 512,
    winLength: Int? = nil,
    window: MLXArray? = nil,
    center: Bool = true,
    length: Int? = nil
) -> MLXArray {
    let actualWinLength = winLength ?? nFFT
    let actualWindow = window ?? createHannWindow(actualWinLength, periodic: true)

    // Get windowed time-domain frames
    let stftComplex = real + MLXArray(real: 0, imaginary: 1) * imag
    let timeFrames = MLX.irfft(stftComplex.transposed(0, 2, 1), n: nFFT, axis: -1)
    let windowedFrames = timeFrames * actualWindow
    
    let batchSize = windowedFrames.shape[0]
    let numFrames = windowedFrames.shape[1]
    let frameLength = windowedFrames.shape[2]
    let olaLen = (numFrames - 1) * hopLength + frameLength
    
    // Pre-compute normalization buffer (with caching)
    let cacheKey = "\(numFrames)_\(hopLength)_\(frameLength)"
    let normBuffer: MLXArray

    if let cached = normBufferCache.get(cacheKey) {
        normBuffer = cached
    } else {
        let windowSquared = actualWindow * actualWindow
        var buffer = MLXArray.zeros([olaLen], dtype: .float32)

        // Vectorized normalization buffer creation using MLX operations
        let positions = MLXArray(0..<numFrames).expandedDimensions(axis: 1) * hopLength +
                       MLXArray(0..<frameLength).expandedDimensions(axis: 0)
        let positionsFlat = positions.flattened()
        let windowSqTiled = tiled(windowSquared, repetitions: [numFrames])

        // Use MLX's at[].add() for efficient scatter-add
        buffer = buffer.at[positionsFlat].add(windowSqTiled)

        buffer = MLX.maximum(buffer, MLXArray(numericalStabilityEpsilon))
        normBufferCache.set(cacheKey, value: buffer)
        normBuffer = buffer
    }
    
    // Optimized overlap-add using MLX scatter operations
    var output = MLXArray.zeros([batchSize, olaLen], dtype: .float32)
    
    // Reshape for vectorized scatter
    let windowedFlat = windowedFrames.reshaped([batchSize, -1])
    
    // Pre-compute positions
    let positions = MLXArray(0..<numFrames).expandedDimensions(axis: 1) * hopLength +
                   MLXArray(0..<frameLength).expandedDimensions(axis: 0)
    let positionsFlat = positions.flattened()
    
    // Optimized scatter-add with minimal overhead
    if batchSize == 1 {
        // Single batch - direct scatter-add without loop
        output[0] = output[0].at[positionsFlat].add(windowedFlat[0])
    } else {
        // Multiple batches - use simple loop
        for b in 0..<batchSize {
            output[b] = output[b].at[positionsFlat].add(windowedFlat[b])
        }
    }
    
    // Apply normalization
    output = output / normBuffer.expandedDimensions(axis: 0)
    
    // Apply center trimming
    if center {
        let startCut = nFFT / 2
        output = output[0..., startCut...]
    }
    
    // Apply length adjustment
    if let audioLength = length {
        let currentLen = output.shape[1]
        if currentLen > audioLength {
            output = output[0..., 0..<audioLength]
        } else if currentLen < audioLength {
            // Pad with zeros - this is the correct approach
            let padAmount = audioLength - currentLen
            let padding = MLXArray.zeros([output.shape[0], padAmount], dtype: output.dtype)
            output = MLX.concatenated([output, padding], axis: 1)
        }
    }
    
    return output
}


// MARK: - Batch Processing with Module

/// ISTFT Module for efficient batch processing
public class ISTFTModule: Module {
    let nFFT: Int
    let hopLength: Int
    let winLength: Int
    let center: Bool
    let windowFunc: MLXArray
    
    public init(nFFT: Int = 2048, hopLength: Int? = nil, winLength: Int? = nil,
                window: String = "hann", center: Bool = true) {
        self.nFFT = nFFT
        self.hopLength = hopLength ?? nFFT / 4
        self.winLength = winLength ?? nFFT
        self.center = center
        
        // Create window
        switch window {
        case "hann", "hann_window":
            self.windowFunc = createHannWindow(self.winLength)
        case "hamming":
            self.windowFunc = createHammingWindow(self.winLength)
        case "blackman":
            self.windowFunc = createBlackmanWindow(self.winLength)
        default:
            self.windowFunc = createHannWindow(self.winLength)
        }
        
        super.init()
    }
    
    /// Process STFT data through inverse transform
    /// - Parameters:
    ///   - real: Real part [batch, channels, freq, time]
    ///   - imag: Imaginary part [batch, channels, freq, time]
    /// - Returns: Reconstructed signal [batch, channels, time]
    public func callAsFunction(real: MLXArray, imag: MLXArray) -> MLXArray {
        let shape = real.shape
        let batchSize = shape[0]
        let channels = shape[1]
        let freqBins = shape[2]
        let timeFrames = shape[3]

        // Reshape to process all channels at once
        let combinedBatch = batchSize * channels

        // Reshape to [batch*channels, freq, time]
        let realFlat = real.reshaped([combinedBatch, freqBins, timeFrames])
        let imagFlat = imag.reshaped([combinedBatch, freqBins, timeFrames])

        // Apply ISTFT
        let outputFlat = istft(real: realFlat, imag: imagFlat, nFFT: nFFT,
                              hopLength: hopLength, winLength: winLength,
                              window: windowFunc, center: center)

        // Reshape back to [batch, channels, time]
        let outputLength = outputFlat.shape[1]
        let output = outputFlat.reshaped([batchSize, channels, outputLength])

        return output
    }
}

// MARK: - Convenience Functions

/// Reconstruct signal from magnitude and phase
public func reconstructFromMagPhase(
    magnitude: MLXArray,
    phase: MLXArray,
    nFFT: Int = 2048,
    hopLength: Int = 512,
    winLength: Int? = nil,
    window: MLXArray? = nil,
    center: Bool = true,
    length: Int? = nil
) -> MLXArray {
    // Convert magnitude and phase to real and imaginary
    let real = magnitude * MLX.cos(phase)
    let imag = magnitude * MLX.sin(phase)

    return istft(
        real: real,
        imag: imag,
        nFFT: nFFT,
        hopLength: hopLength,
        winLength: winLength,
        window: window,
        center: center,
        length: length
    )
}
