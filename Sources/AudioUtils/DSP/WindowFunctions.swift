//
//  WindowFunctions.swift
//  AudioUtils
//
//  Common window functions for audio processing
//  Optimized implementations using MLX
//

import MLX
import Foundation

// MARK: - Window Functions

/// Create a Hann (Hanning) window
///
/// - Parameters:
///   - length: Window length
///   - periodic: If true, create periodic window (for STFT)
/// - Returns: Window coefficients
public func hannWindow(_ length: Int, periodic: Bool = true) -> MLXArray {
    let n = MLXArray(0..<length).asType(.float32)
    let denom = Float(periodic ? length : length - 1)
    return 0.5 * (1 - MLX.cos(2 * Float.pi * n / denom))
}

/// Create a Hamming window
public func hammingWindow(_ length: Int, periodic: Bool = true) -> MLXArray {
    let n = MLXArray(0..<length).asType(.float32)
    let denom = Float(periodic ? length : length - 1)
    return 0.54 - 0.46 * MLX.cos(2 * Float.pi * n / denom)
}

/// Generic window creation function
public func createWindow(winType: String, winLen: Int, periodic: Bool = false) throws -> MLXArray {
    switch winType.lowercased() {
    case "hamming":
        return hammingWindow(winLen, periodic: periodic)
    case "hann", "hanning":
        return hannWindow(winLen, periodic: periodic)
    default:
        throw NSError(domain: "AudioUtils.WindowFunctions", code: 1,
                     userInfo: [NSLocalizedDescriptionKey: "Unsupported window type: \(winType)"])
    }
}

// MARK: - Window Selection

/// Get window by name
public func getWindow(_ window: String, length: Int, periodic: Bool = true) throws -> MLXArray {
    switch window.lowercased() {
    case "hann", "hanning":
        return hannWindow(length, periodic: periodic)
    case "hamming":
        return hammingWindow(length, periodic: periodic)
    default:
        throw NSError(domain: "AudioUtils.WindowFunctions", code: 2,
                     userInfo: [NSLocalizedDescriptionKey: "Unknown window type: \(window)"])
    }
}
