import Foundation
import MLX
import MLXNN

// MARK: - Constants

/// HTK mel scale conversion constant
private let htkMelScaleMultiplier: Float = 2595.0

/// HTK mel scale conversion constant (Hz)
private let htkMelScaleFrequencyBase: Float = 700.0

/// Slaney minimum frequency offset (Hz)
private let slaneyMinFrequency: Float = 0.0

/// Slaney frequency spacing for linear region (Hz/mel)
private let slaneyFrequencySpacing: Float = 200.0 / 3

/// Slaney transition frequency for logarithmic region (Hz)
private let slaneyLogTransitionFrequency: Float = 1000.0

/// Slaney logarithmic step constant
private let slaneyLogStepBase: Float = 6.4

/// Slaney logarithmic step divisor
private let slaneyLogStepDivisor: Float = 27.0

/// Default sample rate divisor for Nyquist frequency
private let sampleRateDivisor: Float = 2.0

/// Magnitude spectrum stability epsilon
private let magnitudeSpectrumEpsilon: Float = 1e-9

// MARK: - Errors

public enum MelSpectrogramError: Error {
    case unsupportedNormalization(String)
}

// MARK: - Thread-safe Caches

/// Thread-safe cache for mel basis matrices
private final class MelBasisCache: @unchecked Sendable {
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

    func clear() {
        lock.lock()
        defer { lock.unlock() }
        cache.removeAll()
    }
}

/// Thread-safe cache for window functions
private final class WindowCache: @unchecked Sendable {
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

    func clear() {
        lock.lock()
        defer { lock.unlock() }
        cache.removeAll()
    }
}

private let melBasisCache = MelBasisCache()
private let hannWindowCache = WindowCache()

// MARK: - Frequency Conversion Functions

/// Convert Hz to Mels
public func hzToMel(_ frequencies: MLXArray, htk: Bool = false) -> MLXArray {
    if htk {
        return htkMelScaleMultiplier * MLX.log10(1.0 + frequencies / htkMelScaleFrequencyBase)
    } else {
        // Slaney's formula
        let minLogMel = (slaneyLogTransitionFrequency - slaneyMinFrequency) / slaneyFrequencySpacing
        let logstep = MLX.log(MLXArray(slaneyLogStepBase)) / slaneyLogStepDivisor

        // For frequencies < 1000 Hz
        var mels = (frequencies - slaneyMinFrequency) / slaneyFrequencySpacing

        // For frequencies >= 1000 Hz
        let logRegion = frequencies .>= MLXArray(slaneyLogTransitionFrequency)
        mels = MLX.where(
            logRegion,
            MLXArray(minLogMel) + MLX.log(frequencies / slaneyLogTransitionFrequency) / logstep,
            mels
        )
        return mels
    }
}

/// Convert Mels to Hz
public func melToHz(_ mels: MLXArray, htk: Bool = false) -> MLXArray {
    if htk {
        return htkMelScaleFrequencyBase * (MLX.pow(10.0, mels / htkMelScaleMultiplier) - 1.0)
    } else {
        // Slaney's formula
        let minLogMel = (slaneyLogTransitionFrequency - slaneyMinFrequency) / slaneyFrequencySpacing
        let logstep = MLX.log(MLXArray(slaneyLogStepBase)) / slaneyLogStepDivisor

        // For mels < min_log_mel
        var freqs = slaneyMinFrequency + slaneyFrequencySpacing * mels

        // For mels >= min_log_mel
        let logRegion = mels .>= MLXArray(minLogMel)
        freqs = MLX.where(
            logRegion,
            MLXArray(slaneyLogTransitionFrequency) * MLX.exp(logstep * (mels - minLogMel)),
            freqs
        )
        return freqs
    }
}

/// Compute FFT bin center frequencies
public func fftFrequencies(sr: Int, nFFT: Int) -> MLXArray {
    return MLX.linspace(0, Float(sr) / sampleRateDivisor, count: 1 + nFFT / 2)
}

/// Compute mel band center frequencies
public func melFrequencies(nMels: Int, fmin: Float = 0.0, fmax: Float = 11025.0, htk: Bool = false) -> MLXArray {
    // Convert frequency limits to mel scale
    let minMel = hzToMel(MLXArray(fmin), htk: htk)
    let maxMel = hzToMel(MLXArray(fmax), htk: htk)
    
    // Equally spaced mel values
    let mels = MLX.linspace(minMel.item(Float.self), maxMel.item(Float.self), count: nMels)
    
    // Convert back to Hz
    return melToHz(mels, htk: htk)
}

/// Create mel filter bank matrix
public func createMelFilterBank(
    sr: Int,
    nFFT: Int,
    nMels: Int = 128,
    fmin: Float = 0.0,
    fmax: Float? = nil,
    htk: Bool = false,
    norm: String? = "slaney"
) throws -> MLXArray {

    let actualFmax = fmax ?? Float(sr) / sampleRateDivisor

    // Center frequencies of each FFT bin
    let fftfreqs = fftFrequencies(sr: sr, nFFT: nFFT)

    // Center frequencies of mel bands - uniformly spaced between limits
    let melF = melFrequencies(nMels: nMels + 2, fmin: fmin, fmax: actualFmax, htk: htk)

    // Calculate differences between adjacent mel frequencies
    let fdiff = melF[1...] - melF[0..<(melF.shape[0] - 1)]

    // Create triangular filters using broadcasting
    let ramps = melF.expandedDimensions(axis: -1) - fftfreqs.expandedDimensions(axis: 0)

    // Vectorized computation using broadcasting
    let lower = -ramps[0..<(ramps.shape[0] - 2)] / fdiff[0..<(fdiff.shape[0] - 1)].expandedDimensions(axis: -1)
    let upper = ramps[2...] / fdiff[1...].expandedDimensions(axis: -1)

    // Intersect them with each other and zero
    var weights = MLX.maximum(MLXArray(0), MLX.minimum(lower, upper))

    // Apply normalization
    if let norm = norm {
        if norm == "slaney" {
            // Slaney-style mel is scaled to be approx constant energy per channel
            let enorm = 2.0 / (melF[2..<(nMels + 2)] - melF[0..<nMels])
            weights = weights * enorm.expandedDimensions(axis: -1)
        } else {
            throw MelSpectrogramError.unsupportedNormalization(norm)
        }
    }
    return weights
}

/// Dynamic range compression
public func dynamicRangeCompression(_ x: MLXArray, C: Float = 1.0, clipVal: Float = 1e-5) -> MLXArray {

    // Apply clipping
    let clipped = MLX.maximum(x, MLXArray(clipVal))

    // Multiply by C
    let scaled = clipped * C

    // Apply log
    let result = MLX.log(scaled)

    return result
}

/// Spectral normalization
public func spectralNormalize(_ magnitudes: MLXArray) -> MLXArray {
    let result = dynamicRangeCompression(magnitudes)
    return result
}

/// Mel spectrogram computation
public func melSpectrogram(
    _ y: MLXArray,
    nFFT: Int,
    numMels: Int,
    samplingRate: Int,
    hopSize: Int,
    winSize: Int,
    fmin: Float,
    fmax: Float,
    center: Bool = false
) throws -> MLXArray {

    // Create cache keys
    let deviceKey = "mlx"
    let fmaxKey = "\(fmax)_\(deviceKey)"

    // Create mel basis if not cached
    if melBasisCache.get(fmaxKey) == nil {
        let mel = try createMelFilterBank(
            sr: samplingRate,
            nFFT: nFFT,
            nMels: numMels,
            fmin: fmin,
            fmax: fmax
        )
        melBasisCache.set(fmaxKey, value: mel)
        hannWindowCache.set(deviceKey, value: hannWindow(winSize, periodic: true))
    }

    let melBasis = melBasisCache.get(fmaxKey)!
    let window = hannWindowCache.get(deviceKey)!

    // Pad signal - torch implementation always pads regardless of center
    var paddedY = y.expandedDimensions(axis: 1)
    let padAmount = (nFFT - hopSize) / 2

    // Use reflection padding approximation
    if paddedY.shape[2] > padAmount {
        let leftPad = paddedY[0..., 0..., 1..<(padAmount + 1)][0..., 0..., .stride(by: -1)]
        let rightPad = paddedY[0..., 0..., -(padAmount + 1)..<(-1)][0..., 0..., .stride(by: -1)]
        paddedY = MLX.concatenated([leftPad, paddedY, rightPad], axis: 2)
    } else {
        // Fall back to constant padding for very short signals
        paddedY = MLX.padded(paddedY, widths: [IntOrPair(0), IntOrPair(0), IntOrPair([padAmount, padAmount])])
    }
    paddedY = paddedY.squeezed(axis: 1)

    // Compute STFT using the existing STFT function
    let (specReal, specImag) = stft(
        paddedY,
        nFFT: nFFT,
        hopLength: hopSize,
        winLength: winSize,
        window: window,
        center: center
    )

    // Convert to magnitude

    // Compute magnitude spectrum on GPU with float32
    let specSquared = specReal * specReal + specImag * specImag

    // Add epsilon
    let specSquaredPlusEps = specSquared + MLXArray(magnitudeSpectrumEpsilon)

    // Compute sqrt
    let spec = MLX.sqrt(specSquaredPlusEps)


    // Apply mel filter bank
    let melSpec = MLX.matmul(melBasis, spec)

    // Apply spectral normalization
    let normalized = spectralNormalize(melSpec)

    return normalized
}

/// Clear mel spectrogram caches
public func clearMelSpectrogramCache() {
    melBasisCache.clear()
    hannWindowCache.clear()
}
