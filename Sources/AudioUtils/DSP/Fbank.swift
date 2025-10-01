import Foundation
import MLX
import Accelerate

// MARK: - Constants

/// Kaldi default dither value for inference
private let defaultDitherValue: Float = 1.0

/// Kaldi preemphasis coefficient
private let kaldiPreemphasisCoefficient: Float = 0.97

/// Minimum frequency for mel filterbank (Hz)
private let defaultLowFrequency: Float = 20.0

/// Numerical stability epsilon
private let numericalStabilityEpsilon: Float = 1e-8

/// Mel scale conversion constant (Hz)
private let melScaleFrequencyBase: Float = 700.0

/// Mel scale conversion constant
private let melScaleLogCoefficient: Float = 1127.0

/// Window function coefficients
private struct WindowCoefficients {
    static let hannCenter: Float = 0.5
    static let hammingAlpha: Float = 0.54
    static let hammingBeta: Float = 0.46
    static let poveyPower: Float = 0.85
}

/// Time conversion factor (seconds to milliseconds)
private let secondsToMilliseconds: Float = 1000.0
private let millisecondsToSeconds: Float = 0.001

// MARK: - Errors

public enum FbankError: Error {
    case invalidWindowType(String)
    case invalidWindowLength(Int, minimum: Int)
}

/// Optimized audio processing functions using MLX vectorized operations
public struct Fbank {

    // MARK: - Configuration
    public struct Args {
        public let samplingRate: Int
        public let winLen: Int
        public let winInc: Int
        public let numMels: Int
        public let winType: String
        
        public init(samplingRate: Int = 16000, winLen: Int = 400, winInc: Int = 160, numMels: Int = 80, winType: String = "hann") {
            self.samplingRate = samplingRate
            self.winLen = winLen
            self.winInc = winInc
            self.numMels = numMels
            self.winType = winType
        }
    }
    
    // MARK: - Thread-safe caches

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

        var count: Int {
            lock.lock()
            defer { lock.unlock() }
            return cache.count
        }
    }

    /// Thread-safe cache for mel filterbanks
    private final class MelFilterbankCache: @unchecked Sendable {
        private var cache: [String: (MLXArray, MLXArray)] = [:]
        private let lock = NSLock()

        func get(_ key: String) -> (MLXArray, MLXArray)? {
            lock.lock()
            defer { lock.unlock() }
            return cache[key]
        }

        func set(_ key: String, value: (MLXArray, MLXArray)) {
            lock.lock()
            defer { lock.unlock() }
            cache[key] = value
        }

        func clear() {
            lock.lock()
            defer { lock.unlock() }
            cache.removeAll()
        }

        var count: Int {
            lock.lock()
            defer { lock.unlock() }
            return cache.count
        }
    }

    private static let windowCache = WindowCache()
    private static let melFilterbankCache = MelFilterbankCache()
    
    // MARK: - Optimized Filterbank Computation
    
    /// Highly optimized mel-filterbank feature computation
    /// - Parameters:
    ///   - audioIn: Input audio tensor of shape [batch, samples] or [samples]
    ///   - args: Configuration arguments
    /// - Returns: Log mel-filterbank features
    public static func computeFbank(
        _ audioIn: MLXArray,
        args: Args
    ) -> MLXArray {
        // Handle input dimensions
        let audio: MLXArray
        if audioIn.ndim == 2 {
            audio = audioIn[0]
        } else {
            audio = audioIn
        }
        
        // Extract parameters
        let sampleFrequency = Float(args.samplingRate)
        let frameLength = Float(args.winLen) / Float(args.samplingRate) * secondsToMilliseconds
        let frameShift = Float(args.winInc) / Float(args.samplingRate) * secondsToMilliseconds
        let numMelBins = args.numMels
        let windowType = args.winType

        // Kaldi default parameters
        let dither: Float = defaultDitherValue
        let preemphasisCoefficient: Float = kaldiPreemphasisCoefficient
        let removeDCOffset = true
        let roundToPowerOfTwo = true
        let snipEdges = true
        let useLogFbank = true
        let usePower = true
        let lowFreq: Float = defaultLowFrequency
        let highFreq: Float = 0.0

        // Window properties
        let windowShiftSamples = Int(sampleFrequency * frameShift * millisecondsToSeconds)
        let windowSize = Int(sampleFrequency * frameLength * millisecondsToSeconds)
        let paddedWindowSize = roundToPowerOfTwo ? nextPowerOf2(windowSize) : windowSize
        
        // Use native MLX.asStrided for frame extraction
        let frames = getOptimizedFrames(
            waveform: audio,
            windowSize: windowSize,
            windowShift: windowShiftSamples,
            snipEdges: snipEdges
        )
        
        if frames.shape[0] == 0 {
            return MLXArray.zeros([0, numMelBins])
        }
        
        // Apply dither
        var processedFrames = frames
        if dither != 0.0 {
            // Note: Original implementation doesn't set a fixed seed, using random dither each time
            // MLXRandom.seed(42) // Uncomment for reproducible results
            let randGauss = MLXRandom.normal(frames.shape) * dither
            processedFrames = processedFrames + randGauss
        }
        
        // Remove DC offset - vectorized
        if removeDCOffset {
            let rowMeans = processedFrames.mean(axis: 1, keepDims: true)
            processedFrames = processedFrames - rowMeans
        }
        
        // Apply preemphasis - vectorized
        if preemphasisCoefficient != 0.0 {
            let firstCol = processedFrames[0..., 0..<1]
            let otherCols = processedFrames[0..., 1...] - preemphasisCoefficient * processedFrames[0..., 0..<(processedFrames.shape[1] - 1)]
            processedFrames = MLX.concatenated([firstCol, otherCols], axis: 1)
        }
        
        //Use cached window function
        let window = getCachedWindow(windowType: windowType, windowSize: windowSize)
        processedFrames = processedFrames * window
        
        // Pad to paddedWindowSize if needed
        if paddedWindowSize > windowSize {
            let padding = paddedWindowSize - windowSize
            processedFrames = MLX.padded(processedFrames, widths: [IntOrPair(0), IntOrPair([0, padding])])
        }
        
        // Compute FFT - vectorized for all frames at once
        let fftResult = rfft(processedFrames, n: paddedWindowSize, axis: 1)
        var spectrum = MLX.abs(fftResult)
        
        // Power spectrum
        if usePower {
            spectrum = spectrum ** 2.0
        }
        
        // Use cached mel filterbank
        let (melEnergies, _) = getCachedMelBanks(
            numBins: numMelBins,
            windowLengthPadded: paddedWindowSize,
            sampleFreq: sampleFrequency,
            lowFreq: lowFreq,
            highFreq: highFreq
        )
        
        // Kaldi pads mel energies with a zero column on the right
        let paddedMelEnergies = MLX.padded(melEnergies, widths: [IntOrPair(0), IntOrPair([0, 1])])
        
        // Apply mel filterbank - vectorized matrix multiplication
        var melFeatures = MLX.matmul(spectrum, paddedMelEnergies.T)
        
        // Apply log
        if useLogFbank {
            melFeatures = MLX.log(MLX.maximum(melFeatures, numericalStabilityEpsilon))
        }
        
        return melFeatures
    }
    
    // MARK: - Optimized Delta Computation

    /// Optimized delta computation using vectorized operations
    /// Supports both 2D (frames x features) and 3D (batch x frames x features) inputs
    public static func computeDeltas(
        _ specgram: MLXArray,
        winLength: Int = 5,
        mode: String = "edge"
    ) throws -> MLXArray {
        guard winLength >= 3 else {
            throw FbankError.invalidWindowLength(winLength, minimum: 3)
        }
        
        // Check for empty input
        if specgram.size == 0 {
            return specgram
        }
        
        let originalShape = specgram.shape
        let isBatch = specgram.ndim == 3
        
        // Check for degenerate shapes
        if (isBatch && (originalShape[1] == 0 || originalShape[2] == 0)) ||
           (!isBatch && (originalShape[0] == 0 || (originalShape.count > 1 && originalShape[1] == 0))) {
            return specgram
        }
        
        // For batch processing, we need to handle the reshape differently
        let reshapedSpecgram: MLXArray
        if isBatch {
            // For 3D input (batch, frames, features), process each batch item
            reshapedSpecgram = specgram.reshaped([originalShape[0] * originalShape[1], originalShape[2]])
        } else {
            // For 2D input (frames, features), keep original logic
            reshapedSpecgram = specgram.reshaped([-1, originalShape[originalShape.count - 1]])
        }
        
        let n = (winLength - 1) / 2
        let denom = Float(n * (n + 1) * (2 * n + 1)) / 3.0
        
        // Pad the specgram
        let padded: MLXArray
        if mode == "edge" {
            // Vectorized edge padding
            let padLeft = MLX.repeated(reshapedSpecgram[0..., 0..<1], count: n, axis: 1)
            let lastCol = reshapedSpecgram[0..., -1..<reshapedSpecgram.shape[1]]
            let padRight = MLX.repeated(lastCol, count: n, axis: 1)
            padded = MLX.concatenated([padLeft, reshapedSpecgram, padRight], axis: 1)
        } else {
            // Constant padding with zeros
            padded = MLX.padded(reshapedSpecgram, widths: [IntOrPair(0), IntOrPair(n)])
        }
        
        // Create kernel weights
        let kernelWeights = MLXArray(-n...n).asType(padded.dtype)
        
        // Vectorized delta computation using strided views
        let timeSteps = padded.shape[1] - 2 * n
        
        // Create strided view for all windows at once
        let windows = MLX.asStrided(
            padded,
            [reshapedSpecgram.shape[0], timeSteps, winLength],
            strides: [padded.shape[1], 1, 1]
        )
        
        // Compute weighted sum for all windows at once
        let weighted = windows * kernelWeights
        let output = weighted.sum(axis: 2) / denom
        
        // Reshape back to original shape
        return output.reshaped(originalShape)
    }
    
    // MARK: - Helper Functions
    
    private static func getOptimizedFrames(
        waveform: MLXArray,
        windowSize: Int,
        windowShift: Int,
        snipEdges: Bool
    ) -> MLXArray {
        let numSamples = waveform.shape[0]
        
        if snipEdges {
            if numSamples < windowSize {
                return MLXArray.zeros([0, 0])
            }
            
            let m = 1 + (numSamples - windowSize) / windowShift
            
            // Use native MLX.asStrided for efficient frame extraction
            return MLX.asStrided(waveform, [m, windowSize], strides: [windowShift, 1])
            
        } else {
            // Reflect padding
            let m = (numSamples + (windowShift / 2)) / windowShift
            let pad = windowSize / 2 - windowShift / 2
            
            var paddedWaveform = waveform
            if pad > 0 && numSamples > pad {
                // Optimized reflection padding
                let actualLeftPad = min(pad, numSamples - 1)
                let leftIndices = MLXArray(Array(1...actualLeftPad).reversed().map { Int32($0) })
                let padLeft = waveform.take(leftIndices, axis: 0)
                
                let actualRightPad = min(pad, numSamples - 1)
                let rightStart = max(0, numSamples - actualRightPad - 1)
                let rightIndices = MLXArray((rightStart..<(numSamples - 1)).reversed().map { Int32($0) })
                let padRight = waveform.take(rightIndices, axis: 0)
                
                paddedWaveform = MLX.concatenated([padLeft, waveform, padRight])
            } else if pad > 0 {
                // Zero padding for short signals
                let zeroPad = MLXArray.zeros([pad], dtype: waveform.dtype)
                paddedWaveform = MLX.concatenated([zeroPad, waveform, zeroPad])
            } else if pad < 0 {
                // Trim from the front
                let trimAmount = min(-pad, numSamples)
                if trimAmount < numSamples {
                    paddedWaveform = waveform[trimAmount...]
                }
            }
            
            // Use native MLX.asStrided for efficient frame extraction
            return MLX.asStrided(paddedWaveform, [m, windowSize], strides: [windowShift, 1])
        }
    }
    
    private static func nextPowerOf2(_ x: Int) -> Int {
        if x <= 0 { return 1 }
        if x == 1 { return 1 }
        
        var power = 1
        while power < x {
            power *= 2
        }
        return power
    }
    
    private static func getCachedWindow(
        windowType: String,
        windowSize: Int,
        periodic: Bool = false
    ) -> MLXArray {
        let cacheKey = "\(windowType)_\(windowSize)_\(periodic)"

        // Check cache first
        if let cachedWindow = windowCache.get(cacheKey) {
            return cachedWindow
        }

        let n = MLXArray(0..<windowSize)
        let window: MLXArray

        switch windowType {
        case "hanning":
            let divisor = periodic ? Float(windowSize) : Float(windowSize - 1)
            window = WindowCoefficients.hannCenter - WindowCoefficients.hannCenter * MLX.cos(2 * Float.pi * n / divisor)

        case "hamming":
            let divisor = Float(windowSize - 1)
            window = WindowCoefficients.hammingAlpha - WindowCoefficients.hammingBeta * MLX.cos(2 * Float.pi * n / divisor)

        case "povey":
            let hann = WindowCoefficients.hannCenter - WindowCoefficients.hannCenter * MLX.cos(2 * Float.pi * n / Float(windowSize - 1))
            window = MLX.pow(hann, WindowCoefficients.poveyPower)

        case "rectangular":
            window = MLXArray.ones([windowSize])

        default:
            window = MLXArray.ones([windowSize])
        }

        // Cache the window
        windowCache.set(cacheKey, value: window)
        return window
    }
    
    private static func melScale(_ freq: MLXArray) -> MLXArray {
        return melScaleLogCoefficient * MLX.log(1.0 + freq / melScaleFrequencyBase)
    }

    private static func inverseMelScale(_ melFreq: MLXArray) -> MLXArray {
        return melScaleFrequencyBase * (MLX.exp(melFreq / melScaleLogCoefficient) - 1.0)
    }
    
    private static func getCachedMelBanks(
        numBins: Int,
        windowLengthPadded: Int,
        sampleFreq: Float,
        lowFreq: Float,
        highFreq: Float
    ) -> (MLXArray, MLXArray) {
        let cacheKey = "\(numBins)_\(windowLengthPadded)_\(sampleFreq)_\(lowFreq)_\(highFreq)"

        // Check cache first
        if let cached = melFilterbankCache.get(cacheKey) {
            return cached
        }

        // Compute mel filterbank
        let result = getMelBanks(
            numBins: numBins,
            windowLengthPadded: windowLengthPadded,
            sampleFreq: sampleFreq,
            lowFreq: lowFreq,
            highFreq: highFreq
        )

        // Cache the result
        melFilterbankCache.set(cacheKey, value: result)
        return result
    }
    
    private static func getMelBanks(
        numBins: Int,
        windowLengthPadded: Int,
        sampleFreq: Float,
        lowFreq: Float,
        highFreq: Float
    ) -> (MLXArray, MLXArray) {
        assert(numBins > 3, "Must have at least 3 mel bins")
        assert(windowLengthPadded % 2 == 0)
        
        let numFftBins = windowLengthPadded / 2
        let nyquist = 0.5 * sampleFreq
        
        var actualHighFreq = highFreq
        if highFreq <= 0.0 {
            actualHighFreq = highFreq + nyquist
        }
        
        assert(0.0 <= lowFreq && lowFreq < nyquist)
        assert(0.0 < actualHighFreq && actualHighFreq <= nyquist)
        assert(lowFreq < actualHighFreq)
        
        let fftBinWidth = sampleFreq / Float(windowLengthPadded)
        
        // Batch convert frequencies to mel scale to avoid individual .item() calls
        let frequencies = MLXArray([lowFreq, actualHighFreq])
        let melFrequencies = melScale(frequencies)
        // eval() not needed - asArray() will trigger evaluation
        let melFreqArray = melFrequencies.asArray(Float.self)
        let melLowFreq = melFreqArray[0]
        let melHighFreq = melFreqArray[1]
        
        let melFreqDelta = (melHighFreq - melLowFreq) / Float(numBins + 1)
        
        // Create mel points - vectorized
        let binIdx = MLXArray(0..<numBins).reshaped([-1, 1])
        let leftMel = melLowFreq + binIdx * melFreqDelta
        let centerMel = melLowFreq + (binIdx + 1.0) * melFreqDelta
        let rightMel = melLowFreq + (binIdx + 2.0) * melFreqDelta
        
        let centerFreqs = inverseMelScale(centerMel)
        
        // Create frequency grid for all FFT bins - vectorized
        let mel = melScale(fftBinWidth * MLXArray(0..<numFftBins)).reshaped([1, -1])
        
        // Calculate filter responses - vectorized
        let upSlope = (mel - leftMel) / (centerMel - leftMel)
        let downSlope = (rightMel - mel) / (rightMel - centerMel)
        
        // Combine slopes, taking minimum and clamping to [0, 1]
        let bins = MLX.maximum(MLXArray.zeros(upSlope.shape), MLX.minimum(upSlope, downSlope))
        
        return (bins, centerFreqs.squeezed())
    }
    
    // MARK: - Cache Management

    /// Clear all caches
    public static func clearCaches() {
        windowCache.clear()
        melFilterbankCache.clear()
    }

    /// Get cache information
    public static func getCacheInfo() -> (windowCount: Int, melFilterbankCount: Int) {
        let windowCount = windowCache.count
        let melCount = melFilterbankCache.count
        return (windowCount, melCount)
    }
}
