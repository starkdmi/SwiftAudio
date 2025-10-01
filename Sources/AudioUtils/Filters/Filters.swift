//
//  Filters.swift
//  AudioUtils
//
//  High-performance audio filters with Metal optimization
//

import MLX
import MLXFast
import Foundation

// MARK: - Error Types

public enum FiltersError: Error {
    case unsupportedFilterType(String)
    case signalTooShort
    case invalidSamplingFrequency
}

// MARK: - Filter Design

/// Design a Butterworth filter
///
/// - Parameters:
///   - order: Filter order
///   - cutoff: Normalized cutoff frequency (0 to 1)
///   - filterType: Filter type ("low", "high", "band", "stop")
///   - fs: Sampling frequency (optional, for denormalized cutoff)
/// - Returns: Tuple of (b, a) filter coefficients
public func butterworth(
    order: Int,
    cutoff: Float,
    filterType: String = "low",
    fs: Float? = nil
) throws -> (b: MLXArray, a: MLXArray) {
    // Normalize cutoff if fs is provided
    let wn: Float
    if let fs = fs {
        guard fs > 0 else {
            throw FiltersError.invalidSamplingFrequency
        }
        wn = cutoff / (fs / 2.0)
    } else {
        wn = cutoff
    }
    
    // Generate analog Butterworth prototype poles
    let k = MLXArray(1...(order)).asType(.float32)
    let theta = Float.pi * (2 * k - 1) / (2 * Float(order))
    
    // s-plane poles on unit circle
    let poles_s = -MLX.sin(theta) + (-MLX.cos(theta)).asImaginary()
    
    // Prewarp the frequency for bilinear transform
    let fsNorm: Float = 2.0
    let warped = 2.0 * fsNorm * tan(Float.pi * wn / 2.0)
    
    // Apply frequency transformation
    let poles_s_transformed: MLXArray
    switch filterType {
    case "low":
        poles_s_transformed = poles_s * warped
    case "high":
        poles_s_transformed = warped / poles_s
    default:
        throw FiltersError.unsupportedFilterType(filterType)
    }
    
    // Bilinear transform
    let poles_z = (2*fsNorm + poles_s_transformed) / (2*fsNorm - poles_s_transformed)
    
    // Zeros in z-domain
    let zeros_z: MLXArray
    switch filterType {
    case "low":
        zeros_z = MLXArray.ones([order]) * -1.0  // at z=-1
    case "high":
        zeros_z = MLXArray.ones([order])         // at z=1
    default:
        zeros_z = MLXArray.ones([order]) * -1.0
    }
    
    // Build polynomials
    var b = MLXArray([Float(1.0)])
    var a = MLXArray([Float(1.0)])
    
    // Numerator: product of (z - zero_i)
    for i in 0..<order {
        let zero = zeros_z[i]
        let poly = MLXArray([1.0, -zero.item(Float.self)])
        b = polyMultiply(b, poly)
    }
    
    // Denominator: product of (z - pole_i)
    for i in 0..<order {
        let pole = poles_z[i]
        let one = MLXArray(real: 1.0, imaginary: 0.0)
        let poly = stacked([one, -pole], axis: 0)
        a = polyMultiply(a, poly)
    }
    a = a.realPart()
    
    // Calculate gain for proper normalization
    switch filterType {
    case "low":
        // Unity gain at DC (z=1)
        let b_sum = MLX.sum(b)
        let a_sum = MLX.sum(a)
        let gain = a_sum / b_sum
        b = b * gain
    case "high":
        // Unity gain at Nyquist (z=-1)
        let signs = MLXArray((0..<b.shape[0]).map { Float(($0 % 2 == 0) ? 1 : -1) })
        let b_sum = MLX.sum(b * signs)
        let a_sum = MLX.sum(a * signs)
        let gain = a_sum / b_sum
        b = b * gain
    default:
        break
    }
    
    // Normalize by a[0]
    let a0 = a[0]
    b = b / a0
    a = a / a0
    
    return (b, a)
}

// MARK: - Filter Application

/// Apply a digital filter forward and backward (zero-phase filtering)
///
/// - Parameters:
///   - b: Numerator coefficients
///   - a: Denominator coefficients
///   - signal: Input signal
/// - Returns: Filtered signal
public func filtfilt(_ b: MLXArray, _ a: MLXArray, _ signal: MLXArray) throws -> MLXArray {
    // Minimum padding multiplier for edge reflection
    let paddingMultiplier = 3
    let padlen = paddingMultiplier * max(a.shape[0], b.shape[0])
    let actualPadlen = min(padlen, signal.shape[0] - 1)

    guard actualPadlen >= 1 else {
        throw FiltersError.signalTooShort
    }
    
    // Extrapolate signal at edges using odd reflection
    let x0 = signal[0]
    let xn = signal[signal.shape[0] - 1]
    
    // Reflect and extrapolate
    let pre = 2 * x0 - signal[.stride(from: actualPadlen, to: 0, by: -1)]
    let post = 2 * xn - signal[.stride(from: signal.shape[0]-2, to: signal.shape[0]-actualPadlen-2, by: -1)]
    let xExt = MLX.concatenated([pre, signal, post], axis: 0)
    
    // Get initial conditions
    let zi = lfilterZi(b, a)
    
    // Forward pass
    let xExt0 = xExt[0]
    let (y1, _) = lfilter(b, a, xExt, zi: zi * xExt0)
    
    // Backward pass
    let yRev = y1[.stride(by: -1)]
    let yRev0 = yRev[0]
    let (y2, _) = lfilter(b, a, yRev, zi: zi * yRev0)
    
    // Extract original portion
    let y = y2[.stride(by: -1)]
    return y[actualPadlen..<(y.shape[0]-actualPadlen)]
}

/// Direct Form II Transposed digital filter implementation.
public func lfilter(_ b: MLXArray, _ a: MLXArray, _ x: MLXArray, zi: MLXArray? = nil) -> (MLXArray, MLXArray) {
    let n = x.shape[0]
    let nfilt = max(a.shape[0], b.shape[0])
    
    // Pad coefficients
    var bPadded = b
    var aPadded = a
    if b.shape[0] < nfilt {
        bPadded = MLX.padded(b, widths: [IntOrPair([0, nfilt - b.shape[0]])])
    }
    if a.shape[0] < nfilt {
        aPadded = MLX.padded(a, widths: [IntOrPair([0, nfilt - a.shape[0]])])
    }
    
    // Normalize by a[0]
    let a0 = aPadded[0].item(Float.self)
    bPadded = bPadded / a0
    aPadded = aPadded / a0
    
    // Special case: FIR filter
    if nfilt == 1 {
        return (bPadded[0] * x, MLXArray([]))
    }
    
    // Check if it's effectively an FIR filter
    let firThreshold: Float = 1e-10
    let aMax = MLX.max(MLX.abs(aPadded[1...])).item(Float.self)
    if aMax < firThreshold {
        // Pure FIR filter - use convolution which is much faster
        let kernel = bPadded[.stride(by: -1)]
        let xPadded = MLX.padded(x, widths: [IntOrPair([nfilt-1, 0])], mode: .constant, value: MLXArray(0))
        let y = MLX.convolve(xPadded, kernel, mode: .valid)
        return (y[0..<n], MLXArray.zeros([nfilt-1]))
    }

    // Minimum signal length for Metal optimization
    let metalThreshold = 1000
    if n > metalThreshold {
        // Try specialized 4th-order kernel first (common case for butter(N: 4))
        if nfilt == 5,
           let result = lfilterMetal4thOrder(bPadded, aPadded, x, zi: zi) {
            return result
        }
        
        // Try general Metal kernel
        if let result = lfilterMetal(bPadded, aPadded, x, zi: zi) {
            return result
        }
        // Fall back to CPU implementation if Metal fails
    }
    
    // For IIR filters, we must process sequentially
    // Convert to Swift arrays for sequential processing
    // Use asArray for batch conversion which is much faster than individual item() calls
    // Note: asArray() implicitly evaluates the arrays, so no need for explicit eval()
    let xVals = x.asArray(Float.self)
    let bVals = bPadded.asArray(Float.self)
    let aVals = aPadded.asArray(Float.self)
    
    // Initialize state
    var zVals: [Float]
    if let zi = zi {
        // asArray() implicitly evaluates the array
        zVals = zi.asArray(Float.self)
    } else {
        zVals = Array(repeating: 0.0, count: nfilt - 1)
    }
    
    // Pre-allocate output
    var yVals = Array(repeating: Float(0.0), count: n)
    
    // Optimize for common filter orders
    if nfilt == 2 {  // 1st order filter
        var z0 = zVals[0]
        let b0 = bVals[0], b1 = bVals[1]
        let a1 = aVals[1]
        for i in 0..<n {
            yVals[i] = b0 * xVals[i] + z0
            z0 = b1 * xVals[i] - a1 * yVals[i]
        }
        zVals[0] = z0
    } else if nfilt == 3 {  // 2nd order filter
        var z0 = zVals[0], z1 = zVals[1]
        let b0 = bVals[0], b1 = bVals[1], b2 = bVals[2]
        let a1 = aVals[1], a2 = aVals[2]
        for i in 0..<n {
            yVals[i] = b0 * xVals[i] + z0
            z0 = b1 * xVals[i] + z1 - a1 * yVals[i]
            z1 = b2 * xVals[i] - a2 * yVals[i]
        }
        zVals[0] = z0
        zVals[1] = z1
    } else if nfilt == 5 {  // 4th order filter (common for butter(4))
        var z0 = zVals[0], z1 = zVals[1], z2 = zVals[2], z3 = zVals[3]
        let b0 = bVals[0], b1 = bVals[1], b2 = bVals[2], b3 = bVals[3], b4 = bVals[4]
        let a1 = aVals[1], a2 = aVals[2], a3 = aVals[3], a4 = aVals[4]
        for i in 0..<n {
            yVals[i] = b0 * xVals[i] + z0
            z0 = b1 * xVals[i] + z1 - a1 * yVals[i]
            z1 = b2 * xVals[i] + z2 - a2 * yVals[i]
            z2 = b3 * xVals[i] + z3 - a3 * yVals[i]
            z3 = b4 * xVals[i] - a4 * yVals[i]
        }
        zVals[0] = z0
        zVals[1] = z1
        zVals[2] = z2
        zVals[3] = z3
    } else {
        // General case
        for i in 0..<n {
            yVals[i] = bVals[0] * xVals[i] + zVals[0]
            
            // Update states
            for j in 0..<(nfilt-2) {
                zVals[j] = bVals[j+1] * xVals[i] + zVals[j+1] - aVals[j+1] * yVals[i]
            }
            
            zVals[nfilt-2] = bVals[nfilt-1] * xVals[i] - aVals[nfilt-1] * yVals[i]
        }
    }
    
    // Convert back to MLXArray
    let y = MLXArray(yVals)
    let zf = MLXArray(zVals)
    
    return (y, zf)
}

/// Compute initial conditions for lfilter
private func lfilterZi(_ b: MLXArray, _ a: MLXArray) -> MLXArray {
    let n = max(b.shape[0], a.shape[0])

    if n == 1 {
        return MLXArray([])
    }

    // Pad coefficients
    var bPadded = b
    var aPadded = a
    if b.shape[0] < n {
        bPadded = MLX.padded(b, widths: [IntOrPair([0, n - b.shape[0]])])
    }
    if a.shape[0] < n {
        aPadded = MLX.padded(a, widths: [IntOrPair([0, n - a.shape[0]])])
    }

    // Normalize
    let a0 = aPadded[0].item(Float.self)
    let aNorm = aPadded / a0
    let bNorm = bPadded / a0

    // Simplified computation for initial conditions
    let zi = MLXArray.zeros([n-1])

    // Compute steady-state values
    let sumA = MLX.sum(aNorm).item(Float.self)
    let sumB = MLX.sum(bNorm).item(Float.self)

    // Minimum threshold for DC gain computation
    let dcGainThreshold: Float = 1e-6
    if abs(sumA) > dcGainThreshold {
        let dcGain = sumB / sumA
        let b0 = bNorm[0].item(Float.self)

        // Distribute initial state
        for i in 0..<(n-1) {
            let weight = Float(n - 1 - i) / Float(n - 1)
            zi[i] = MLXArray(dcGain * weight * b0)
        }
    }

    return zi
}

// MARK: - Metal Kernel Implementation

/// IIR filter implementation using Metal kernel
private func lfilterMetal(_ b: MLXArray, _ a: MLXArray, _ x: MLXArray, zi: MLXArray? = nil) -> (MLXArray, MLXArray)? {
    // let n = x.shape[0]
    let nfilt = max(a.shape[0], b.shape[0])

    // Maximum filter order supported by Metal kernel
    let maxMetalFilterOrder = 16
    guard nfilt <= maxMetalFilterOrder else {
        return nil
    }
    
    // Initialize state
    let state = zi ?? MLXArray.zeros([nfilt - 1])
    
    let source = """
        // Single thread processes all samples sequentially
        uint tid = thread_position_in_grid.x;
        if (tid != 0) return;
        
        const int n_samples = x_shape[0];
        const int n_state = state_shape[0];
        const int nfilt = n_state + 1;
        
        // Create working copy of state
        T z[16];  // Max filter order 16
        for (int i = 0; i < n_state; i++) {
            z[i] = state[i];
        }
        
        // Process all samples sequentially
        for (int i = 0; i < n_samples; i++) {
            T x_val = x[i];
            T y_val = b[0] * x_val + z[0];
            
            // Update state (Direct Form II Transposed)
            for (int j = 0; j < n_state - 1; j++) {
                z[j] = b[j + 1] * x_val + z[j + 1] - a[j + 1] * y_val;
            }
            if (n_state > 0) {
                z[n_state - 1] = b[nfilt - 1] * x_val - a[nfilt - 1] * y_val;
            }
            
            // Write output
            y[i] = y_val;
        }
        
        // Copy final state to output
        for (int i = 0; i < n_state; i++) {
            final_state[i] = z[i];
        }
    """
    
    let kernel = MLXFast.metalKernel(
        name: "iir_filter_sequential",
        inputNames: ["x", "b", "a", "state"],
        outputNames: ["y", "final_state"],
        source: source
    )
    
    let outputs = kernel(
        [x, b, a, state],
        template: [("T", DType.float32)],
        grid: (1, 1, 1),
        threadGroup: (1, 1, 1),
        outputShapes: [x.shape, state.shape],
        outputDTypes: [.float32, .float32]
    )
    
    return (outputs[0], outputs[1])
}

/// Specialized Metal kernel for 4th-order Butterworth filters.
/// Optimized for the common case of butter(N: 4) used in lowpass/highpass filters.
func lfilterMetal4thOrder(_ b: MLXArray, _ a: MLXArray, _ x: MLXArray, zi: MLXArray? = nil) -> (MLXArray, MLXArray)? {
    // let n = x.shape[0]
    // 4th order filter coefficients length
    let fourthOrderCoeffsLength = 5
    let fourthOrderStateLength = 4
    guard b.shape[0] == fourthOrderCoeffsLength && a.shape[0] == fourthOrderCoeffsLength else {
        return nil
    }

    // Initialize state
    let state = zi ?? MLXArray.zeros([fourthOrderStateLength])
    
    let source = """
        // Single thread processes all samples sequentially
        uint tid = thread_position_in_grid.x;
        if (tid != 0) return;  // Only first thread works
        
        const int n_samples = x_shape[0];
        
        // Load coefficients into registers for better performance
        T b0 = b[0], b1 = b[1], b2 = b[2], b3 = b[3], b4 = b[4];
        T a1 = a[1], a2 = a[2], a3 = a[3], a4 = a[4];
        
        // Load initial state
        T z0 = state[0], z1 = state[1], z2 = state[2], z3 = state[3];
        
        // Unrolled loop for 4th order filter
        for (int i = 0; i < n_samples; i++) {
            T x_val = x[i];
            T y_val = b0 * x_val + z0;
            
            // Direct Form II Transposed for 4th order
            z0 = b1 * x_val + z1 - a1 * y_val;
            z1 = b2 * x_val + z2 - a2 * y_val;
            z2 = b3 * x_val + z3 - a3 * y_val;
            z3 = b4 * x_val - a4 * y_val;
            
            y[i] = y_val;
        }
        
        // Save final state
        final_state[0] = z0;
        final_state[1] = z1;
        final_state[2] = z2;
        final_state[3] = z3;
    """
    
    let kernel = MLXFast.metalKernel(
        name: "iir_filter_4th_order",
        inputNames: ["x", "b", "a", "state"],
        outputNames: ["y", "final_state"],
        source: source
    )
    
    let outputs = kernel(
        [x, b, a, state],
        template: [("T", DType.float32)],
        grid: (1, 1, 1),  // Single thread
        threadGroup: (1, 1, 1),  // Single thread
        outputShapes: [x.shape, state.shape],
        outputDTypes: [.float32, .float32]
    )
    
    return (outputs[0], outputs[1])
}

// MARK: - High-Level Filter Functions

/// Apply a lowpass Butterworth filter
/// - Note: This function expects a 1D audio signal. For multi-channel audio, process each channel separately.
public func lowpassFilter(
    _ signal: MLXArray,
    cutoff: Float,
    fs: Float,
    order: Int = 4
) throws -> MLXArray {
    let nyquist = 0.5 * fs
    // Avoid numerical issues at frequency boundaries
    let minNormalizedFreq: Float = 0.001
    let maxNormalizedFreq: Float = 0.999
    var wn = cutoff / nyquist
    wn = max(minNormalizedFreq, min(maxNormalizedFreq, wn))

    let (b, a) = try butterworth(order: order, cutoff: wn, filterType: "low")
    return try filtfilt(b, a, signal)
}

/// Apply a highpass Butterworth filter
/// - Note: This function expects a 1D audio signal. For multi-channel audio, process each channel separately.
public func highpassFilter(
    _ signal: MLXArray,
    cutoff: Float,
    fs: Float,
    order: Int = 4
) throws -> MLXArray {
    let nyquist = 0.5 * fs
    // Avoid numerical issues at frequency boundaries
    let minNormalizedFreq: Float = 0.001
    let maxNormalizedFreq: Float = 0.999
    var wn = cutoff / nyquist
    wn = max(minNormalizedFreq, min(maxNormalizedFreq, wn))

    let (b, a) = try butterworth(order: order, cutoff: wn, filterType: "high")
    return try filtfilt(b, a, signal)
}

/// Apply a bandpass Butterworth filter
/// - Note: This function expects a 1D audio signal. For multi-channel audio, process each channel separately.
public func bandpassFilter(
    _ signal: MLXArray,
    lowCutoff: Float,
    highCutoff: Float,
    fs: Float,
    order: Int = 4
) throws -> MLXArray {
    // Apply highpass then lowpass
    let highpassed = try highpassFilter(signal, cutoff: lowCutoff, fs: fs, order: order)
    return try lowpassFilter(highpassed, cutoff: highCutoff, fs: fs, order: order)
}

// MARK: - Utility Functions

/// Polynomial multiplication using convolution
private func polyMultiply(_ p1: MLXArray, _ p2: MLXArray) -> MLXArray {
    let n1 = p1.shape[0]
    let n2 = p2.shape[0]
    
    if n1 == 0 || n2 == 0 {
        return MLXArray([])
    }
    
    // Check if complex
    let isComplex = p1.dtype == .complex64 || p2.dtype == .complex64
    let dtype: DType = isComplex ? .complex64 : .float32
    
    let p1Cast = p1.asType(dtype)
    let p2Cast = p2.asType(dtype)
    
    if isComplex {
        // Handle complex polynomial multiplication
        let p1Real = p1Cast.realPart()
        let p1Imag = p1Cast.imaginaryPart()
        let p2Real = p2Cast.realPart()
        let p2Imag = p2Cast.imaginaryPart()
        
        let realReal = MLX.convolve(p1Real, p2Real, mode: .full)
        let imagImag = MLX.convolve(p1Imag, p2Imag, mode: .full)
        let realImag = MLX.convolve(p1Real, p2Imag, mode: .full)
        let imagReal = MLX.convolve(p1Imag, p2Real, mode: .full)
        
        let resultReal = realReal - imagImag
        let resultImag = realImag + imagReal
        
        return resultReal + resultImag.asImaginary()
    } else {
        return MLX.convolve(p1Cast, p2Cast, mode: .full)
    }
}
