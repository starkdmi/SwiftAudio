//
//  AudioSaver.swift
//  AudioUtils
//
//  Audio saving functionality with format support
//

import Foundation
import AVFoundation
import MLX

// MARK: - Audio Saver

/// Saves MLX audio arrays to various audio file formats
public class AudioSaver {
    
    // MARK: - Configuration
    
    public struct Configuration {
        public let sampleRate: Double
        public let bitDepth: BitDepth
        public let fileFormat: FileFormat
        
        public enum BitDepth {
            case int16
            case int24
            case int32
            case float32
            
            var avFormat: AVAudioCommonFormat {
                switch self {
                case .int16: return .pcmFormatInt16
                case .int24: return .pcmFormatInt32  // 24-bit uses 32-bit container
                case .int32: return .pcmFormatInt32
                case .float32: return .pcmFormatFloat32
                }
            }
        }
        
        public enum FileFormat {
            case wav
            case m4a(bitRate: Int)
            case aiff
            case caf
            
            var fileExtension: String {
                switch self {
                case .wav: return "wav"
                case .m4a: return "m4a"
                case .aiff: return "aiff"
                case .caf: return "caf"
                }
            }
        }
        
        public init(
            sampleRate: Double = 48000.0,
            bitDepth: BitDepth = .float32,
            fileFormat: FileFormat = .wav
        ) {
            self.sampleRate = sampleRate
            self.bitDepth = bitDepth
            self.fileFormat = fileFormat
        }
    }
    
    // MARK: - Properties
    
    private let config: Configuration
    
    // MARK: - Initialization
    
    public init(config: Configuration = Configuration()) {
        self.config = config
    }
    
    // MARK: - Public Methods
    
    /// Save audio to file
    public func save(_ audio: MLXArray, to url: URL) throws {
        // Convert MLX array to Float array
        let samples: [Float]
        if audio.ndim > 1 {
            // Flatten multi-dimensional arrays
            samples = audio.flattened().asArray(Float.self)
        } else {
            samples = audio.asArray(Float.self)
        }
        
        // Validate samples
        guard !samples.isEmpty else {
            throw AudioSaverError.emptySamples
        }
        
        // Create audio format
        guard let format = AVAudioFormat(
            commonFormat: config.bitDepth.avFormat,
            sampleRate: config.sampleRate,
            channels: 1,
            interleaved: false
        ) else {
            throw AudioSaverError.formatCreationFailed
        }
        
        // Configure file settings based on format
        let settings = createSettings(for: format)
        
        // Create audio file
        guard let audioFile = try? AVAudioFile(
            forWriting: url,
            settings: settings
        ) else {
            throw AudioSaverError.fileCreationFailed(url.path)
        }
        
        // Create buffer
        let frameCount = AVAudioFrameCount(samples.count)
        guard let buffer = AVAudioPCMBuffer(
            pcmFormat: format,
            frameCapacity: frameCount
        ) else {
            throw AudioSaverError.bufferCreationFailed
        }
        
        buffer.frameLength = frameCount
        
        // Copy samples to buffer
        switch config.bitDepth {
        case .float32:
            copyFloatSamples(samples, to: buffer)
        case .int16, .int24, .int32:
            copyIntSamples(samples, to: buffer, bitDepth: config.bitDepth)
        }
        
        // Write to file
        try audioFile.write(from: buffer)
    }
    
    /// Save audio to file path
    public func save(_ audio: MLXArray, to path: String) throws {
        let url = URL(fileURLWithPath: path)
        try save(audio, to: url)
    }
    
    /// Save with automatic file extension
    public func save(_ audio: MLXArray, to basePath: String, addExtension: Bool) throws {
        let path = addExtension ? "\(basePath).\(config.fileFormat.fileExtension)" : basePath
        try save(audio, to: path)
    }
    
    // MARK: - AudioData Support
    
    /// Save AudioData to file
    public func saveBuffer(_ buffer: AudioData, to url: URL) throws {
        // Convert Float array to MLXArray and save
        let mlxArray = MLXArray(buffer.samples)
        
        // Create custom config with buffer's sample rate
        let customConfig = Configuration(
            sampleRate: Double(buffer.sampleRate),
            bitDepth: config.bitDepth,
            fileFormat: config.fileFormat
        )
        
        let customSaver = AudioSaver(config: customConfig)
        try customSaver.save(mlxArray, to: url)
    }
    
    /// Save AudioData to file path
    public func saveBuffer(_ buffer: AudioData, to path: String) throws {
        let url = URL(fileURLWithPath: path)
        try saveBuffer(buffer, to: url)
    }
    
    /// Save AudioData with automatic file extension
    public func saveBuffer(_ buffer: AudioData, to basePath: String, addExtension: Bool) throws {
        let path = addExtension ? "\(basePath).\(config.fileFormat.fileExtension)" : basePath
        try saveBuffer(buffer, to: path)
    }
    
    /// Save stereo audio
    public func saveStereo(left: MLXArray, right: MLXArray, to url: URL) throws {
        // Ensure equal length
        let length = min(left.shape[0], right.shape[0])
        let leftSamples = left[0..<length].asArray(Float.self)
        let rightSamples = right[0..<length].asArray(Float.self)
        
        // Create stereo format
        guard let format = AVAudioFormat(
            commonFormat: config.bitDepth.avFormat,
            sampleRate: config.sampleRate,
            channels: 2,
            interleaved: false
        ) else {
            throw AudioSaverError.formatCreationFailed
        }
        
        // Configure settings
        let settings = createSettings(for: format)
        
        // Create file
        guard let audioFile = try? AVAudioFile(
            forWriting: url,
            settings: settings
        ) else {
            throw AudioSaverError.fileCreationFailed(url.path)
        }
        
        // Create buffer
        let frameCount = AVAudioFrameCount(length)
        guard let buffer = AVAudioPCMBuffer(
            pcmFormat: format,
            frameCapacity: frameCount
        ) else {
            throw AudioSaverError.bufferCreationFailed
        }
        
        buffer.frameLength = frameCount
        
        // Copy samples to both channels
        if config.bitDepth == .float32 {
            guard let leftChannel = buffer.floatChannelData?[0],
                  let rightChannel = buffer.floatChannelData?[1] else {
                throw AudioSaverError.channelDataAccessFailed
            }
            
            memcpy(leftChannel, leftSamples, length * MemoryLayout<Float>.size)
            memcpy(rightChannel, rightSamples, length * MemoryLayout<Float>.size)
        } else {
            // Handle integer formats
            guard let leftChannel = buffer.int32ChannelData?[0],
                  let rightChannel = buffer.int32ChannelData?[1] else {
                throw AudioSaverError.channelDataAccessFailed
            }
            
            let scale = getIntScale(for: config.bitDepth)
            for i in 0..<length {
                leftChannel[i] = Int32(leftSamples[i] * scale)
                rightChannel[i] = Int32(rightSamples[i] * scale)
            }
        }
        
        // Write to file
        try audioFile.write(from: buffer)
    }
    
    // MARK: - Private Methods
    
    private func createSettings(for format: AVAudioFormat) -> [String: Any] {
        var settings = format.settings
        
        switch config.fileFormat {
        case .wav, .aiff, .caf:
            // PCM formats use the format settings as-is
            break
            
        case .m4a(let bitRate):
            // AAC encoding settings
            settings[AVFormatIDKey] = kAudioFormatMPEG4AAC
            settings[AVEncoderBitRateKey] = bitRate
            settings[AVEncoderAudioQualityKey] = AVAudioQuality.high.rawValue
        }
        
        return settings
    }
    
    private func copyFloatSamples(_ samples: [Float], to buffer: AVAudioPCMBuffer) {
        guard let channelData = buffer.floatChannelData?[0] else { return }
        memcpy(channelData, samples, samples.count * MemoryLayout<Float>.size)
    }
    
    private func copyIntSamples(_ samples: [Float], to buffer: AVAudioPCMBuffer, bitDepth: Configuration.BitDepth) {
        switch bitDepth {
        case .int16:
            guard let channelData = buffer.int16ChannelData?[0] else { return }
            let scale = Float(Int16.max)
            for i in 0..<samples.count {
                // Clip to [-1, 1] range
                let clipped = max(-1.0, min(1.0, samples[i]))
                channelData[i] = Int16(clipped * scale)
            }
        case .int24, .int32:
            guard let channelData = buffer.int32ChannelData?[0] else { return }
            let scale = getIntScale(for: bitDepth)
            for i in 0..<samples.count {
                // Clip to [-1, 1] range
                let clipped = max(-1.0, min(1.0, samples[i]))
                channelData[i] = Int32(clipped * scale)
            }
        case .float32:
            break // Should not reach here
        }
    }
    
    private func getIntScale(for bitDepth: Configuration.BitDepth) -> Float {
        switch bitDepth {
        case .int16: return Float(Int16.max)
        case .int24: return Float(1 << 23) - 1  // 24-bit max
        case .int32: return Float(Int32.max)
        case .float32: return 1.0
        }
    }
}

// MARK: - Errors

public enum AudioSaverError: LocalizedError {
    case emptySamples
    case formatCreationFailed
    case fileCreationFailed(String)
    case bufferCreationFailed
    case channelDataAccessFailed
    case writeFailed(String)
    
    public var errorDescription: String? {
        switch self {
        case .emptySamples:
            return "Cannot save empty audio"
        case .formatCreationFailed:
            return "Failed to create audio format"
        case .fileCreationFailed(let path):
            return "Failed to create file: \(path)"
        case .bufferCreationFailed:
            return "Failed to create audio buffer"
        case .channelDataAccessFailed:
            return "Failed to access channel data"
        case .writeFailed(let reason):
            return "Write failed: \(reason)"
        }
    }
}

// MARK: - Convenience Extensions

extension AudioSaver {
    
    /// Quick save with default settings
    public static func quickSave(_ audio: MLXArray, to path: String) throws {
        let saver = AudioSaver()
        try saver.save(audio, to: path)
    }
    
    /// Save as high-quality WAV
    public static func saveWAV(_ audio: MLXArray, to path: String, sampleRate: Double = 48000) throws {
        let config = Configuration(
            sampleRate: sampleRate,
            bitDepth: .float32,
            fileFormat: .wav
        )
        let saver = AudioSaver(config: config)
        try saver.save(audio, to: path)
    }
    
    /// Save as compressed M4A
    public static func saveM4A(_ audio: MLXArray, to path: String, bitRate: Int = 256000) throws {
        let config = Configuration(
            sampleRate: 48000,
            bitDepth: .float32,
            fileFormat: .m4a(bitRate: bitRate)
        )
        let saver = AudioSaver(config: config)
        try saver.save(audio, to: path)
    }
}
