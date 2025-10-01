//
//  AudioData.swift
//  AudioUtils
//
//  Audio data types for Float array-based audio processing
//

import Foundation

// MARK: - Audio Data

/// Audio data containing samples as Float array with metadata
public struct AudioData {
    public let samples: [Float]
    public let sampleRate: Int
    public let duration: TimeInterval
    
    public init(samples: [Float], sampleRate: Int, duration: TimeInterval) {
        self.samples = samples
        self.sampleRate = sampleRate
        self.duration = duration
    }
    
    /// Create AudioData from calculated duration
    public init(samples: [Float], sampleRate: Int) {
        self.samples = samples
        self.sampleRate = sampleRate
        self.duration = Double(samples.count) / Double(sampleRate)
    }
}

// MARK: - Audio Data Segment

/// A segment of audio data with timing information
public struct AudioDataSegment {
    public let index: Int
    public let samples: [Float]
    public let sampleRate: Int
    public let startTime: TimeInterval
    public let duration: TimeInterval
    
    public init(index: Int, samples: [Float], sampleRate: Int, startTime: TimeInterval, duration: TimeInterval) {
        self.index = index
        self.samples = samples
        self.sampleRate = sampleRate
        self.startTime = startTime
        self.duration = duration
    }
}

// MARK: - Segmentation

public extension AudioData {
    /// Segment audio data into overlapping segments
    func segment(segmentDuration: TimeInterval, overlap: TimeInterval) -> [AudioDataSegment] {
        let segmentSamples = Int(segmentDuration * Double(sampleRate))
        let hopSamples = Int((segmentDuration - overlap) * Double(sampleRate))
        
        var segments: [AudioDataSegment] = []
        var startSample = 0
        var index = 0
        
        while startSample < samples.count {
            let endSample = min(startSample + segmentSamples, samples.count)
            let segmentArray = Array(samples[startSample..<endSample])
            let startTime = Double(startSample) / Double(sampleRate)
            let actualDuration = Double(segmentArray.count) / Double(sampleRate)
            
            segments.append(AudioDataSegment(
                index: index,
                samples: segmentArray,
                sampleRate: sampleRate,
                startTime: startTime,
                duration: actualDuration
            ))
            
            startSample += hopSamples
            index += 1
        }
        
        return segments
    }
}