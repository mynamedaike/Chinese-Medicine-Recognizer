import Foundation
import UIKit
import CoreML

class YOLO {
    public static let inputWidth = 416
    public static let inputHeight = 416
    public static let maxBoundingBoxes = 10
    
    // Tweak these values to get more or fewer predictions.
    let confidenceThreshold: Float = 0.7
    let iouThreshold: Float = 0.5
    
    struct Prediction {
        let classIndex: Int
        let score: Float
        let rect: CGRect
    }
    
    let model = MedicineRecognizer()
    
    public init() { }
    
    public func predict(image: CVPixelBuffer) -> [Prediction]? {
        if let output = try? model.prediction(image: image) {
            var predictions = [Prediction]()
            predictions += computeBoundingBoxes(features: [output.featureMap1, output.featureMap2, output.featureMap3])
            return predictions
        }else {
            return nil
        }
    }
    
    public func computeBoundingBoxes(features: [MLMultiArray]) -> [Prediction] {
        assert(features[0].count == 24*13*13)
        assert(features[1].count == 24*26*26)
        assert(features[2].count == 24*52*52)
        
        let gridHeight = [13, 26, 52]
        let gridWidth = [13, 26, 52]
        let boxesPerCell = 3
        let numClasses = 3
        
        var predictions = [Prediction]()
        
        var blockSize: Float
        
        var channelStride: Int
        var yStride: Int
        var xStride: Int
        
        var featurePointer: UnsafeMutablePointer<Float32>
        
        @inline(__always) func offset(_ channel: Int, _ x: Int, _ y: Int) -> Int {
            return channel * channelStride + y * yStride + x * xStride
        }
        
        for i in 0..<3 {
            for cy in 0..<gridHeight[i] {
                for cx in 0..<gridWidth[i] {
                    for b in 0..<boxesPerCell {
                        blockSize = (Float) (YOLO.inputWidth / gridHeight[i])
                        
                        channelStride = features[i].strides[0].intValue
                        yStride = features[i].strides[1].intValue
                        xStride = features[i].strides[2].intValue
                        
                        featurePointer = UnsafeMutablePointer<Float32>(OpaquePointer(features[i].dataPointer))
                        
                        // For the first bounding box (b = 0) we have to read channels 0 - 7,
                        // for b = 1 we have to read channels 8 - 15, and so on.
                        let channel = b * (numClasses + 5)
                        
                        let tx = Float(featurePointer[offset(channel, cx, cy)])
                        let ty = Float(featurePointer[offset(channel + 1, cx, cy)])
                        let tw = Float(featurePointer[offset(channel + 2, cx, cy)])
                        let th = Float(featurePointer[offset(channel + 3, cx, cy)])
                        let tc = Float(featurePointer[offset(channel + 4, cx, cy)])
                        
                        let x = (Float(cx) + sigmoid(tx)) * blockSize
                        let y = (Float(cy) + sigmoid(ty)) * blockSize
                        
                        let w = exp(tw) * anchors[i][2 * b    ]
                        let h = exp(th) * anchors[i][2 * b + 1]
                        
                        let confidence = sigmoid(tc)
                        
                        var classes = [Float](repeating: 0, count: numClasses)
                        for c in 0..<numClasses {
                            classes[c] = sigmoid(Float(featurePointer[offset(channel + 5 + c, cx, cy)]))
                        }
                        
                        let (detectedClass, bestClassScore) = classes.argmax()
                        
                        let confidenceInClass = bestClassScore * confidence
                        
                        if confidenceInClass > confidenceThreshold {
                            print(confidenceInClass)
                            let rect = CGRect(x: CGFloat(x - w / 2), y: CGFloat(y - h / 2),
                                              width: CGFloat(w), height: CGFloat(h))
                            let prediction = Prediction(classIndex: detectedClass,
                                                        score: confidenceInClass,
                                                        rect: rect)
                            predictions.append(prediction)
                        }
                    }
                }
            }
        }
        return nonMaxSuppression(boxes: predictions, limit: YOLO.maxBoundingBoxes, threshold: iouThreshold)
    }
}
