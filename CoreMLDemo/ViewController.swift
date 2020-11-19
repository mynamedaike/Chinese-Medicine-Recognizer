//
//  ViewController.swift
//  CoreMLDemo
//
//  Created by Sai Kambampati on 14/6/2017.
//  Copyright © 2017 AppCoda. All rights reserved.
//

import Foundation
import UIKit
import Vision
import AVFoundation
import CoreML

class ViewController: UIViewController, UINavigationControllerDelegate {
    
    @IBOutlet weak var imageView: UIImageView!
    @IBOutlet weak var classifier: UILabel!
    
    let yolo = YOLO()
    
    var boundingBoxes = [BoundingBox]()
    var colors: [UIColor] = []
    
    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view, typically from a nib.
        setUpBoundingBoxes()
        setUpLayer()
    }
    
    func setUpBoundingBoxes() {
        for _ in 0..<YOLO.maxBoundingBoxes {
            boundingBoxes.append(BoundingBox())
        }
        
        var list: [[CGFloat]] = [[1, 0, 0], [0, 1, 0], [1, 1, 0]]
        for elem in list {
            let color = UIColor(red: elem[0], green: elem[1], blue: elem[2], alpha: 1)
            colors.append(color)
        }
    }
    
    func setUpLayer() {
        for box in self.boundingBoxes {
            box.addToLayer(self.imageView.layer)
        }
    }
    
    override func viewWillAppear(_ animated: Bool) {
        
    }
    
    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
        // Dispose of any resources that can be recreated.
    }
    
    @IBAction func camera(_ sender: Any) {
        
        if !UIImagePickerController.isSourceTypeAvailable(.camera) {
            return
        }
        
        let cameraPicker = UIImagePickerController()
        cameraPicker.delegate = self
        cameraPicker.sourceType = .camera
        cameraPicker.allowsEditing = false
        
        present(cameraPicker, animated: true)
    }
    
    @IBAction func openLibrary(_ sender: Any) {
        let picker = UIImagePickerController()
        picker.allowsEditing = false
        picker.delegate = self
        picker.sourceType = .photoLibrary
        present(picker, animated: true)
    }

}

extension ViewController: UIImagePickerControllerDelegate {
    func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
        dismiss(animated: true, completion: nil)
    }
    
    func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [String : Any]) {
        
        picker.dismiss(animated: true)

        classifier.text = "分析图片中..."
        guard let image = info["UIImagePickerControllerOriginalImage"] as? UIImage else {
            return
        } //1
        
        UIGraphicsBeginImageContextWithOptions(CGSize(width: YOLO.inputWidth, height: YOLO.inputHeight), true, 2.0)
        image.draw(in: CGRect(x: 0, y: 0, width: YOLO.inputWidth, height: YOLO.inputHeight))
        let newImage = UIGraphicsGetImageFromCurrentImageContext()!
        UIGraphicsEndImageContext()
        
        imageView.image = newImage
        
        let pixelBuffer = toCVPixelBuffer(from: newImage)
        
        // Core ML
        guard let boundingBoxes = try? yolo.predict(image: pixelBuffer!) else {
            return
        }
        
        if boundingBoxes?.count == 0 {
            classifier.text = "无法识别"
        }else {
            classifier.text = ""
        }
        
        show(predictions: boundingBoxes!)
    }
    
    func toCVPixelBuffer(from image: UIImage) -> CVPixelBuffer? {
        let attrs = [kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue, kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue] as CFDictionary
        var pixelBuffer : CVPixelBuffer?
        let status = CVPixelBufferCreate(kCFAllocatorDefault, Int(image.size.width), Int(image.size.height), kCVPixelFormatType_32ARGB, attrs, &pixelBuffer)
        guard (status == kCVReturnSuccess) else {
            return nil
        }

        CVPixelBufferLockBaseAddress(pixelBuffer!, CVPixelBufferLockFlags(rawValue: 0))
        let pixelData = CVPixelBufferGetBaseAddress(pixelBuffer!)

        let rgbColorSpace = CGColorSpaceCreateDeviceRGB()
        let context = CGContext(data: pixelData, width: Int(image.size.width), height: Int(image.size.height), bitsPerComponent: 8, bytesPerRow: CVPixelBufferGetBytesPerRow(pixelBuffer!), space: rgbColorSpace, bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue)

        context?.translateBy(x: 0, y: image.size.height)
        context?.scaleBy(x: 1.0, y: -1.0)

        UIGraphicsPushContext(context!)
        image.draw(in: CGRect(x: 0, y: 0, width: image.size.width, height: image.size.height))
        UIGraphicsPopContext()
        CVPixelBufferUnlockBaseAddress(pixelBuffer!, CVPixelBufferLockFlags(rawValue: 0))

        return pixelBuffer
    }
    
    func show(predictions: [YOLO.Prediction]) {
        for i in 0..<boundingBoxes.count {
            if i < predictions.count {
                let prediction = predictions[i]
                
                let width = imageView.bounds.width                
                let height = width
                let scaleX = width / CGFloat(YOLO.inputWidth)
                let scaleY = height / CGFloat(YOLO.inputHeight)
                let top = (imageView.bounds.height - height) / 2
                
                var rect = prediction.rect
                rect.origin.x *= scaleX
                rect.origin.y *= scaleY
                rect.origin.y += top
                rect.size.width *= scaleX
                rect.size.height *= scaleY
                
                let label = String(format: "%@ %.3f", labels[prediction.classIndex], prediction.score * 100)
                let color = colors[prediction.classIndex]
                boundingBoxes[i].show(frame: rect, label: label, color: color)
            }else {
                boundingBoxes[i].hide()
            }
        }
    }
}
