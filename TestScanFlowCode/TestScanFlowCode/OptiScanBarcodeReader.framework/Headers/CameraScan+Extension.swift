//
//  CameraScan+Extension.swift
//  OptiScanBarcodeReader
//
//  Created by MAC-OBS-25 on 01/03/22.
//

import UIKit
import Foundation
import AVFoundation
import opencv2
import Vision

extension CameraScan:UINavigationControllerDelegate{
    
   private var hintsZxing:ZXDecodeHints {
        let formats = ZXDecodeHints()
        formats.addPossibleFormat(ZXBarcodeFormat.init(rawValue: 1))
        formats.addPossibleFormat(ZXBarcodeFormat.init(rawValue: 2))
        formats.addPossibleFormat(ZXBarcodeFormat.init(rawValue: 3))
        formats.addPossibleFormat(ZXBarcodeFormat.init(rawValue: 4))
        formats.addPossibleFormat(ZXBarcodeFormat.init(rawValue: 5))
        formats.addPossibleFormat(ZXBarcodeFormat.init(rawValue: 6))
        formats.addPossibleFormat(ZXBarcodeFormat.init(rawValue: 7))
        formats.addPossibleFormat(ZXBarcodeFormat.init(rawValue: 8))
        formats.addPossibleFormat(ZXBarcodeFormat.init(rawValue: 9))
        formats.addPossibleFormat(ZXBarcodeFormat.init(rawValue: 10))
        formats.addPossibleFormat(ZXBarcodeFormat.init(rawValue: 11))
        formats.addPossibleFormat(ZXBarcodeFormat.init(rawValue: 12))
        formats.addPossibleFormat(ZXBarcodeFormat.init(rawValue: 13))
        formats.addPossibleFormat(ZXBarcodeFormat.init(rawValue: 14))
        formats.addPossibleFormat(ZXBarcodeFormat.init(rawValue: 15))
        formats.addPossibleFormat(ZXBarcodeFormat.init(rawValue: 16))
        formats.addPossibleFormat(ZXBarcodeFormat.init(rawValue: 17))
        return formats
    }
    
    /** This method runs the live camera pixelBuffer through tensorFlow to get the result.
     */
    func runModel(onPixelBuffer pixelBuffer: CVPixelBuffer) {
        // Run the live camera pixelBuffer through tensorFlow to get the result
        //        let pixImage = pixelBuffer.toImage()
        //        let fixOrient = self.fixOrientation(img: pixImage)
        
        result = YoloV4Classifier.shared.runModel(onFrame: pixelBuffer,previewSize: self.previewSize)
        //        let src = Mat(uiImage: UIImage(named: "long_distance.jpg")!)
        ////
        //        let dst = Mat()
        //        opencv2.Core.normalize(src: src, dst: dst, alpha: 1.0, beta: 127.5, norm_type: NormTypes.NORM_INF)
        guard let displayResult = result else {
            return
        }
   
        for inference in displayResult.inferences {
            self.processResult(cropImage: inference.outputImage,previewWidth: inference.previewWidth,previewHeight: inference.previewHeight, inference: inference)
        }
        
        //      let width = CVPixelBufferGetWidth(pixelBuffer)
        //      let height = CVPixelBufferGetHeight(pixelBuffer)
        
        DispatchQueue.main.async {
            print("BEFORE DRAW: \(self.getCurrentMillis())")
            
            // Draws the bounding boxes and displays class names and confidence scores.
            self.drawAfterPerformingCalculations(onInferences: displayResult.inferences, withImageSize: CGSize(width: 0.0, height: 0.0))
            print("AFTER DRAW: \(self.getCurrentMillis())")
        }
    }
    
    internal func processResult(cropImage:UIImage,previewWidth:CGFloat,previewHeight:CGFloat,inference:Inference){
//        print("###### processResult")
//        print("AFTER performRotate: \(getCurrentMillis())")
        if inference.className == "QR" {
//            print("BEFORE QR DECODE: \(getCurrentMillis())")
            
//            print("CROPPED IMAGE SIZE",cropImage.size)
            var resultImage = UIImage()
            if isQrLongDistance(image: cropImage,previewWidth: previewWidth,previewHeight: previewHeight) {
//                print("QR Long Distance")
                resultImage = cropImage.upscaleQRcode()
                
                resultImage = SuperResolution.shared.convertImgToSRImg(inputImage: resultImage) ?? UIImage()
//                print("UPSCALE RESIZE",resultImage.size)
            }
            else{
                resultImage = cropImage
            }

            let points = NSMutableArray()
            let mat = Mat.init(uiImage: resultImage)
            let result = WeChatQRCode().detectAndDecode(img: mat, points: points)
            print("WECHAT RESULT",result)
            
            if result.first == nil || result.first == "" {
                self.decodeZxing(image: resultImage)
            }
            else{
//                print("###### FOUND QR",result.first ?? "")
//                AudioServicesPlaySystemSound(SystemSoundID(kSystemSoundID_Vibrate))
                found(code: result.first ?? "")
            }
//            print("AFTER QR DECODE: \(getCurrentMillis())")
//            print("AFTER ROTATE: \(getCurrentMillis())")
        }
        else{
            let resultImage:UIImage?
//            UIImageWriteToSavedPhotosAlbum(cropImage , self, #selector(image(_:didFinishSavingWithError:contextInfo:)), nil)
            if isBarcodeLongDistance(image: cropImage,previewWidth: previewWidth,previewHeight: previewHeight) {
               print("BAR Long Distance")
                resultImage = cropImage.upscaleBarcode()
                print("UPSCALE RESIZE",resultImage?.size)
            }
            else{
                resultImage = cropImage
            }


           // DispatchQueue.main.async {
           //     self.previewImage.image = cropImage
           // }
            let rotatedImage = self.processImage(image: resultImage ?? UIImage())
            self.decodeZxing(image: rotatedImage ?? UIImage())
        }
        startCamera()
    }
    
//    //MARK: - Add image to Library
//       @objc func image(_ image: UIImage, didFinishSavingWithError error: Error?, contextInfo: UnsafeRawPointer) {
//           if let error = error {
//               // we got back an error!
//               print(error.localizedDescription)
//           } else {
//               print("Your image has been saved to your photos.")
//           }
//       }
    
    
   internal func decodeZxing(image:UIImage){
    
        let source: ZXLuminanceSource = ZXCGImageLuminanceSource(cgImage: image.cgImage)
        let binazer = ZXHybridBinarizer(source: source)
        let bitmap = ZXBinaryBitmap(binarizer: binazer)
//        print("###### BITMAP IMAGE WIDTH \(bitmap?.width ?? 0)")

        let reader = ZXMultiFormatReader()
        let hints = hintsZxing
        print("###### DECODE BITMAP",try? reader.decode(bitmap, hints: hints))
        if let result = try? reader.decode(bitmap, hints: hints){
            barDecodeCount = barDecodeCount + 1
//            print("$$$ $$$ $$ $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ $$$$$$$$$$$$$$ BAR DECODED RESULT COUNT",barDecodeCount)
//            AudioServicesPlaySystemSound(SystemSoundID(kSystemSoundID_Vibrate))
            found(code: "Barcode Response : \(result.text ?? "")")
        }else{
            print("###### DECODE BITMAP Nil")
           // UIImageWriteToSavedPhotosAlbum(image, self, #selector(image(_:didFinishSavingWithError:contextInfo:)), nil)

        }

    }
    
    private func fixOrientation(img: UIImage) -> UIImage {
        
        if ( img.imageOrientation == UIImage.Orientation.up ) {
            return img;
        }
        
        // We need to calculate the proper transformation to make the image upright.
        // We do it in 2 steps: Rotate if Left/Right/Down, and then flip if Mirrored.
        var transform: CGAffineTransform = CGAffineTransform.identity
        
        if ( img.imageOrientation == UIImage.Orientation.down || img.imageOrientation == UIImage.Orientation.downMirrored ) {
            transform = transform.translatedBy(x: img.size.width, y: img.size.height)
            transform = transform.rotated(by: CGFloat(Double.pi))
        }
        
        if ( img.imageOrientation == UIImage.Orientation.left || img.imageOrientation == UIImage.Orientation.leftMirrored ) {
            transform = transform.translatedBy(x: img.size.width, y: 0)
            transform = transform.rotated(by: CGFloat(Double.pi / 2.0))
        }
        
        if ( img.imageOrientation == UIImage.Orientation.right || img.imageOrientation == UIImage.Orientation.rightMirrored ) {
            transform = transform.translatedBy(x: 0, y: img.size.height);
            transform = transform.rotated(by: CGFloat(-Double.pi / 2.0));
        }
        
        if ( img.imageOrientation == UIImage.Orientation.upMirrored || img.imageOrientation == UIImage.Orientation.downMirrored ) {
            transform = transform.translatedBy(x: img.size.width, y: 0)
            transform = transform.scaledBy(x: -1, y: 1)
        }
        
        if ( img.imageOrientation == UIImage.Orientation.leftMirrored || img.imageOrientation == UIImage.Orientation.rightMirrored ) {
            transform = transform.translatedBy(x: img.size.height, y: 0);
            transform = transform.scaledBy(x: -1, y: 1);
        }
        
        // Now we draw the underlying CGImage into a new context, applying the transform
        // calculated above.
        let ctx: CGContext = CGContext(data: nil, width: Int(img.size.width), height: Int(img.size.height),
                                       bitsPerComponent: img.cgImage!.bitsPerComponent, bytesPerRow: 0,
                                       space: img.cgImage!.colorSpace!,
                                       bitmapInfo: img.cgImage!.bitmapInfo.rawValue)!;
        
        ctx.concatenate(transform)
        
        if ( img.imageOrientation == UIImage.Orientation.left ||
             img.imageOrientation == UIImage.Orientation.leftMirrored ||
             img.imageOrientation == UIImage.Orientation.right ||
             img.imageOrientation == UIImage.Orientation.rightMirrored ) {
            ctx.draw(img.cgImage!, in: CGRect(x: 0,y: 0,width: img.size.height,height: img.size.width))
        } else {
            ctx.draw(img.cgImage!, in: CGRect(x: 0,y: 0,width: img.size.width,height: img.size.height))
        }
        
        // And now we just create a new UIImage from the drawing context and return it
        return UIImage(cgImage: ctx.makeImage()!)
    }
    
    private func isQrLongDistance(image:UIImage,previewWidth:CGFloat,previewHeight:CGFloat) ->Bool{
        let isLong = LongDistance().isLongDistanceQRImage(cropImageWidth: image.size.width, cropImageHeight: image.size.height, previewWidth: previewWidth, previewHeight: previewHeight)
        return isLong
    }
    
   private func isBarcodeLongDistance(image:UIImage,previewWidth:CGFloat,previewHeight:CGFloat) ->Bool{
        let isLong = LongDistance().isLongDistanceBarcodeImage(cropImageWidth: image.size.width, cropImageHeight: image.size.height, previewWidth: previewWidth, previewHeight: previewHeight)
        return isLong
    }
    
    /**
     This method takes the results, translates the bounding box rects to the current view, draws the bounding boxes, classNames and confidence scores of inferences.
     */
    func drawAfterPerformingCalculations(onInferences inferences: [Inference], withImageSize imageSize:CGSize) {

        self.overlayView.objectOverlays = []
        self.overlayView.setNeedsDisplay()
        
      let displayFont = UIFont.systemFont(ofSize: 14.0, weight: .medium)

      guard !inferences.isEmpty else {
        return
      }

      var objectOverlays: [ObjectOverlay] = []

      for inference in inferences {

        // Translates bounding box rect to current view.
          var convertedRect = inference.rect
//          print("overlayView width",self.overlayView.bounds.size.width)
//          print("overlayView height",self.overlayView.bounds.size.height)
//          print("inference.rect",inference.rect)

        if convertedRect.origin.x < 0 {
            convertedRect.origin.x = self.edgeOffset
        }

        if convertedRect.origin.y < 0 {
          convertedRect.origin.y = self.edgeOffset
        }

          if convertedRect.maxY > self.overlayView.bounds.maxY {
              convertedRect.size.height = self.overlayView.bounds.maxY - convertedRect.origin.y - self.edgeOffset
          }

          if convertedRect.maxX > self.overlayView.bounds.maxX {
              convertedRect.size.width = self.overlayView.bounds.maxX - convertedRect.origin.x - self.edgeOffset
        }

        let confidenceValue = Int(inference.confidence * 100.0)
        let string = "\(inference.className)  (\(confidenceValue)%)"

        let size = string.size(usingFont: displayFont)
          print("Converted Rect",convertedRect)

          let objectOverlay = ObjectOverlay(name: string, borderRect: inference.boundingRect, nameStringSize: size, color: inference.displayColor, font: displayFont)
          print("BOUNDING Rect",inference.boundingRect)

        objectOverlays.append(objectOverlay)
      }

      // Hands off drawing to the OverlayView
      self.draw(objectOverlays: objectOverlays)

    }
    
    /** Calls methods to update overlay view with detected bounding boxes and class names.
     */
    func draw(objectOverlays: [ObjectOverlay]) {
//        print("&&&&&& &&&&&&& &&&&&&& OBJECT OVERLAY",objectOverlays)
        self.overlayView.objectOverlays = objectOverlays
        self.overlayView.setNeedsDisplay()
    }
    
    
}

extension String {

  /**This method gets size of a string with a particular font.
   */
  func size(usingFont font: UIFont) -> CGSize {
    return size(withAttributes: [.font: font])
  }

}

extension AVCaptureDevice {
    var isLocked: Bool {
        do {
            try lockForConfiguration()
            return true
        } catch {
            print(error)
            return false
        }
    }
    func setTorch(enable: Bool) {
       guard hasTorch && isLocked else { return }
        defer { unlockForConfiguration() }
        if enable {
            torchMode = .on
        } else {
            torchMode = .off
        }
    }
}

extension UIView{
    func addSubviews(_ views: UIView...) {
        views.forEach{ addSubview($0) }
    }
    
    func makeConstraints(top: NSLayoutYAxisAnchor?, left: NSLayoutXAxisAnchor?, right: NSLayoutXAxisAnchor?, bottom: NSLayoutYAxisAnchor?, topMargin: CGFloat, leftMargin: CGFloat, rightMargin: CGFloat, bottomMargin: CGFloat, width: CGFloat, height: CGFloat) {
        
        self.translatesAutoresizingMaskIntoConstraints = false
        if let top = top {
            self.topAnchor.constraint(equalTo: top, constant: topMargin).isActive = true
        }
        
        if let left = left {
            self.leftAnchor.constraint(equalTo: left, constant: leftMargin).isActive = true
        }
        
        if let right = right {
            self.rightAnchor.constraint(equalTo: right, constant: -rightMargin).isActive = true
        }
        
        if let bottom = bottom {
            self.bottomAnchor.constraint(equalTo: bottom, constant: -bottomMargin).isActive = true
        }
        
        if width != 0 {
            self.widthAnchor.constraint(equalToConstant: width).isActive = true
        }
        
        if height != 0 {
            self.heightAnchor.constraint(equalToConstant: height).isActive = true
        }
    }
}


extension UIImage {

    func imageResized(to size: CGSize) -> UIImage {
        return UIGraphicsImageRenderer(size: size).image { _ in
            draw(in: CGRect(origin: .zero, size: size))
        }
    }
    
    func rotate(radians: Float) -> UIImage? {
        let size = CGSize(width: self.size.width + 200, height: self.size.height + 200)
        var newSize = CGRect(origin: CGPoint.zero, size: size).applying(CGAffineTransform(rotationAngle: CGFloat(radians * .pi / 180))).size
        newSize.width = floor(newSize.width)
        newSize.height = floor(newSize.height)
        UIGraphicsBeginImageContextWithOptions(newSize, false, self.scale)
        let context = UIGraphicsGetCurrentContext()!
        // Move origin to middle
        context.translateBy(x: newSize.width/2, y: newSize.height/2)
        // Rotate around middle
        context.rotate(by: CGFloat(radians * .pi / 180))
        // Draw the image at its center
        self.draw(in: CGRect(x: -self.size.width/2, y: -self.size.height/2, width: self.size.width, height: self.size.height))
        let newImage = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        return newImage
    }
    
    /// Returns the data representation of the image after scaling to the given `size` and converting
    /// to grayscale.
    ///
    /// - Parameters
    ///   - size: Size to scale the image to (i.e. image size used while training the model).
    /// - Returns: The scaled image as data or `nil` if the image could not be scaled.
    public func scaledData(with size: CGSize) -> Data? {
        guard let cgImage = self.cgImage, cgImage.width > 0, cgImage.height > 0 else { return nil }
        let bitmapInfo = CGBitmapInfo(
            rawValue: CGImageAlphaInfo.none.rawValue)
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        
        let width = Int(size.width)
        guard let context = CGContext(
            data: nil,
            width: width,
            height: Int(size.height),
            bitsPerComponent: cgImage.bitsPerComponent,
            bytesPerRow: width * 3,
            space: CGColorSpaceCreateDeviceGray(),
            bitmapInfo: bitmapInfo.rawValue)
        else {
            return nil
        }
        context.draw(cgImage, in: CGRect(origin: .zero, size: size))
        
        let img = UIImage(cgImage: context.makeImage()!)
        
        guard let scaledBytes = context.makeImage()?.dataProvider?.data as Data? else { return nil }
        //    let scaledFloats = scaledBytes.map { Float32($0) / 255.0 }
        let scaledFloats = scaledBytes.map { (Float32($0) - 127.5) / 1.0 }
        
        let imgFromData = UIImage(data: Data(copyingBufferOf: scaledFloats))
        
        return Data(copyingBufferOf: scaledFloats)
    }

     func from(color: UIColor) -> UIImage {
        let rect = CGRect(x: 0, y: 0, width: 414, height: 896)
        UIGraphicsBeginImageContext(rect.size)
        let context = UIGraphicsGetCurrentContext()
        context!.setFillColor(color.cgColor)
        context!.fill(rect)
        let img = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        return img!
    }
    
}


extension CGImage {
    var brightnessValue: Int {
        get {
            guard let imageData = self.dataProvider?.data else { return 0 }
            guard let ptr = CFDataGetBytePtr(imageData) else { return 0 }
            let length = CFDataGetLength(imageData)
            
            var R = 0
            var G = 0
            var B = 0
            var n = 0
            
            for i in stride(from: 0, to: length, by: 4) {
                
                R += Int(ptr[i])
                G += Int(ptr[i + 1])
                B += Int(ptr[i + 2])
                n += 1
                
            }
            
            let res = (R + B + G) / (n * 3)
            print(res)
            return res
        }
    }
}

extension UIImage {
    var brightness: Int {
        get {
            return self.cgImage?.brightnessValue ?? 0
        }
    }
}

