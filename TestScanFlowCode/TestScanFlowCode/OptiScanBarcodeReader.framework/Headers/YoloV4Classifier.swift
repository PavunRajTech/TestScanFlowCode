//
//  YoloV4Classifier.swift
//  OptiScanBarcodeReader
//
//  Created by MAC-OBS-25 on 28/02/22.
//

import UIKit
import CoreImage
import Accelerate

/// Stores results for a particular frame that was successfully run through the `Interpreter`.
struct Result {
  let inferences: [Inference]
}

/// Stores one formatted inference.
struct Inference {
  let confidence: Float
  let className: String
  let rect: CGRect
  let boundingRect:CGRect
  let displayColor: UIColor
  let outputImage : UIImage
  let previewWidth : CGFloat
  let previewHeight : CGFloat
}

/// Information about a model file or labels file.
typealias FileInfo = (name: String, extension: String)


fileprivate let tensorModelName:String = "yolov4-tiny_final_28_03_22"
fileprivate let tensorModelExtension:String = "tflite"
fileprivate let tensorLabelDataName:String = "labelmap"
fileprivate let tensorLabelDataExt:String = "txt"


/// Information about theYoloV4 model.
enum YoloV4 {
  static let modelInfo: FileInfo = (name: tensorModelName, extension: tensorModelExtension)
  static let labelsInfo: FileInfo = (name: tensorLabelDataName, extension: tensorLabelDataExt)
}


/// This class handles all data preprocessing and makes calls to run inference on a given frame
/// by invoking the `Interpreter`. It then formats the inferences obtained and returns the top N
/// results for a successful inference.
class YoloV4Classifier: NSObject {
    
    
    // MARK: - Internal Properties
    /// The current thread count used by the TensorFlow Lite Interpreter.
    var threadCount: Int = 0
    let threadCountLimit = 10
    var resizedBufferImage:UIImage?

    let threshold: Double = 0.5

    // MARK: Model parameters
    let batchSize = 1
    let inputChannels = 3
    let inputWidth = 416.0
    let inputHeight = 416.0

    // image mean and std for floating model, should be consistent with parameters used in model training
    let imageMean: Float = 127.5
    let imageStd:  Float = 127.5
   
    var qrcount:Int = 0
    var barcount:Int = 0

    // MARK: Private properties
    private var labels: [String] = []

    /// TensorFlow Lite `Interpreter` object for performing inference on a given model.
    private var interpreter: Interpreter?

    private let bgraPixel = (channels: 4, alphaComponent: 3, lastBgrComponent: 2)
    private let rgbPixelChannels = 3
    private let colorStrideValue = 10
    private let colors = [
      UIColor.red,
      UIColor(displayP3Red: 90.0/255.0, green: 200.0/255.0, blue: 250.0/255.0, alpha: 1.0),
      UIColor.green,
      UIColor.orange,
      UIColor.blue,
      UIColor.purple,
      UIColor.magenta,
      UIColor.yellow,
      UIColor.cyan,
      UIColor.brown
    ]
    private var scannerType:ScannerType?
    static let shared = YoloV4Classifier()


    // MARK: - Initialization

    /// A failable initializer for `YoloV4Classifier`. A new instance is created if the model and
    /// labels files are successfully loaded from the app's main bundle. Default `threadCount` is 1.
   
    func initializeModelInfo(selectedScannerType:ScannerType){
        print("INIT YOLO CLASSIFIER")
        scannerType = selectedScannerType
        print("Selected type",scannerType)
        let modelFilename = YoloV4.modelInfo.name
        // Construct the path to the model file.
        let bundle = Bundle(for: type(of: self))
        guard let modelPath = bundle.path(forResource: modelFilename, ofType:  YoloV4.modelInfo.extension) else {
            print("Failed to load the model file with name: \(modelFilename).")
            return
        }
        
        // Specify the options for the `Interpreter`.
        self.threadCount = 1
        var options = Interpreter.Options()
        options.threadCount = threadCount
        do {
            // Create the `Interpreter`.
            interpreter = try Interpreter(modelPath: modelPath, options: options)
            // Allocate memory for the model's input `Tensor`s.
            try interpreter?.allocateTensors()
        } catch let error {
            print("Failed to create the interpreter with error: \(error.localizedDescription)")
            return
        }

//        super.init()

        // Load the classes listed in the labels file.
        loadLabels(fileInfo: YoloV4.labelsInfo)

    }
    
    private func setupUI() {
//        let window = UIApplication.shared.keyWindow!
//        window.addSubviews(previewImage)
//        DispatchQueue.main.async {
//            print("SETUP UI")
//    //        window.addSubview(takePhotoButton)
//            if let keyWindow = UIWindow.key {
//                let image = UIImageView()
//                let mainScreen = UIScreen.main.bounds
//                image.frame =  CGRect(x: 0, y: mainScreen.height - 500, width:mainScreen.width , height: 300)
//                image.contentMode = .scaleAspectFit
//                image.backgroundColor = UIColor.gray
//                image.image = buffer
//                keyWindow.addSubview(image)
//            }
//
//
//        }
    }
//    init?(threadCount: Int = 1,selectedScannerType:ScannerType) {
//        print("INIT YOLO CLASSIFIER")
//        scannerType = selectedScannerType
//        let modelFilename = YoloV4.modelInfo.name
//        // Construct the path to the model file.
//        let bundle = Bundle(for: type(of: self))
//        guard let modelPath = bundle.path(forResource: modelFilename, ofType:  YoloV4.modelInfo.extension) else {
//            print("Failed to load the model file with name: \(modelFilename).")
//            return nil
//        }
//
//        // Specify the options for the `Interpreter`.
//        self.threadCount = threadCount
//        var options = Interpreter.Options()
//        options.threadCount = threadCount
//        do {
//            // Create the `Interpreter`.
//            interpreter = try Interpreter(modelPath: modelPath, options: options)
//            // Allocate memory for the model's input `Tensor`s.
//            try interpreter?.allocateTensors()
//        } catch let error {
//            print("Failed to create the interpreter with error: \(error.localizedDescription)")
//            return nil
//        }
//
//        super.init()
//
//        // Load the classes listed in the labels file.
//        loadLabels(fileInfo: YoloV4.labelsInfo)
//    }
//
    @objc func image(_ image: UIImage, didFinishSavingWithError error: Error?, contextInfo: UnsafeRawPointer) {
        if let error = error {
            // we got back an error!
            print(error.localizedDescription)
        } else {
            print("Your image has been saved to your photos.")
        }
    }
    
//    func sampleTest{
//        // Getting model path
//        guard
//          let modelPath = Bundle.main.path(forResource: "yolov4-tiny_final_28_03_22", ofType: "tflite")
//        else {
//          // Error handling...
//        }
//
//        do {
//          // Initialize an interpreter with the model.
//          let interpreter = try Interpreter(modelPath: modelPath)
//
//          // Allocate memory for the model's input `Tensor`s.
//          try interpreter.allocateTensors()
//
//          let inputData: Data  // Should be initialized
//
//          // input data preparation...
//
//          // Copy the input data to the input `Tensor`.
//          try self.interpreter.copy(inputData, toInputAt: 0)
//
//          // Run inference by invoking the `Interpreter`.
//          try self.interpreter.invoke()
//
//          // Get the output `Tensor`
//          let outputTensor = try self.interpreter.output(at: 0)
//
//          // Copy output to `Data` to process the inference results.
//          let outputSize = outputTensor.shape.dimensions.reduce(1, {x, y in x * y})
//          let outputData =
//                UnsafeMutableBufferPointer<Float32>.allocate(capacity: outputSize)
//          outputTensor.data.copyBytes(to: outputData)
//
//          if (error != nil) { /* Error handling... */ }
//        } catch error {
//          // Error handling...
//        }
//    }
    
    /// This class handles all data preprocessing and makes calls to run inference on a given frame
    /// through the `Interpreter`. It then formats the inferences obtained and returns the top N
    /// results for a successful inference.
    func runModel(onFrame pixelBuffer: CVPixelBuffer, previewSize: CGSize) -> Result? {
//        print("YOLO CLASSIFIER runModel")
        
        print("***** Start Runmodel: \(getCurrentMillis())")

      let imageWidth = CVPixelBufferGetWidth(pixelBuffer)
      let imageHeight = CVPixelBufferGetHeight(pixelBuffer)

      let imageChannels = 4
      assert(imageChannels >= inputChannels)
                
    resizedBufferImage = pixelBuffer.toImage()
//        print("ORIGINAL SIZE", self.resizedBufferImage?.size ?? CGSize.zero)
      // Crops the image to the biggest square in the center and scales it down to model dimensions.
//        print("BEFORE IMAGE RESIZE: \(getCurrentMillis())")
        print("***** BEFORE RESIZE: \(getCurrentMillis())")
      let scaledSize = CGSize(width: inputWidth, height: inputHeight)
      guard let scaledPixelBuffer = pixelBuffer.resized(to: scaledSize) else {
        return nil
      }
//       resizedBufferImage = scaledPixelBuffer.toImage()

//       UIImageWriteToSavedPhotosAlbum(resizedBufferImage ?? UIImage() , self, #selector(image(_:didFinishSavingWithError:contextInfo:)), nil)
        
        
        print("***** AFTER RESIZE: \(getCurrentMillis())")

//        print("416 Image size",CVPixelBufferGetWidth(scaledPixelBuffer),CVPixelBufferGetHeight(scaledPixelBuffer))
//      print("AFTER IMAGE RESIZE: \(getCurrentMillis())")
        
      let outputBoundingBox: Tensor
      let outputClasses: Tensor
      do {
          print("***** MODEL STARTED: \(getCurrentMillis())")

        let inputTensor = try interpreter?.input(at: 0)
//          print("INPUT TENSOR:",inputTensor)

          print("***** BEFORE RGB RESIZE: \(getCurrentMillis())")
        // Remove the alpha component from the image buffer to get the RGB data.
          
        guard let rgbData = rgbDataFromBuffer(
          scaledPixelBuffer,
          byteCount: batchSize * Int(inputWidth) * Int(inputHeight) * inputChannels,
          isModelQuantized: inputTensor?.dataType == .uInt8
        ) else {
          print("Failed to convert the image buffer to RGB data.")
          return nil
        }
          
          
          print("***** AFTER RGB SIZE: \(getCurrentMillis())")
        // Copy the RGB data to the input `Tensor`.
        try interpreter?.copy(rgbData, toInputAt: 0)
          
          

        // Run inference by invoking the `Interpreter`.
        try interpreter?.invoke()

          outputBoundingBox = try interpreter?.output(at: 0) as! Tensor
          outputClasses = try interpreter?.output(at: 1) as! Tensor
          print("***** MODEL COMPLETED: \(getCurrentMillis())")


      } catch let error {
        print("Failed to invoke the interpreter with error: \(error.localizedDescription)")
        return nil
      }
       
        let outputcount: Int = outputBoundingBox.shape.dimensions[1]
       
        let boundingBox = [BoundingBox](unsafeData: outputBoundingBox.data)!
        
        let OutScore = [OutScore](unsafeData: outputClasses.data)!

      // Formats the results
        print("***** BEFORE RESULT ARRAY: \(getCurrentMillis())")
      let resultArray = formatResults(
        boundingBox: boundingBox,
        outputClasses: OutScore,
        outputCount: outputcount,
        width: CGFloat(imageWidth),
        height: CGFloat(imageHeight), previewSize: previewSize
      )
        print("***** AFTER RESULT ARRAY: \(getCurrentMillis())")

      // Returns the inference time and inferences
        let result = Result(inferences: resultArray)

        return result
    }
    
    
    /// Filters out all the results with confidence score < threshold and returns the top N results
    /// sorted in descending order.
    func formatResults(boundingBox: [BoundingBox], outputClasses: [OutScore], outputCount: Int, width: CGFloat, height: CGFloat,previewSize:CGSize) -> [Inference]{
        print("PREVIEW SIZE",previewSize)
        let modelConfidence:Float = 0.5
        
        var resultsArray: [Inference] = []
        if (outputCount == 0) {
            return resultsArray
        }
        //        print("BEFORE MAP: \(getCurrentMillis())")
        print("***** BEFORE OUTPUT ARRAY: \(getCurrentMillis())")

        print("BEFORE Sort ARRAY: \(outputClasses)")
        
        
        //let sortedAry = outputClasses.sorted(by: { $0.indexTwo < $1.indexTwo })
        

       // print("After Sort ARRAY: \(sortedAry    )")
        
        let maxOne = outputClasses.map { $0.indexOne }.max()
        let maxTwo = outputClasses.map { $0.indexTwo }.max()
        
        print("***** Max ARRAY One: \(String(describing: maxOne))")
        print("***** Max ARRAY Two : \(String(describing: maxTwo))")
        
        let indexOne = outputClasses.firstIndex(where: {$0.indexOne == maxOne})
        let indexTwo = outputClasses.firstIndex(where: {$0.indexTwo == maxTwo})

        
        print("Index One: \(String(describing: indexOne))")
        print("Index Two : \(String(describing: indexTwo))")
        print("***** After OUTPUT ARRAY: \(getCurrentMillis())")
        
        switch scannerType {
        case .qrcode:
            if maxOne! > modelConfidence{
                
                print("***** BEFORE Bounding calculation: \(getCurrentMillis())")
                
                let boundBoxAry = boundingBox[indexOne!]
                print("Bounding box: \(String(describing: boundBoxAry))")
                let boundingBoxRect = self.calculateBoundBoxRect(boundingBox: boundBoxAry, previewHeight: previewSize.height, previewWidth: previewSize.width)
                
                print("***** After bounding calculation: \(getCurrentMillis())")
                let cropRect = self.calculateCropingRect(boundingBox: boundBoxAry)
                
                print("***** After cropping rect: \(getCurrentMillis())")
                let croppedBar = self.resizedBufferImage?.cropImage(frame: cropRect) ?? UIImage()
                
                print("***** After resized image: \(getCurrentMillis())")

//                UIImageWriteToSavedPhotosAlbum(croppedBar , self, #selector(image(_:didFinishSavingWithError:contextInfo:)), nil)

                //Check with yuvaraj
                let inference = Inference(confidence: Float(threshold),
                                          className: "QR",
                                          rect: cropRect, boundingRect: boundingBoxRect,
                                          displayColor: UIColor.red, outputImage: croppedBar,previewWidth: width,previewHeight: height)
                resultsArray.append(inference)
            }else{
                
                print("***** Else Confidence level : \(getCurrentMillis())")
                
                UIImageWriteToSavedPhotosAlbum(resizedBufferImage! , self, #selector(image(_:didFinishSavingWithError:contextInfo:)), nil)
            }
            
        case .barcode:
            if maxTwo! > modelConfidence {
                
                let boundBoxAry = boundingBox[indexTwo!]
                print("Bounding box: \(String(describing: boundBoxAry))")
                let boundingBoxRect = self.calculateBoundBoxRect(boundingBox: boundBoxAry, previewHeight: previewSize.height, previewWidth: previewSize.width)
                
                let cropRect = self.calculateCropingRect(boundingBox: boundBoxAry)
                
                let croppedBar = self.resizedBufferImage?.cropImage(frame: cropRect) ?? UIImage()

//                UIImageWriteToSavedPhotosAlbum(croppedBar , self, #selector(image(_:didFinishSavingWithError:contextInfo:)), nil)

                //Check with yuvaraj
                let inference = Inference(confidence: Float(threshold),
                                          className: "BAR",
                                          rect: cropRect, boundingRect: boundingBoxRect,
                                          displayColor: UIColor.green, outputImage: croppedBar,previewWidth: width,previewHeight: height)
                resultsArray.append(inference)
            }
            
        case .any:
            
            let one = maxOne!
            
            let two = maxTwo!
           
            if(one >= modelConfidence || two >= modelConfidence){

            if(one > two){

                let boundBoxAry = boundingBox[indexOne!]
                print("Bounding box: \(String(describing: boundBoxAry))")
                let boundingBoxRect = self.calculateBoundBoxRect(boundingBox: boundBoxAry, previewHeight: previewSize.height, previewWidth: previewSize.width)
                
                let cropRect = self.calculateCropingRect(boundingBox: boundBoxAry)
                
                let croppedBar = self.resizedBufferImage?.cropImage(frame: cropRect) ?? UIImage()

//                UIImageWriteToSavedPhotosAlbum(croppedBar , self, #selector(image(_:didFinishSavingWithError:contextInfo:)), nil)

                //Check with yuvaraj
                let inference = Inference(confidence: Float(threshold),
                                          className: "QR",
                                          rect: cropRect, boundingRect: boundingBoxRect,
                                          displayColor: UIColor.red, outputImage: croppedBar,previewWidth: width,previewHeight: height)
                resultsArray.append(inference)

            }else{

                let boundBoxAry = boundingBox[indexTwo!]
                print("Bounding box: \(String(describing: boundBoxAry))")
                let boundingBoxRect = self.calculateBoundBoxRect(boundingBox: boundBoxAry, previewHeight: previewSize.height, previewWidth: previewSize.width)
                
                let cropRect = self.calculateCropingRect(boundingBox: boundBoxAry)
                
                let croppedBar = self.resizedBufferImage?.cropImage(frame: cropRect) ?? UIImage()

//                UIImageWriteToSavedPhotosAlbum(croppedBar , self, #selector(image(_:didFinishSavingWithError:contextInfo:)), nil)

                //Check with yuvaraj
                let inference = Inference(confidence: Float(threshold),
                                          className: "BAR",
                                          rect: cropRect, boundingRect: boundingBoxRect,
                                          displayColor: UIColor.green, outputImage: croppedBar,previewWidth: width,previewHeight: height)
                resultsArray.append(inference)

            }

            }
            
        default:
            break
        }
        
    
        return resultsArray
    }
    
    private func calculateOriginalCropRect(index:Int,height:CGFloat,width:CGFloat,boundingBox:[Float]) -> CGRect{
        print("BUFFER SIZE",width,height)
        print("IPHONE SIZE",UIScreen.main.bounds.size.width,UIScreen.main.bounds.size.height)
        print("DIVIDED INDEX",index)

        var rect: CGRect = CGRect.zero
        rect.origin.y = CGFloat(boundingBox[4*index+1])
        rect.origin.x = CGFloat(boundingBox[4*index])
        rect.size.width = CGFloat(boundingBox[4*index+2])
        rect.size.height = CGFloat(boundingBox[4*index+3])
        print("416 MODEL OUTPUT RECT",rect)

       // let xpos = max(0, rect.origin.x - (rect.size.width / 2))
      //  let ypos = max(0, rect.origin.y - (rect.size.height / 2))
//        let fwidth = min(415 , rect.origin.x + (rect.size.width / 2))
//        let fheight = min(415 , rect.origin.y + (rect.size.height / 2))
        
          let ratioHeight = height / CGFloat(self.inputWidth)
         let ratioWidth = width / CGFloat(self.inputWidth)
//        print("ORIGINAL SIZE --<<<<",height,width)
//
       let x1 = CGFloat(rect.origin.x - rect.size.width / 2)
       let y1 =  CGFloat(rect.origin.y - rect.size.height / 2)
       let x2 = CGFloat(rect.origin.x + rect.size.width / 2)
       let y2 = CGFloat(rect.origin.y + rect.size.height / 2)
//
        
//        let fin = CGRect(x: x1, y: y1, width: x2, height: y2).standardized

    
        
        let rec = CGRect(
            x: CGFloat(min(x1, x2)),
            y: CGFloat(min(y1, y2)),
            width: CGFloat(abs(x1 - x2)),
            height: CGFloat(abs(y1 - y2)))
   
        
        let finalRect = CGRect(x: (rec.origin.x * ratioWidth) , y: (rec.origin.y * ratioHeight) , width: (rec.size.width * ratioWidth)  , height: (rec.size.height * ratioHeight) )

        
        print("1920x1080 RECT FRAME",finalRect)

        return finalRect
       
    }
    
    private func calculateCropingRect(boundingBox: BoundingBox) -> CGRect{

        
        var rect: CGRect = CGRect.zero
        rect.origin.x = CGFloat(boundingBox.indexOne)
        rect.origin.y = CGFloat(boundingBox.indexTwo)
        rect.size.width = CGFloat(boundingBox.indexThree)
        rect.size.height = CGFloat(boundingBox.indexFour)
        
        let x = rect.origin.x/inputWidth
        let y = rect.origin.y/inputHeight
        let w = rect.size.width/inputWidth
        let h = rect.size.height/inputHeight
        
        let img = resizedBufferImage
        
        let image_h = img?.size.height ?? 0.0
        let image_w = img?.size.width ?? 0.0
        
        let orig_x       = x * image_w
        let orig_y       = y * image_h
        let orig_width   = w * image_w
        let orig_height  = h * image_h

        let x1 = orig_x + orig_width / 2
        let y1 = orig_y + orig_height / 2
        let x2 = orig_x - orig_width / 2
        let y2 = orig_y - orig_height / 2
        
        print("Rect X1:\(x1)  Y1: \(y1)")
        print("Rect X2:\(x2)  Y2: \(y2)")
        
        //        let finalRect = CGRect(x: (rec.origin.x * ratioWidth) - 25 , y: (rec.origin.y * ratioHeight) - 25, width: (rec.size.width * ratioWidth) + 50  , height: (rec.size.height * ratioHeight) + 50)

        
        var xMinValue = CGFloat(min(x1, x2))
        var yMinValue = CGFloat(min(y1, y2))
        
        if xMinValue > 25{
            xMinValue = xMinValue - 25
        }
        
        if yMinValue > 25{
            yMinValue = yMinValue - 25
        }
        
        let finalRect = CGRect(
            x: xMinValue,
            y: yMinValue,
            width: CGFloat(abs(x1 - x2)) + 50,
            height: CGFloat(abs(y1 - y2)) + 50)
         
        return finalRect
        
    }
    
    private func calculateBoundBoxRect(boundingBox: BoundingBox, previewHeight:CGFloat, previewWidth:CGFloat) -> CGRect{
        
        
        var rect: CGRect = CGRect.zero
        rect.origin.x = CGFloat(boundingBox.indexOne)
        rect.origin.y = CGFloat(boundingBox.indexTwo)
        rect.size.width = CGFloat(boundingBox.indexThree)
        rect.size.height = CGFloat(boundingBox.indexFour)
        
        let x = rect.origin.x/inputWidth
        let y = rect.origin.y/inputHeight
        let w = rect.size.width/inputWidth
        let h = rect.size.height/inputHeight
        
//        let img = resizedBufferImage
        
        let image_h = previewHeight ?? 0.0
        let image_w = previewWidth ?? 0.0
        
        let orig_x       = x * image_w
        let orig_y       = y * image_h
        let orig_width   = w * image_w
        let orig_height  = h * image_h

        let x1 = orig_x + orig_width / 2
        let y1 = orig_y + orig_height / 2
        let x2 = orig_x - orig_width / 2
        let y2 = orig_y - orig_height / 2
        
        print("Rect X1:\(x1)  Y1: \(y1)")
        print("Rect X2:\(x2)  Y2: \(y2)")
         
         let finalRec = CGRect(
             x: CGFloat(min(x1, x2)),
             y: CGFloat(min(y1, y2)),
             width: CGFloat(abs(x1 - x2)),
             height: CGFloat(abs(y1 - y2)))
        
        return finalRec
        
    }
    
    private func calculateBoundingBoxRect(index:Int,previewHeight:CGFloat,previewWidth:CGFloat,boundingBox:[Float]) -> CGRect{
        var rect: CGRect = CGRect.zero
        rect.origin.y = CGFloat(boundingBox[4*index+1])
        rect.origin.x = CGFloat(boundingBox[4*index])
        rect.size.width = CGFloat(boundingBox[4*index+2])
        rect.size.height = CGFloat(boundingBox[4*index+3])
        
        let rec = CGRect(x: CGFloat(rect.origin.x - rect.size.width/2), y: CGFloat(rect.origin.y - rect.size.height/2),
                              width: CGFloat(rect.size.width), height: CGFloat(rect.size.height))

      
//        let x1 = CGFloat(rect.origin.x - rect.size.width / 2)
//        let y1 =  CGFloat(rect.origin.y - rect.size.height / 2)
//        let x2 = CGFloat(rect.origin.x + rect.size.width / 2) - x1
//        let y2 = CGFloat(rect.origin.y + rect.size.height / 2) - y1
        
        let ratioHeight = previewHeight / CGFloat(self.inputWidth)
        let ratioWidth = previewWidth / CGFloat(self.inputWidth)
        //          let ynew = (896.0 * y1) / 416.0
        let finalRect = CGRect(x: rec.origin.x * ratioWidth , y:rec.origin.y * ratioHeight, width: rec.size.width * ratioWidth , height: rec.size.height * ratioHeight )
        //        print("### Rect",finalRect)
        
//        let screen = UIScreen.main.bounds.size
//        let width = screen.width
//        let height = width * 4 / 3
//        let scaleX = width / CGFloat(inputWidth)
//        let scaleY = height / CGFloat(inputHeight)
//        let top = (screen.height - height) / 2
//
//        // Translate and scale the rectangle to our own coordinate system.
////        var rect = prediction.rect
//        rec.origin.x *= scaleX
//        rec.origin.y *= scaleY
//        rec.origin.y += top
//        rec.size.width *= scaleX
//        rec.size.height *= scaleY
        
        
        return finalRect
       
    }
    
    /// Loads the labels from the labels file and stores them in the `labels` property.
    private func loadLabels(fileInfo: FileInfo) {
      let filename = fileInfo.name
      let fileExtension = fileInfo.extension
      let bundle = Bundle(for: type(of: self))

      guard let fileURL = bundle.url(forResource: filename, withExtension: fileExtension) else {
        fatalError("Labels file not found in bundle. Please add a labels file with name " +
                       "\(filename).\(fileExtension) and try again.")
      }
      do {
        let contents = try String(contentsOf: fileURL, encoding: .utf8)
        labels = contents.components(separatedBy: .newlines)
      } catch {
        fatalError("Labels file named \(filename).\(fileExtension) cannot be read. Please add a " +
                     "valid labels file and try again.")
      }
    }
    
    /// Returns the RGB data representation of the given image buffer with the specified `byteCount`.
    ///
    /// - Parameters
    ///   - buffer: The BGRA pixel buffer to convert to RGB data.
    ///   - byteCount: The expected byte count for the RGB data calculated using the values that the
    ///       model was trained on: `batchSize * imageWidth * imageHeight * componentsCount`.
    ///   - isModelQuantized: Whether the model is quantized (i.e. fixed point values rather than
    ///       floating point values).
    /// - Returns: The RGB data representation of the image buffer or `nil` if the buffer could not be
    ///     converted.
    private let alphaComponent = (baseOffset: 4, moduloRemainder: 3)
    private func rgbDataFromBuffer(
      _ buffer: CVPixelBuffer,
      byteCount: Int,
      isModelQuantized: Bool
    ) -> Data? {
      CVPixelBufferLockBaseAddress(buffer, .readOnly)
      defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
      guard let mutableRawPointer = CVPixelBufferGetBaseAddress(buffer) else {
        return nil
      }
      let count = CVPixelBufferGetDataSize(buffer)
      let bufferData = Data(bytesNoCopy: mutableRawPointer, count: count, deallocator: .none)
      var rgbBytes = [UInt8](repeating: 0, count: byteCount)
      var index = 0
      for component in bufferData.enumerated() {
        let offset = component.offset
        let isAlphaComponent = (offset % alphaComponent.baseOffset) == alphaComponent.moduloRemainder
        guard !isAlphaComponent else { continue }
        rgbBytes[index] = component.element
        index += 1
      }
      if isModelQuantized { return Data(rgbBytes) }
      return Data(copyingBufferOf: rgbBytes.map { Float($0) / 255.0 })
    }
    /* func rgbDataFromBuffer(
      _ buffer: CVPixelBuffer,
      byteCount: Int,
      isModelQuantized: Bool
    ) -> Data? {
      CVPixelBufferLockBaseAddress(buffer, .readOnly)
      defer {
        CVPixelBufferUnlockBaseAddress(buffer, .readOnly)
      }
      guard let sourceData = CVPixelBufferGetBaseAddress(buffer) else {
        return nil
      }
      
      let width = CVPixelBufferGetWidth(buffer)
      let height = CVPixelBufferGetHeight(buffer)
      let sourceBytesPerRow = CVPixelBufferGetBytesPerRow(buffer)
      let destinationChannelCount = 3
      let destinationBytesPerRow = destinationChannelCount * width
      
      var sourceBuffer = vImage_Buffer(data: sourceData,
                                       height: vImagePixelCount(height),
                                       width: vImagePixelCount(width),
                                       rowBytes: sourceBytesPerRow)
      
      guard let destinationData = malloc(height * destinationBytesPerRow) else {
        print("Error: out of memory")
        return nil
      }
      
      defer {
        free(destinationData)
      }

      var destinationBuffer = vImage_Buffer(data: destinationData,
                                            height: vImagePixelCount(height),
                                            width: vImagePixelCount(width),
                                            rowBytes: destinationBytesPerRow)
      
      if (CVPixelBufferGetPixelFormatType(buffer) == kCVPixelFormatType_32BGRA){
        vImageConvert_BGRA8888toRGB888(&sourceBuffer, &destinationBuffer, UInt32(kvImageNoFlags))
      } else if (CVPixelBufferGetPixelFormatType(buffer) == kCVPixelFormatType_32ARGB) {
        vImageConvert_ARGB8888toRGB888(&sourceBuffer, &destinationBuffer, UInt32(kvImageNoFlags))
      }

      let byteData = Data(bytes: destinationBuffer.data, count: destinationBuffer.rowBytes * height)
      if isModelQuantized {
        return byteData
      }

        // Not quantized, convert to floats
        let bytes = Array<UInt8>(unsafeData: byteData)!
        var floats = [Float]()
        for i in 0..<bytes.count {
            floats.append(Float(bytes[i]) / 255.0)
        }
        return Data(copyingBufferOf: floats)
    }*/
    
    func getCurrentMillis()->String {
       let dateFormatter : DateFormatter = DateFormatter()
       dateFormatter.dateFormat = "yyyy-MMM-dd HH:mm:ss.SSSS"
       let date = Date()
       let dateString = dateFormatter.string(from: date)
       return dateString
   }
    
}


extension UIWindow {
    static var key: UIWindow? {
        if #available(iOS 13, *) {
            return UIApplication.shared.windows.first { $0.isKeyWindow }
        } else {
            return UIApplication.shared.keyWindow
        }
    }
}


struct BoundingBox {
    var indexOne: Float
    var indexTwo: Float
    var indexThree: Float
    var indexFour: Float
}

struct OutScore {
    var indexOne: Float
    var indexTwo: Float
}

