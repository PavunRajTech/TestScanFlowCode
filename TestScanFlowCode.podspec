


Pod::Spec.new do |spec|

  spec.name         = "TestScanFlowCode"
  spec.version      = "1.0.0"
  spec.summary      = "A short description of OptiScanBarcodeReader with barcodeScanner."
  spec.description  = "This is create the sample app OptiScanBarcodeReader"
  spec.homepage     = "https://github.com/PavunRajTech/TestScanFlowCode"
  spec.license      = "MIT"
  spec.author             = { "PavunRajTech" => "pavunraj.p@optisolbusiness.com" }
  # spec.authors            = { "PavunRajTech" => "pavunraj.p@optisolbusiness.com" }
  spec.platform     = :ios, "13.0"
  spec.source       = { :git => "https://github.com/PavunRajTech/TestScanFlowCode.git",:branch => "master", :tag => spec.version.to_s}
  spec.source_files  = "TestScanFlowCode", "TestScanFlowCode/**/*.{h,m,swift}"
  spec.swift_versions = "5.0"
  spec.platform     = :ios, "13.0"
  
  end
