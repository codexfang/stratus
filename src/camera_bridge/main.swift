import AVFoundation
import Foundation

let cameraName = CommandLine.arguments.count > 1 ? CommandLine.arguments[1] : "Arducam 1080P Low Light"
let outputPath = CommandLine.arguments.count > 2 ? CommandLine.arguments[2] : "/tmp/stratus_frame.jpg"

let discovery = AVCaptureDevice.DiscoverySession(
    deviceTypes: [.external, .builtInWideAngleCamera],
    mediaType: .video,
    position: .unspecified)

guard let device = discovery.devices.first(where: { $0.localizedName == cameraName }) ?? discovery.devices.first else {
    print("ERROR: No camera found")
    exit(1)
}

var session: AVCaptureSession?
let semaphore = DispatchSemaphore(value: 0)

// Request permission
AVCaptureDevice.requestAccess(for: .video) { granted in
    guard granted else {
        print("ERROR: Camera access denied")
        semaphore.signal()
        return
    }
    session = AVCaptureSession()
    session?.sessionPreset = .high
    guard let input = try? AVCaptureDeviceInput(device: device) else {
        print("ERROR: Could not create input")
        semaphore.signal()
        return
    }
    guard session?.canAddInput(input) == true else {
        print("ERROR: Could not add input")
        semaphore.signal()
        return
    }
    session?.addInput(input)

    let output = AVCaptureVideoDataOutput()
    output.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
    guard session?.canAddOutput(output) == true else {
        print("ERROR: Could not add output")
        semaphore.signal()
        return
    }
    session?.addOutput(output)

    let delegate = CaptureDelegate(outputPath: outputPath, semaphore: semaphore)
    output.setSampleBufferDelegate(delegate, queue: DispatchQueue(label: "camera"))
    session?.startRunning()
}
semaphore.wait()
print("OK: \(outputPath)")
