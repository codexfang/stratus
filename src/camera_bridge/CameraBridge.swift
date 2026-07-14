import AVFoundation
import Foundation

let cameraName = "Arducam 1080P Low Light"
let outputPath = "/tmp/stratus_frame.jpg"

let discovery = AVCaptureDevice.DiscoverySession(
    deviceTypes: [.external, .builtInWideAngleCamera],
    mediaType: .video,
    position: .unspecified)

guard let device = discovery.devices.first(where: { $0.localizedName == cameraName })
          ?? discovery.devices.first else {
    print("ERROR: no camera")
    exit(1)
}

let semaphore = DispatchSemaphore(value: 0)

AVCaptureDevice.requestAccess(for: .video) { granted in
    guard granted else { print("ERROR: denied"); semaphore.signal(); return }
    let session = AVCaptureSession()
    session.sessionPreset = .high
    guard let input = try? AVCaptureDeviceInput(device: device),
          session.canAddInput(input) else {
        print("ERROR: input"); semaphore.signal(); return
    }
    session.addInput(input)

    let output = AVCaptureVideoDataOutput()
    output.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
    guard session.canAddOutput(output) else {
        print("ERROR: output"); semaphore.signal(); return
    }
    session.addOutput(output)

    let delegate = CaptureDelegate(path: outputPath, sem: semaphore)
    output.setSampleBufferDelegate(delegate, queue: DispatchQueue(label: "cam"))
    session.startRunning()
}
_ = semaphore.wait(timeout: .now() + 10)
print("OK")
