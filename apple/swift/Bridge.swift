// Actor-owned C ABI bridge. Rust retains the handle until free; the callback
// context is released exactly once after the native task and emissions finish.
import CoreGraphics
import Foundation
import FoundationModels
import ImageIO
import UniformTypeIdentifiers

private let cbAccepted: Int32 = 0
private let cbFull: Int32 = 1
private let cbClosed: Int32 = 2

public typealias AitherEventCallback =
    @Sendable @convention(c) (UnsafeMutableRawPointer?, Int32, UnsafePointer<UInt8>?, Int) -> Int32
public typealias AitherReleaseCallback = @Sendable @convention(c) (UnsafeMutableRawPointer?) -> Void

/// Pointer lifetime is confined to this actor; callbacks never overlap release.
actor CallbackGate {
    private var closed = false
    private let context: UInt
    private let callback: AitherEventCallback
    private let releaseCallback: AitherReleaseCallback

    init(context: UInt, cb: @escaping AitherEventCallback,
         release: @escaping AitherReleaseCallback) {
        self.context = context
        callback = cb
        releaseCallback = release
    }

    func emit(_ kind: BridgeEventKind, _ payload: [UInt8]) -> Int32 {
        guard !closed else { return cbClosed }
        return payload.withUnsafeBufferPointer {
            callback(UnsafeMutableRawPointer(bitPattern: context), kind.rawValue,
                     $0.baseAddress, $0.count)
        }
    }

    func stop() { closed = true }

    func release() {
        precondition(closed)
        releaseCallback(UnsafeMutableRawPointer(bitPattern: context))
    }
}

actor ResumeHub {
    private var waiters: [CheckedContinuation<Bool, Never>] = []
    private var generation: UInt64 = 0
    private var open = true

    func currentGeneration() -> UInt64 { generation }

    func waitForSpace(after observed: UInt64) async -> Bool {
        guard open else { return false }
        if generation != observed { return true }
        return await withCheckedContinuation { waiters.append($0) }
    }

    func resume() {
        guard open else { return }
        generation &+= 1
        let pending = waiters
        waiters.removeAll()
        for waiter in pending { waiter.resume(returning: true) }
    }

    func close() {
        open = false
        let pending = waiters
        waiters.removeAll()
        for waiter in pending { waiter.resume(returning: false) }
    }
}

enum ToolOutcome: Sendable {
    case result(String)
    case failure(String)
    case cancelled
}

private struct ToolSlot {
    var continuation: CheckedContinuation<String, any Error>?
    var outcome: ToolOutcome?
}

struct ToolCallInterrupted: Error {}

actor ToolRegistry {
    private var nextSequence: UInt64 = 0
    private var slots: [UInt64: ToolSlot] = [:]
    private var batchReported = false
    private var cancelled = false
    private weak var session: LanguageModelSession?

    func setSession(_ session: LanguageModelSession) { self.session = session }

    func beginCall() -> UInt64 {
        let sequence = nextSequence
        nextSequence += 1
        slots[sequence] = ToolSlot(outcome: cancelled ? .cancelled : nil)
        return sequence
    }

    func awaitResolution(_ sequence: UInt64) async throws -> String {
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation {
                (continuation: CheckedContinuation<String, any Error>) in
                guard let slot = slots[sequence] else {
                    preconditionFailure("tool sequence was not reserved or was consumed twice")
                }
                if let outcome = slot.outcome {
                    slots.removeValue(forKey: sequence)
                    resume(continuation, outcome: outcome)
                } else {
                    precondition(slot.continuation == nil)
                    slots[sequence]?.continuation = continuation
                }
            }
        } onCancel: {
            Task { await self.resolve(sequence, outcome: .cancelled) }
        }
    }

    private func resume(_ continuation: CheckedContinuation<String, any Error>,
                        outcome: ToolOutcome) {
        switch outcome {
        case .result(let text): continuation.resume(returning: text)
        case .failure(let message): continuation.resume(throwing: BridgeToolError(message: message))
        case .cancelled: continuation.resume(throwing: ToolCallInterrupted())
        }
    }

    @discardableResult
    func resolve(_ sequence: UInt64, outcome: ToolOutcome) -> Bool {
        guard var slot = slots[sequence], slot.outcome == nil else { return false }
        if let continuation = slot.continuation {
            slots.removeValue(forKey: sequence)
            resume(continuation, outcome: outcome)
        } else {
            slot.outcome = outcome
            slots[sequence] = slot
        }
        return true
    }

    func failAll() {
        cancelled = true
        for sequence in Array(slots.keys) { resolve(sequence, outcome: .cancelled) }
    }

    func transcriptBatch() -> [WireCapturedCall]? {
        guard let session else { return nil }
        for entry in session.transcript.reversed() {
            if case .toolCalls(let calls) = entry {
                return calls.map {
                    WireCapturedCall(id: $0.id, name: $0.toolName,
                                     arguments: $0.arguments.jsonString)
                }
            }
        }
        return nil
    }

    func markBatchReported() -> Bool {
        if batchReported { return false }
        batchReported = true
        return true
    }
}

struct BridgeToolError: Error, LocalizedError {
    var message: String
    var errorDescription: String? { message }
}

/// Configured on the Rust owning task before start, then immutable. No C entry
/// point mutates configuration after start. The engine retains it through finish.
final class RequestInput: @unchecked Sendable {
    let config: Result<WireRequest, any Error>?
    let script: Data?
    let images: [CGImage]
    init(_ handle: RequestHandle) {
        config = handle.config
        script = handle.script
        images = handle.images
    }
}

/// Only Rust's single request owner accesses these configuration fields.
final class RequestHandle {
    let owner: BridgeRequest
    var config: Result<WireRequest, any Error>?
    var script: Data?
    var images: [CGImage] = []
    init(context: UInt, cb: @escaping AitherEventCallback,
         release: @escaping AitherReleaseCallback) {
        owner = BridgeRequest(context: context, cb: cb, release: release)
    }
}

actor BridgeRequest {
    nonisolated let gate: CallbackGate
    nonisolated let hub = ResumeHub()
    nonisolated let tools = ToolRegistry()
    private var task: Task<Void, Never>?
    private var cancelled = false
    private var freed = false

    init(context: UInt, cb: @escaping AitherEventCallback,
         release: @escaping AitherReleaseCallback) {
        gate = CallbackGate(context: context, cb: cb, release: release)
    }

    @discardableResult
    func emit(_ kind: BridgeEventKind, _ payload: String) async -> Bool {
        let bytes = Array(payload.utf8)
        while true {
            if Task.isCancelled && kind != .end { return false }
            let observed = await hub.currentGeneration()
            switch await gate.emit(kind, bytes) {
            case cbAccepted: return true
            case cbFull:
                guard await hub.waitForSpace(after: observed) else { return false }
            default: return false
            }
        }
    }

    func finish(_ end: WireEnd) async {
        let payload = String(decoding: try! JSONEncoder().encode(end), as: UTF8.self)
        _ = await emit(.end, payload)
    }

    func start(_ input: RequestInput) {
        // Cancellation/free may arrive before start's actor message.
        guard !freed, !cancelled, task == nil else { return }
        task = Task { await Engine.run(self, input: input) }
    }

    func cancel() async {
        cancelled = true
        task?.cancel()
        await tools.failAll()
        await hub.close()
    }

    func free() async {
        precondition(!freed)
        freed = true
        // Close emission before joining. Actor isolation ensures no callback
        // overlaps stop; the context remains retained until the join finishes.
        await gate.stop()
        await cancel()
        if let task { await task.value }
        self.task = nil
        await gate.release()
    }
}
// MARK: - C entry points

public typealias AitherAvailabilityDetailCallback =
    @convention(c) (UnsafeMutableRawPointer?, UnsafePointer<UInt8>?, Int) -> Void

/// 0 = available; 1...4 = known reasons; 5 = unknown reason with native detail.
/// The context and callback never escape; detail bytes are borrowed only during the call.
@_cdecl("aither_apple_availability")
public func aitherAppleAvailability(
    _ context: UnsafeMutableRawPointer?,
    _ detail: AitherAvailabilityDetailCallback
) -> Int32 {
    guard #available(macOS 26.0, iOS 26.0, *) else {
        return 1
    }
    switch SystemLanguageModel.default.availability {
    case .available:
        return 0
    case .unavailable(let reason):
        switch reason {
        case .deviceNotEligible: return 2
        case .appleIntelligenceNotEnabled: return 3
        case .modelNotReady: return 4
        default:
            Array(String(describing: reason).utf8).withUnsafeBufferPointer {
                detail(context, $0.baseAddress, $0.count)
            }
            return 5
        }
    }
}

/// Native context window in tokens, or -1 when it cannot be queried.
@_cdecl("aither_apple_context_size")
public func aitherAppleContextSize() -> Int64 {
    guard #available(macOS 26.0, iOS 26.0, *) else {
        return -1
    }
    return Int64(SystemLanguageModel.default.contextSize)
}

/// Capability bitmask. Bit0 vision, bit1 guided generation, bit2 reasoning,
/// bit3 tool calling. Querying capabilities is a macOS/iOS 27 API; earlier
/// systems report the 26 baseline (guided generation + tool calling).
@_cdecl("aither_apple_capabilities")
public func aitherAppleCapabilities() -> UInt32 {
    guard #available(macOS 26.0, iOS 26.0, *) else {
        return 0
    }
    #if AITHER_SDK_27
        if #available(macOS 27.0, iOS 27.0, *) {
            var bits: UInt32 = 1 << 4 // OS27 surface (toolCallingMode, custom models)
            let capabilities = SystemLanguageModel.default.capabilities
            if capabilities.contains(.vision) { bits |= 1 }
            if capabilities.contains(.guidedGeneration) { bits |= 1 << 1 }
            if capabilities.contains(.reasoning) { bits |= 1 << 2 }
            if capabilities.contains(.toolCalling) { bits |= 1 << 3 }
            return bits
        }
    #endif
    // Baseline for the 26.x framework: guided generation and tool calling.
    return (1 << 1) | (1 << 3)
}

/// Creates a retained request object. The Rust side owns the returned pointer
/// and must pass it to `aither_apple_request_free` exactly once.
@_cdecl("aither_apple_request_new")
public func aitherAppleRequestNew(
    ctx: UnsafeMutableRawPointer?,
    cb: AitherEventCallback,
    release: AitherReleaseCallback
) -> UnsafeMutableRawPointer {
    let request = RequestHandle(context: UInt(bitPattern: ctx), cb: cb, release: release)
    return Unmanaged.passRetained(request).toOpaque()
}

private func request(_ opaque: UnsafeMutableRawPointer?) -> RequestHandle {
    Unmanaged<RequestHandle>.fromOpaque(opaque!).takeUnretainedValue()
}

/// Decodes the JSON request configuration.
@_cdecl("aither_apple_request_set_json")
public func aitherAppleRequestSetJson(
    _ opaque: UnsafeMutableRawPointer?,
    _ data: UnsafePointer<UInt8>?,
    _ length: Int
) {
    let request = request(opaque)
    let bytes = Data(bytes: data!, count: length)
    request.config = Result {
        try JSONDecoder().decode(WireRequest.self, from: bytes)
    }
}

/// Registers one image (already validated for a supported MIME type by the
/// caller). Returns 0 on success, 3 when the MIME type is not decodable by the
/// framework, 4 when the bytes do not decode to an image, 5 when the runtime
/// lacks transcript attachment support (OS < 27).
@_cdecl("aither_apple_request_add_image")
public func aitherAppleRequestAddImage(
    _ opaque: UnsafeMutableRawPointer?,
    _ data: UnsafePointer<UInt8>?,
    _ length: Int,
    _ mime: UnsafePointer<UInt8>?,
    _ mimeLength: Int
) -> Int32 {
    let request = request(opaque)
    let mimeString = String(
        decoding: UnsafeBufferPointer(start: mime, count: mimeLength),
        as: UTF8.self
    )
    guard #available(macOS 27.0, iOS 27.0, *) else {
        return 5 // image prompting needs OS 27: capability, not media error
    }
    guard let type = UTType(mimeType: mimeString),
        (CGImageSourceCopyTypeIdentifiers() as! [String]).contains(type.identifier)
    else {
        return 3
    }
    let bytes = Data(bytes: data!, count: length)
    guard let source = CGImageSourceCreateWithData(bytes as CFData, nil),
        let image = CGImageSourceCreateImageAtIndex(source, 0, nil)
    else {
        return 4
    }
    request.images.append(image)
    return 0
}

#if AITHER_SCRIPTED
/// Installs a scripted deterministic model instead of the system model.
/// Compiled only into debug/test builds (`AITHER_SCRIPTED`); requires
/// macOS/iOS 27 at runtime. Returns 0 on success, 5 when the runtime OS is
/// too old for the custom-model API.
@_cdecl("aither_apple_request_set_script")
public func aitherAppleRequestSetScript(
    _ opaque: UnsafeMutableRawPointer?,
    _ data: UnsafePointer<UInt8>?,
    _ length: Int
) -> Int32 {
    #if AITHER_SDK_27
        guard #available(macOS 27.0, iOS 27.0, *) else {
            return 5
        }
        let request = request(opaque)
        request.script = Data(bytes: data!, count: length)
        return 0
    #else
        return 5
    #endif
}
#endif

/// Starts generation. Events flow through the registered callback.
@_cdecl("aither_apple_request_start")
public func aitherAppleRequestStart(_ opaque: UnsafeMutableRawPointer?) {
    let handle = request(opaque)
    let input = RequestInput(handle)
    let owner = handle.owner
    Task { await owner.start(input) }
}

/// Cancels generation and unwinds any suspended tool continuations.
@_cdecl("aither_apple_request_cancel")
public func aitherAppleRequestCancel(_ opaque: UnsafeMutableRawPointer?) {
    let owner = request(opaque).owner
    Task { await owner.cancel() }
}

/// Signals that the Rust consumer drained a channel slot.
@_cdecl("aither_apple_request_resume")
public func aitherAppleRequestResume(_ opaque: UnsafeMutableRawPointer?) {
    let owner = request(opaque).owner
    Task { await owner.hub.resume() }
}

/// Delivers a tool result to a suspended native `Tool.call`.
/// `is_error != 0` resumes the call with a tool error.
@_cdecl("aither_apple_request_tool_result")
public func aitherAppleRequestToolResult(
    _ opaque: UnsafeMutableRawPointer?,
    _ sequence: UInt64,
    _ isError: Int32,
    _ data: UnsafePointer<UInt8>?,
    _ length: Int
) {
    let request = request(opaque)
    let text = String(
        decoding: UnsafeBufferPointer(start: data, count: length),
        as: UTF8.self
    )
    let owner = request.owner
    Task { await owner.tools.resolve(sequence, outcome: isError == 0 ? .result(text) : .failure(text)) }
}

/// Releases the request handle. Teardown runs asynchronously and ends by
/// invoking the release callback, after which no event callback can be in
/// flight or arrive later — Rust frees the callback context there.
@_cdecl("aither_apple_request_free")
public func aitherAppleRequestFree(_ opaque: UnsafeMutableRawPointer?) {
    guard let opaque else { return }
    let owner = request(opaque).owner
    Task { await owner.free() }
    Unmanaged<RequestHandle>.fromOpaque(opaque).release()
}
