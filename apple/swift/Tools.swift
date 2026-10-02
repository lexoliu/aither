// Native tool bridge.
//
// Every declared tool is a `BridgedTool`: a real `FoundationModels.Tool` whose
// `call` suspends on a continuation registered with `ToolRegistry`.
//
// In `internalTools` mode the Rust driver answers each call through
// `aither_apple_request_tool_result` and generation continues.
//
// In `externalTools` mode the first suspension captures the complete native
// batch — the framework appends the `toolCalls` transcript entry before
// invoking any tool — reports it once to Rust, and the suspended calls are
// unwound by task cancellation when Rust ends the turn. Identical tool names
// called multiple times in one batch are preserved because each `call` is a
// separate invocation with its own continuation.
import Foundation
import FoundationModels

@available(macOS 26.0, iOS 26.0, *)
final class BridgedTool: Tool, @unchecked Sendable {
    typealias Arguments = GeneratedContent
    typealias Output = String

    let toolName: String
    let toolDescription: String
    let schema: GenerationSchema
    let registry: ToolRegistry
    let external: Bool
    let emit: @Sendable (BridgeEventKind, String) async -> Bool

    var name: String { toolName }
    var description: String { toolDescription }
    var parameters: GenerationSchema { schema }
    var includesSchemaInInstructions: Bool { false }

    init(
        name: String,
        toolDescription: String,
        schema: GenerationSchema,
        registry: ToolRegistry,
        external: Bool,
        emit: @escaping @Sendable (BridgeEventKind, String) async -> Bool
    ) {
        toolName = name
        self.toolDescription = toolDescription
        self.schema = schema
        self.registry = registry
        self.external = external
        self.emit = emit
    }

    func call(arguments: GeneratedContent) async throws -> String {
        // The sequence is reserved before any emission so a fast Rust answer
        // or a cancellation can never race ahead of the suspension.
        let sequence = await registry.beginCall()
        if external {
            // One batch event per turn: the framework appends the complete
            // `toolCalls` transcript entry — native ids and arguments,
            // including duplicate tool names — before invoking any tool.
            // An unreadable transcript is an error, never a partial batch
            // with fabricated ids.
            if await registry.markBatchReported() {
                guard let calls = await registry.transcriptBatch() else {
                    throw BridgeError(
                        message: "native transcript did not expose the tool-call batch"
                    )
                }
                let payload = try encodeBatch(calls)
                _ = await emit(.toolBatch, payload)
            }
        } else {
            let payload = try encodeToolEvent(
                sequence: sequence,
                name: toolName,
                arguments: arguments.jsonString
            )
            _ = await emit(.toolCall, payload)
        }
        return try await registry.awaitResolution(sequence)
    }
}

@available(macOS 26.0, iOS 26.0, *)
private func encodeBatch(_ calls: [WireCapturedCall]) throws -> String {
    struct Batch: Encodable {
        var calls: [WireCapturedCall]
    }
    let data = try JSONEncoder().encode(Batch(calls: calls))
    return String(decoding: data, as: UTF8.self)
}

@available(macOS 26.0, iOS 26.0, *)
private func encodeToolEvent(sequence: UInt64, name: String, arguments: String) throws -> String {
    struct Event: Encodable {
        var seq: UInt64
        var name: String
        var arguments: String
    }
    let event = Event(
        seq: sequence,
        name: name,
        arguments: arguments
    )
    let data = try JSONEncoder().encode(event)
    return String(decoding: data, as: UTF8.self)
}
