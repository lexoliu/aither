// Deterministic test model: a real `LanguageModel`/`LanguageModelExecutor`
// pair driven by a JSON script instead of inference.
//
// Exists solely so tests can exercise the genuine framework machinery —
// `LanguageModelSession`, `Transcript`, `Tool` dispatch, transcript
// preservation, usage reporting — on hardware where the on-device model is
// ineligible. It never fabricates production behavior; it is compiled only
// into debug/test builds that define `AITHER_SCRIPTED` and reachable only
// through `aither_apple_request_set_script`.
#if AITHER_SCRIPTED && AITHER_SDK_27
import Foundation
import FoundationModels

@available(macOS 27.0, iOS 27.0, *)
struct ScriptedScript: Decodable, Hashable, Sendable {
    var turns: [Turn]

    struct Turn: Decodable, Hashable, Sendable {
        var steps: [Step]
    }

    struct Step: Decodable, Hashable, Sendable {
        /// Append a text fragment to the response (also used for guided
        /// content, which rides the same channel as JSON text).
        var text: String?
        /// Emit reasoning text (recorded as a reasoning transcript entry).
        var reasoning: String?
        /// Issue tool calls; the framework then invokes matching
        /// `Tool.call`s with the real `GeneratedContent` arguments.
        var toolCalls: [Call]?
        /// Report token usage.
        var usage: Usage?
        /// Emit a JSON dump of the observed generation request as response
        /// text, so tests can assert the reconstructed transcript, tool
        /// definitions and generation options.
        var probe: Bool?
        /// Suspend until the request is cancelled.
        var hang: Bool?
        /// Throw a framework error identified by code, e.g.
        /// `guardrail_violation`, `rate_limited`, `context_size_exceeded`,
        /// `refusal`, `decoding_failure`, `unsupported_language_or_locale`.
        var fail: String?
        var message: String?
        var bridgeChecks: Bool?
    }

    struct Call: Decodable, Hashable, Sendable {
        var id: String
        var name: String
        /// Arguments JSON; appended as one fragment.
        var arguments: String
    }

    struct Usage: Decodable, Hashable, Sendable {
        var input: Int
        var output: Int
        var reasoning: Int?
        var cached: Int?
    }
}

struct ScriptedError: Error, LocalizedError {
    var message: String
    var errorDescription: String? { message }
}

/// Shared suspension point for `hang` steps, resumed by task cancellation.
/// Cancellation latches: `onCancel` can fire before `suspend` registers its
/// continuation (a task that is already cancelled), so the flag and the
/// continuation swap atomically under the lock — whichever arrives second
/// completes the operation exactly once.
actor ScriptHang: Hashable {
    let started: @Sendable () async -> Void
    init(started: @escaping @Sendable () async -> Void) { self.started = started }
    private var continuation: CheckedContinuation<Void, any Error>?
    private var cancelled = false
    nonisolated static func == (lhs: ScriptHang, rhs: ScriptHang) -> Bool { lhs === rhs }
    nonisolated func hash(into hasher: inout Hasher) { hasher.combine(ObjectIdentifier(self)) }

    private func cancel() {
        cancelled = true
        continuation?.resume(throwing: CancellationError())
        continuation = nil
    }

    func suspend() async throws {
        // Explicit executor-entry signal: framework snapshot coalescing may
        // hold text until generation completes, which a hanging turn never does.
        await started()
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation {
                (pending: CheckedContinuation<Void, any Error>) in
                if cancelled { pending.resume(throwing: CancellationError()) }
                else { continuation = pending }
            }
        } onCancel: {
            Task { await self.cancel() }
        }
    }
}

@available(macOS 27.0, iOS 27.0, *)
struct ScriptedModel: LanguageModel {
    typealias Executor = ScriptedExecutor

    let script: ScriptedScript
    let hang: ScriptHang

    var capabilities: LanguageModelCapabilities {
        LanguageModelCapabilities([.guidedGeneration, .toolCalling, .vision, .reasoning])
    }

    var executorConfiguration: ScriptedExecutor.Configuration {
        ScriptedExecutor.Configuration(script: script, hang: hang)
    }
}

@available(macOS 27.0, iOS 27.0, *)
struct ScriptedExecutor: LanguageModelExecutor {
    struct Configuration: Hashable, Sendable {
        var script: ScriptedScript
        var hang: ScriptHang
    }

    let script: ScriptedScript
    let hang: ScriptHang

    init(configuration: Configuration) throws {
        script = configuration.script
        hang = configuration.hang
    }

    func prewarm(model: ScriptedModel, transcript: Transcript) {}

    func respond(
        to request: LanguageModelExecutorGenerationRequest,
        model: ScriptedModel,
        streamingInto channel: LanguageModelExecutorGenerationChannel
    ) async throws {
        let index = turnIndex(of: request.transcript)
        guard script.turns.indices.contains(index) else {
            throw ScriptedError(
                message: "no scripted turn \(index); transcript has \(request.transcript.count) entries"
            )
        }
        for step in script.turns[index].steps {
            try await perform(step, request: request, channel: channel)
        }
    }

    /// Turns are keyed by generated entries already in the transcript: each
    /// finished turn leaves one `.response` or one `.toolCalls` entry.
    private func turnIndex(of transcript: Transcript) -> Int {
        transcript.reduce(0) { count, entry in
            switch entry {
            case .response, .toolCalls: count + 1
            default: count
            }
        }
    }

    private func perform(
        _ step: ScriptedScript.Step,
        request: LanguageModelExecutorGenerationRequest,
        channel: LanguageModelExecutorGenerationChannel
    ) async throws {
        if step.bridgeChecks == true {
            let registry = ToolRegistry()
            let early = await registry.beginCall()
            let accepted = await registry.resolve(early, outcome: .result("early"))
            precondition(accepted)
            let value = try await registry.awaitResolution(early)
            precondition(value == "early")
            let duplicate = await registry.resolve(early, outcome: .result("duplicate"))
            precondition(!duplicate)
            let cancelled = await registry.beginCall()
            await registry.failAll()
            do {
                _ = try await registry.awaitResolution(cancelled)
                preconditionFailure("cancel before suspension was lost")
            } catch is ToolCallInterrupted {}
            let late = await registry.beginCall()
            do {
                _ = try await registry.awaitResolution(late)
                preconditionFailure("late tool escaped cancellation")
            } catch is ToolCallInterrupted {}
            let hub = ResumeHub()
            let observed = await hub.currentGeneration()
            await hub.resume()
            let credit = await hub.waitForSpace(after: observed)
            precondition(credit)
            await hub.close()
            let closed = await hub.waitForSpace(after: observed)
            precondition(!closed)
        }
        if let text = step.text {
            await channel.send(
                .response(action: .appendText(text, tokenCount: text.utf8.count))
            )
        }
        if let reasoning = step.reasoning {
            await channel.send(
                .reasoning(action: .appendText(reasoning, tokenCount: reasoning.utf8.count))
            )
        }
        if let calls = step.toolCalls {
            for call in calls {
                await channel.send(
                    .toolCalls(
                        action: .toolCall(
                            id: call.id,
                            name: call.name,
                            action: .appendArguments(
                                call.arguments,
                                tokenCount: call.arguments.utf8.count
                            )
                        )
                    )
                )
            }
        }
        if let usage = step.usage {
            await channel.send(
                .response(
                    action: .updateUsage(
                        input: .init(
                            totalTokenCount: usage.input,
                            cachedTokenCount: usage.cached ?? 0
                        ),
                        output: .init(
                            totalTokenCount: usage.output,
                            reasoningTokenCount: usage.reasoning ?? 0
                        )
                    )
                )
            )
        }
        if step.probe == true {
            await channel.send(
                .response(
                    action: .appendText(
                        probeJson(of: request),
                        tokenCount: 1
                    )
                )
            )
        }
        if let code = step.fail {
            throw scriptedError(code: code, message: step.message ?? "scripted failure")
        }
        if step.hang == true {
            try await hang.suspend()
        }
    }

    private func probeJson(of request: LanguageModelExecutorGenerationRequest) -> String {
        struct Report: Encodable {
            var entries: [String]
            var toolDefinitions: [String]
            var temperature: Double?
            var maximumResponseTokens: Int?
            var toolCallingMode: String?
            var hasSchema: Bool
        }
        var entries: [String] = []
        for entry in request.transcript {
            switch entry {
            case .instructions(let instructions):
                let text = instructions.segments.compactMap { segment -> String? in
                    if case .text(let part) = segment { return part.content }
                    return nil
                }.joined(separator: "|")
                entries.append("instructions:\(text)")
            case .prompt(let prompt):
                let text = prompt.segments.compactMap { segment -> String? in
                    if case .text(let part) = segment { return part.content }
                    return nil
                }.joined(separator: "|")
                entries.append("prompt:\(text)")
            case .response(let response):
                let text = response.segments.compactMap { segment -> String? in
                    if case .text(let part) = segment { return part.content }
                    return nil
                }.joined(separator: "|")
                entries.append("response:\(text)")
            case .toolCalls(let calls):
                let joined = calls.map { "\($0.id):\($0.toolName):\($0.arguments.jsonString)" }
                    .joined(separator: ",")
                entries.append("toolCalls:\(joined)")
            case .toolOutput(let output):
                let text = output.segments.compactMap { segment -> String? in
                    if case .text(let part) = segment { return part.content }
                    return nil
                }.joined(separator: "|")
                entries.append("toolOutput:\(output.id):\(output.toolName):\(text)")
            case .reasoning(let reasoning):
                let text = reasoning.segments.compactMap { segment -> String? in
                    if case .text(let part) = segment { return part.content }
                    return nil
                }.joined(separator: "|")
                entries.append("reasoning:\(text)")
            @unknown default:
                entries.append("unknown")
            }
        }
        let options = request.generationOptions
        let report = Report(
            entries: entries,
            toolDefinitions: request.enabledToolDefinitions.map(\.name),
            temperature: options.temperature,
            maximumResponseTokens: options.maximumResponseTokens,
            toolCallingMode: options.toolCallingMode.map { "\($0.kind)" },
            hasSchema: request.schema != nil
        )
        let data = try! JSONEncoder().encode(report)
        return String(decoding: data, as: UTF8.self)
    }

    private func scriptedError(code: String, message: String) -> any Error {
        let context = LanguageModelSession.GenerationError.Context(debugDescription: message)
        switch code {
        case "guardrail_violation":
            return LanguageModelSession.GenerationError.guardrailViolation(context)
        case "rate_limited":
            return LanguageModelSession.GenerationError.rateLimited(context)
        case "context_size_exceeded":
            return LanguageModelSession.GenerationError.exceededContextWindowSize(context)
        case "refusal":
            return LanguageModelSession.GenerationError.refusal(
                .init(transcriptEntries: []),
                context
            )
        case "decoding_failure":
            return LanguageModelSession.GenerationError.decodingFailure(context)
        case "unsupported_language_or_locale":
            return LanguageModelSession.GenerationError.unsupportedLanguageOrLocale(context)
        case "concurrent_requests":
            return LanguageModelSession.GenerationError.concurrentRequests(context)
        case "model_not_ready":
            return LanguageModelSession.GenerationError.assetsUnavailable(context)
        case "unsupported_guide":
            return LanguageModelSession.GenerationError.unsupportedGuide(context)
        default:
            return ScriptedError(message: "\(code): \(message)")
        }
    }
}
#endif
