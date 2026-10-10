// Session construction and generation driving.
//
// One `Engine.run` drives one independent `LanguageModelSession`. The Rust
// side sends the complete conversation as wire transcript entries; the final
// user content becomes the `Prompt` the session generates from. Tool calls in
// `internalTools` mode suspend inside `Tool.call` until Rust answers through
// `aither_apple_request_tool_result`; in `externalTools` mode the complete
// native batch is captured once and the suspended calls are unwound by
// cancellation when Rust ends the turn.
import Foundation
import FoundationModels

@available(macOS 26.0, iOS 26.0, *)
enum Engine {
    static func run(_ request: BridgeRequest, input: RequestInput) async {
        guard let configResult = input.config else {
            await request.finish(.init(status: .error, code: "config", message: "request used before configuration"))
            return
        }
        let config: WireRequest
        switch configResult {
        case .success(let decoded):
            config = decoded
        case .failure(let error):
            await request.finish(
                .init(status: .error, code: "config", message: String(describing: error))
            )
            return
        }

        do {
            let tools = try makeTools(request: request, config: config)
            let transcript = try makeTranscript(config: config, request: input, tools: tools)
            let session = try makeSession(
                owner: request,
                request: input,
                transcript: transcript,
                tools: tools
            )
            await request.tools.setSession(session)
            let options = try makeOptions(config: config)
            let prompt = try makePrompt(config: config, request: input)

            switch config.mode {
            case .guided:
                guard let wireSchema = config.schema else {
                    throw BridgeError(message: "guided mode requires a schema")
                }
                let schema = try buildGenerationSchema(wireSchema)
                // Structured snapshots are revised JSON, not append-only: the
                // Rust side keeps only the latest and emits it at completion.
                try await drive(
                    request: request,
                    config: config,
                    stream: session.streamResponse(
                        to: prompt,
                        schema: schema,
                        options: options
                    ),
                    kind: .structured,
                    content: { $0.jsonString }
                )
            default:
                try await drive(
                    request: request,
                    config: config,
                    stream: session.streamResponse(to: prompt, options: options),
                    kind: .text,
                    content: { $0 }
                )
            }
        } catch {
            await request.finish(endEvent(for: error))
        }
    }

    private static func drive<C: Generable>(
        request: BridgeRequest,
        config: WireRequest,
        stream: LanguageModelSession.ResponseStream<C>,
        kind: BridgeEventKind,
        content: (C.PartiallyGenerated) -> String
    ) async throws {
        var iterator = stream.makeAsyncIterator()
        var usage: WireUsage?
        var reasoning = ""
        while let snapshot = try await iterator.next() {
            if Task.isCancelled {
                throw CancellationError()
            }
            _ = await request.emit(kind, content(snapshot.content))
            #if AITHER_SDK_27
                if #available(macOS 27.0, iOS 27.0, *) {
                    usage = WireUsage(
                        input: snapshot.usage.input.totalTokenCount,
                        output: snapshot.usage.output.totalTokenCount,
                        reasoning: snapshot.usage.output.reasoningTokenCount,
                        cached: snapshot.usage.input.cachedTokenCount
                    )
                    if config.includeReasoning {
                        let updated = reasoningText(of: snapshot.transcriptEntries)
                        if updated != reasoning {
                            reasoning = updated
                            _ = await request.emit(.reasoning, reasoning)
                        }
                    }
                }
            #endif
        }
        try Task.checkCancellation()
        await request.finish(WireEnd(status: .completed, usage: usage))
    }

    private static func makeTools(
        request: BridgeRequest,
        config: WireRequest
    ) throws -> [BridgedTool] {
        try config.tools.map { tool in
            BridgedTool(
                name: tool.name,
                toolDescription: tool.description,
                schema: try buildGenerationSchema(tool.schema),
                registry: request.tools,
                external: config.toolsAreExternal,
                emit: { kind, payload in await request.emit(kind, payload) }
            )
        }
    }

    private static func makeTranscript(
        config: WireRequest,
        request: RequestInput,
        tools: [BridgedTool]
    ) throws -> Transcript {
        var entries: [Transcript.Entry] = []
        // Tool definitions live on the instructions entry: it must be
        // emitted whenever tools exist, even with no system message.
        if !config.instructions.isEmpty || !tools.isEmpty {
            let segments = config.instructions.map {
                Transcript.Segment.text(Transcript.TextSegment(content: $0))
            }
            let toolDefinitions = tools.map(Transcript.ToolDefinition.init(tool:))
            entries.append(
                .instructions(
                    Transcript.Instructions(
                        segments: segments,
                        toolDefinitions: toolDefinitions
                    )
                )
            )
        }
        for (index, entry) in config.history.enumerated() {
            entries.append(try makeEntry(entry, request: request, position: index))
        }
        return Transcript(entries: entries)
    }

    private static func makeEntry(
        _ entry: WireEntry,
        request: RequestInput,
        position: Int
    ) throws -> Transcript.Entry {
        switch entry.kind {
        case .instructions:
            return .instructions(
                Transcript.Instructions(
                    segments: [.text(Transcript.TextSegment(content: entry.text ?? ""))],
                    toolDefinitions: []
                )
            )
        case .prompt:
            return .prompt(
                Transcript.Prompt(
                    segments: try promptSegments(
                        text: entry.text ?? "",
                        imageIndexes: entry.imageIndexes,
                        request: request
                    )
                )
            )
        case .response:
            return .response(
                Transcript.Response(
                    assetIDs: [],
                    segments: [.text(Transcript.TextSegment(content: entry.text ?? ""))]
                )
            )
        case .toolCalls:
            let calls = try entry.calls.map { call in
                try Transcript.ToolCall(
                    id: call.id,
                    toolName: call.name,
                    arguments: GeneratedContent(json: call.arguments)
                )
            }
            return .toolCalls(Transcript.ToolCalls(calls))
        case .toolOutput:
            guard let id = entry.id, let name = entry.name else {
                throw BridgeError(
                    message: "tool output at position \(position) missing id or name"
                )
            }
            return .toolOutput(
                Transcript.ToolOutput(
                    id: id,
                    toolName: name,
                    segments: [.text(Transcript.TextSegment(content: entry.text ?? ""))]
                )
            )
        }
    }

    private static func promptSegments(
        text: String,
        imageIndexes: [Int],
        request: RequestInput
    ) throws -> [Transcript.Segment] {
        var segments: [Transcript.Segment] = [
            .text(Transcript.TextSegment(content: text))
        ]
        for index in imageIndexes {
            guard request.images.indices.contains(index) else {
                throw BridgeError(message: "image index \(index) out of range")
            }
            #if AITHER_SDK_27
                if #available(macOS 27.0, iOS 27.0, *) {
                    segments.append(
                        .attachment(
                            Transcript.AttachmentSegment(
                                content: .image(
                                    Transcript.ImageAttachment(request.images[index])
                                )
                            )
                        )
                    )
                    continue
                }
                throw BridgeError(message: "image attachments require macOS/iOS 27")
            #else
                throw BridgeError(message: "image attachments require SDK 27")
            #endif
        }
        return segments
    }

    private static func makeSession(
        owner: BridgeRequest,
        request: RequestInput,
        transcript: Transcript,
        tools: [BridgedTool]
    ) throws -> LanguageModelSession {
        #if AITHER_SCRIPTED
            if let script = request.script {
                #if AITHER_SDK_27
                    if #available(macOS 27.0, iOS 27.0, *) {
                        let decoded = try JSONDecoder().decode(
                            ScriptedScript.self, from: script
                        )
                        let session = LanguageModelSession(
                            model: ScriptedModel(script: decoded, hang: ScriptHang(started: {
                                _ = await owner.emit(.text, "started")
                            })),
                            tools: tools,
                            transcript: transcript
                        )
                        session.transcriptErrorHandlingPolicy = .preserveTranscript
                        return session
                    }
                    throw BridgeError(message: "scripted models require macOS/iOS 27")
                #else
                    throw BridgeError(message: "scripted models require SDK 27")
                #endif
            }
        #endif
        let session = LanguageModelSession(
            model: SystemLanguageModel.default,
            tools: tools,
            transcript: transcript
        )
        #if AITHER_SDK_27
            if #available(macOS 27.0, iOS 27.0, *) {
                session.transcriptErrorHandlingPolicy = .preserveTranscript
            }
        #endif
        return session
    }

    private static func makePrompt(
        config: WireRequest,
        request: RequestInput
    ) throws -> Prompt {
        #if AITHER_SDK_27
            if #available(macOS 27.0, iOS 27.0, *), !config.prompt.imageIndexes.isEmpty {
                var attachments: [Attachment<ImageAttachmentContent>] = []
                for index in config.prompt.imageIndexes {
                    guard request.images.indices.contains(index) else {
                        throw BridgeError(message: "image index \(index) out of range")
                    }
                    attachments.append(Attachment(request.images[index]))
                }
                if config.prompt.text.isEmpty {
                    return PromptBuilder.buildBlock(attachments)
                }
                return PromptBuilder.buildBlock(config.prompt.text, attachments)
            }
        #endif
        return Prompt(config.prompt.text)
    }

    private static func makeOptions(config: WireRequest) throws -> GenerationOptions {
        var options = GenerationOptions()
        options.temperature = config.options.temperature
        options.maximumResponseTokens = config.options.maximumResponseTokens
        if let sampling = config.options.sampling {
            switch sampling.kind {
            case .greedy:
                options.samplingMode = .greedy
            case .topK:
                guard let k = sampling.k else {
                    throw BridgeError(message: "top_k sampling requires k")
                }
                options.samplingMode = .random(top: k, seed: sampling.seed)
            case .topP:
                guard let p = sampling.p else {
                    throw BridgeError(message: "top_p sampling requires p")
                }
                options.samplingMode = .random(probabilityThreshold: p, seed: sampling.seed)
            }
        }
        #if AITHER_SDK_27
            if #available(macOS 27.0, iOS 27.0, *), let mode = config.options.toolCallingMode {
                switch mode {
                case "required": options.toolCallingMode = .required
                case "allowed": options.toolCallingMode = .allowed
                default:
                    throw BridgeError(message: "unknown toolCallingMode '\(mode)'")
                }
            } else if config.options.toolCallingMode != nil {
                throw BridgeError(message: "toolCallingMode requires macOS/iOS 27")
            }
        #else
            if config.options.toolCallingMode != nil {
                throw BridgeError(message: "toolCallingMode requires SDK 27")
            }
        #endif
        return options
    }

    /// Cumulative reasoning text across the snapshot's transcript entries.
    #if AITHER_SDK_27
        @available(macOS 27.0, iOS 27.0, *)
        private static func reasoningText(of entries: ArraySlice<Transcript.Entry>) -> String {
            var text = ""
            for entry in entries {
                if case .reasoning(let reasoning) = entry {
                    for segment in reasoning.segments {
                        if case .text(let part) = segment {
                            text += part.content
                        }
                    }
                }
            }
            return text
        }
    #endif
}

struct BridgeError: Error, LocalizedError {
    var message: String
    var errorDescription: String? { message }
}

/// Maps any thrown error to the terminal event the Rust driver understands.
func endEvent(for error: any Error) -> WireEnd {
    if error is CancellationError || error is ToolCallInterrupted {
        return WireEnd(status: .cancelled)
    }
    if #available(macOS 26.0, iOS 26.0, *) {
        if let error = error as? LanguageModelSession.GenerationError {
            switch error {
            case .exceededContextWindowSize(let context):
                return WireEnd(
                    status: .error,
                    code: "context_size_exceeded",
                    message: context.debugDescription
                )
            case .assetsUnavailable(let context):
                return WireEnd(
                    status: .error,
                    code: "model_not_ready",
                    message: context.debugDescription
                )
            case .guardrailViolation(let context):
                return WireEnd(
                    status: .error,
                    code: "guardrail_violation",
                    message: context.debugDescription
                )
            case .unsupportedGuide(let context):
                return WireEnd(
                    status: .error,
                    code: "unsupported_guide",
                    message: context.debugDescription
                )
            case .unsupportedLanguageOrLocale(let context):
                return WireEnd(
                    status: .error,
                    code: "unsupported_language_or_locale",
                    message: context.debugDescription
                )
            case .decodingFailure(let context):
                return WireEnd(
                    status: .error,
                    code: "decoding_failure",
                    message: context.debugDescription
                )
            case .rateLimited(let context):
                return WireEnd(
                    status: .error,
                    code: "rate_limited",
                    message: context.debugDescription
                )
            case .concurrentRequests(let context):
                return WireEnd(
                    status: .error,
                    code: "concurrent_requests",
                    message: context.debugDescription
                )
            case .refusal(_, let context):
                return WireEnd(
                    status: .error,
                    code: "refusal",
                    message: context.debugDescription
                )
            @unknown default:
                return WireEnd(
                    status: .error,
                    code: "unknown",
                    message: String(describing: error)
                )
            }
        }
        if let error = error as? LanguageModelSession.ToolCallError {
            if error.underlyingError is ToolCallInterrupted || error.underlyingError is CancellationError {
                return WireEnd(status: .cancelled)
            }
            return WireEnd(
                status: .error,
                code: "tool_error",
                message: error.underlyingError.localizedDescription,
                tool: error.tool.name
            )
        }
        #if AITHER_SDK_27
            if #available(macOS 27.0, iOS 27.0, *) {
                if let error = error as? LanguageModelError {
                    switch error {
                    case .contextSizeExceeded(let detail):
                        return WireEnd(
                            status: .error,
                            code: "context_size_exceeded",
                            message: "\(detail.debugDescription) (\(detail.tokenCount)/\(detail.contextSize))"
                        )
                    case .rateLimited(let detail):
                        return WireEnd(
                            status: .error,
                            code: "rate_limited",
                            message: detail.debugDescription
                        )
                    case .guardrailViolation(let detail):
                        return WireEnd(
                            status: .error,
                            code: "guardrail_violation",
                            message: detail.debugDescription
                        )
                    case .refusal(let detail):
                        return WireEnd(
                            status: .error,
                            code: "refusal",
                            message: detail.debugDescription
                        )
                    case .unsupportedCapability(let detail):
                        return WireEnd(
                            status: .error,
                            code: "unsupported_capability",
                            message: detail.debugDescription
                        )
                    case .unsupportedTranscriptContent(let detail):
                        return WireEnd(
                            status: .error,
                            code: "unsupported_transcript_content",
                            message: detail.debugDescription
                        )
                    case .unsupportedGenerationGuide(let detail):
                        return WireEnd(
                            status: .error,
                            code: "unsupported_guide",
                            message: detail.debugDescription
                        )
                    case .unsupportedLanguageOrLocale(let detail):
                        return WireEnd(
                            status: .error,
                            code: "unsupported_language_or_locale",
                            message: detail.debugDescription
                        )
                    case .timeout(let detail):
                        return WireEnd(
                            status: .error,
                            code: "timeout",
                            message: detail.debugDescription
                        )
                    @unknown default:
                        return WireEnd(
                            status: .error,
                            code: "unknown",
                            message: String(describing: error)
                        )
                    }
                }
            }
        #endif
    }
    if let error = error as? BridgeToolError {
        return WireEnd(status: .error, code: "tool_error", message: error.message)
    }
    if let error = error as? BridgeError {
        return WireEnd(status: .error, code: "invalid_request", message: error.message)
    }
    return WireEnd(
        status: .error,
        code: "unknown",
        message: String(describing: error)
    )
}
