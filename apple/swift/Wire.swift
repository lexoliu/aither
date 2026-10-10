// Wire format shared between the Rust driver and this bridge.
//
// Everything crossing the C boundary is serialized JSON: requests travel
// Rust -> Swift once at request setup, and events travel Swift -> Rust as
// (kind, UTF-8 payload) pairs through the event callback. Wire keys are
// snake_case; `src/wire.rs` holds the matching serde types.
import Foundation

/// One message converted into a `Transcript.Entry` before the final prompt.
struct WireEntry: Decodable {
    enum Kind: String, Decodable {
        case instructions
        case prompt
        case response
        case toolCalls = "tool_calls"
        case toolOutput = "tool_output"
    }

    var kind: Kind
    /// `instructions`, `prompt`, `response`, `tool_output` text payload.
    var text: String?
    /// Image slots (registered via `aither_apple_request_add_image`).
    var imageIndexes: [Int]
    /// `tool_calls` batch.
    var calls: [WireToolCall]
    /// `tool_output` correlation id and tool name.
    var id: String?
    var name: String?

    private enum CodingKeys: String, CodingKey {
        case kind, text, calls, id, name
        case imageIndexes = "image_indexes"
    }

    init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        kind = try c.decode(Kind.self, forKey: .kind)
        text = try c.decodeIfPresent(String.self, forKey: .text)
        imageIndexes = try c.decodeIfPresent([Int].self, forKey: .imageIndexes) ?? []
        calls = try c.decodeIfPresent([WireToolCall].self, forKey: .calls) ?? []
        id = try c.decodeIfPresent(String.self, forKey: .id)
        name = try c.decodeIfPresent(String.self, forKey: .name)
    }
}

struct WireToolCall: Decodable {
    var id: String
    var name: String
    /// Arguments as a JSON document, parsed into `GeneratedContent`.
    var arguments: String
}

struct WirePrompt: Decodable {
    var text: String
    var imageIndexes: [Int]

    private enum CodingKeys: String, CodingKey {
        case text
        case imageIndexes = "image_indexes"
    }

    init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        text = try c.decode(String.self, forKey: .text)
        imageIndexes = try c.decodeIfPresent([Int].self, forKey: .imageIndexes) ?? []
    }
}

/// The response stream shape; guided requires `schema`.
enum WireMode: String, Decodable {
    /// Plain text generation.
    case text
    /// Guided generation against `schema`.
    case guided
}

struct WireSampling: Decodable {
    enum Kind: String, Decodable {
        case greedy
        case topK = "top_k"
        case topP = "top_p"
    }

    var kind: Kind
    var k: Int?
    var p: Double?
    var seed: UInt64?
}

struct WireOptions: Decodable {
    var temperature: Double?
    var maximumResponseTokens: Int?
    var sampling: WireSampling?
    /// `required` or `allowed`; only meaningful on macOS/iOS 27+.
    var toolCallingMode: String?

    private enum CodingKeys: String, CodingKey {
        case temperature, sampling
        case maximumResponseTokens = "maximum_response_tokens"
        case toolCallingMode = "tool_calling_mode"
    }
}

/// Provider-neutral schema tree the Rust side produces from a schemars schema.
/// Recursive: decoded by the schema builder in Engine.swift.
indirect enum WireSchema: Decodable {
    case object(name: String, description: String?, properties: [Property])
    case anyOf(name: String, description: String?, choices: [WireSchema])
    case enumeration(name: String, description: String?, values: [String])
    case primitive(name: String, description: String?, kind: PrimKind, guides: Guides)
    case array(
        name: String,
        description: String?,
        items: WireSchema,
        minItems: Int?,
        maxItems: Int?
    )
    case reference(name: String)
    case null

    /// The declared schema name where one exists; references and null carry
    /// none and fall back to the surrounding property name.
    var name: String {
        switch self {
        case .object(let name, _, _),
            .anyOf(let name, _, _),
            .enumeration(let name, _, _),
            .primitive(let name, _, _, _),
            .array(let name, _, _, _, _):
            name
        case .reference(let name):
            name
        case .null:
            "null"
        }
    }

    enum PrimKind: String, Decodable {
        case string
        case integer
        case number
        case boolean
    }

    struct Property: Decodable {
        var name: String
        var description: String?
        var schema: WireSchema
        var optional: Bool

        init(from decoder: any Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            name = try c.decode(String.self, forKey: .name)
            description = try c.decodeIfPresent(String.self, forKey: .description)
            schema = try c.decode(WireSchema.self, forKey: .schema)
            optional = try c.decodeIfPresent(Bool.self, forKey: .optional) ?? false
        }

        private enum CodingKeys: String, CodingKey {
            case name, description, schema, optional
        }
    }

    struct Guides: Decodable {
        var integerMinimum: Int?
        var integerMaximum: Int?
        var minimum: Double?
        var maximum: Double?
        var pattern: String?

        private enum CodingKeys: String, CodingKey {
            case integerMinimum = "integer_minimum"
            case integerMaximum = "integer_maximum"
            case minimum, maximum, pattern
        }
    }

    private enum CodingKeys: String, CodingKey {
        case type, name, description, properties, choices, values, kind
        case guides, items, ref
        case minItems = "min_items"
        case maxItems = "max_items"
    }

    init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let type = try c.decode(String.self, forKey: .type)
        let name = try c.decodeIfPresent(String.self, forKey: .name) ?? "field"
        let description = try c.decodeIfPresent(String.self, forKey: .description)
        switch type {
        case "object":
            let props = try c.decode([Property].self, forKey: .properties)
            self = .object(name: name, description: description, properties: props)
        case "any_of":
            let choices = try c.decode([WireSchema].self, forKey: .choices)
            self = .anyOf(name: name, description: description, choices: choices)
        case "enumeration":
            let values = try c.decode([String].self, forKey: .values)
            self = .enumeration(name: name, description: description, values: values)
        case "primitive":
            let kind = try c.decode(PrimKind.self, forKey: .kind)
            let guides = try c.decodeIfPresent(Guides.self, forKey: .guides) ?? Guides()
            self = .primitive(
                name: name,
                description: description,
                kind: kind,
                guides: guides
            )
        case "array":
            let items = try c.decode(WireSchema.self, forKey: .items)
            self = .array(
                name: name,
                description: description,
                items: items,
                minItems: try c.decodeIfPresent(Int.self, forKey: .minItems),
                maxItems: try c.decodeIfPresent(Int.self, forKey: .maxItems)
            )
        case "reference":
            self = .reference(name: try c.decode(String.self, forKey: .ref))
        case "null":
            self = .null
        default:
            throw DecodingError.dataCorruptedError(
                forKey: .type,
                in: c,
                debugDescription: "unknown wire schema type '\(type)'"
            )
        }
    }
}

/// A dynamic schema plus its named dependencies (from `$defs`).
struct WireSchemaRoot: Decodable {
    var schema: WireSchema
    var defs: [String: WireSchema]

    init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        schema = try c.decode(WireSchema.self, forKey: .schema)
        defs = try c.decodeIfPresent([String: WireSchema].self, forKey: .defs) ?? [:]
    }

    private enum CodingKeys: String, CodingKey {
        case schema, defs
    }
}

struct WireTool: Decodable {
    var name: String
    var description: String
    var schema: WireSchemaRoot
}

struct WireRequest: Decodable {
    var instructions: [String]
    var history: [WireEntry]
    var prompt: WirePrompt
    var mode: WireMode
    var options: WireOptions
    var tools: [WireTool]
    /// `true`: suspended tool calls are captured for the caller (`respond`).
    /// `false`: Rust resolves them while generation continues
    /// (`respond_with_tools`).
    var toolsAreExternal: Bool
    var schema: WireSchemaRoot?
    var includeReasoning: Bool

    private enum CodingKeys: String, CodingKey {
        case instructions, history, prompt, mode, options, tools, schema
        case includeReasoning = "include_reasoning"
        case toolsAreExternal = "tools_are_external"
    }

    init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        instructions = try c.decodeIfPresent([String].self, forKey: .instructions) ?? []
        history = try c.decodeIfPresent([WireEntry].self, forKey: .history) ?? []
        prompt = try c.decode(WirePrompt.self, forKey: .prompt)
        mode = try c.decode(WireMode.self, forKey: .mode)
        options = try c.decode(WireOptions.self, forKey: .options)
        tools = try c.decodeIfPresent([WireTool].self, forKey: .tools) ?? []
        toolsAreExternal =
            try c.decodeIfPresent(Bool.self, forKey: .toolsAreExternal) ?? true
        schema = try c.decodeIfPresent(WireSchemaRoot.self, forKey: .schema)
        includeReasoning = try c.decodeIfPresent(Bool.self, forKey: .includeReasoning) ?? false
    }
}

/// Event kinds sent through the C event callback.
enum BridgeEventKind: Int32 {
    /// Cumulative response text snapshot (UTF-8 string payload).
    case text = 1
    /// Cumulative reasoning text snapshot.
    case reasoning = 2
    /// Revised structured JSON snapshot (guided generation).
    case structured = 3
    /// One tool request needing resolution (internal-tools mode).
    case toolCall = 4
    /// Complete tool-call batch captured at the external boundary.
    case toolBatch = 5
    /// Terminal event; no callbacks follow. Payload carries status JSON.
    case end = 6
}

/// Payload for the terminal `end` event.
struct WireEnd: Encodable {
    enum Status: String, Encodable {
        case completed
        case cancelled
        case error
    }

    var status: Status
    /// Stable machine-readable code for `error`.
    var code: String?
    var message: String?
    var tool: String?
    /// OS27+ token usage when the framework reports it.
    var usage: WireUsage?

    init(status: Status, code: String? = nil, message: String? = nil, usage: WireUsage? = nil, tool: String? = nil) {
        self.status = status
        self.code = code
        self.message = message
        self.tool = tool
        self.usage = usage
    }
}

struct WireUsage: Encodable {
    var input: Int
    var output: Int
    var reasoning: Int
    var cached: Int
}

/// One captured native tool call, inside a `tool_batch` payload.
struct WireCapturedCall: Encodable {
    var id: String
    var name: String
    var arguments: String
}
