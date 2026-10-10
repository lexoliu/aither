// Wire schema -> DynamicGenerationSchema / GenerationSchema conversion.
//
// The Rust side has already rejected every JSON Schema construct the
// framework cannot express; what arrives here is a normalized tree of
// objects, primitives, arrays, enumerations, `anyOf` choices, named
// references and (26.4+) explicit null.
import Foundation
import FoundationModels

#if AITHER_SCRIPTED
@_cdecl("aither_apple_test_schema")
public func aitherAppleTestSchema(_ bytes: UnsafePointer<UInt8>, _ length: Int) -> Int32 {
    do {
        let wire = try JSONDecoder().decode(WireSchemaRoot.self, from: Data(bytes: bytes, count: length))
        _ = try buildGenerationSchema(wire)
        return 0
    } catch {
        return 1
    }
}
#endif

@available(macOS 26.0, iOS 26.0, *)
func buildGenerationSchema(_ root: WireSchemaRoot) throws -> GenerationSchema {
    var dependencies: [DynamicGenerationSchema] = []
    for (name, def) in root.defs.sorted(by: { $0.key < $1.key }) {
        let dynamic = try buildDynamic(def, name: name)
        switch def {
        case .primitive, .array, .null, .reference:
            dependencies.append(DynamicGenerationSchema(name: name, anyOf: [dynamic]))
        default:
            dependencies.append(dynamic)
        }
    }
    let rootSchema = try buildDynamic(root.schema, name: "root")
    return try GenerationSchema(root: rootSchema, dependencies: dependencies)
}

@available(macOS 26.0, iOS 26.0, *)
private func buildDynamic(_ schema: WireSchema, name: String) throws -> DynamicGenerationSchema {
    switch schema {
    case .object(let name, let description, let properties):
        let props = try properties.map { property in
            try DynamicGenerationSchema.Property(
                name: property.name,
                description: property.description,
                schema: buildDynamic(property.schema, name: property.name),
                isOptional: property.optional
            )
        }
        return DynamicGenerationSchema(
            name: name,
            description: description,
            properties: props
        )
    case .anyOf(let name, let description, let choices):
        if choices.isEmpty {
            throw BridgeError(message: "anyOf schema '\(name)' has no choices")
        }
        let converted = try choices.map { try buildDynamic($0, name: $0.name) }
        return DynamicGenerationSchema(
            name: name,
            description: description,
            anyOf: converted
        )
    case .enumeration(let name, let description, let values):
        if values.isEmpty {
            throw BridgeError(message: "enum schema '\(name)' has no values")
        }
        return DynamicGenerationSchema(
            name: name,
            description: description,
            anyOf: values
        )
    case .primitive(let name, let description, let kind, let guides):
        switch kind {
        case .string:
            var stringGuides: [GenerationGuide<String>] = []
            if let pattern = guides.pattern {
                stringGuides.append(.pattern(try Regex(pattern)))
            }
            return DynamicGenerationSchema(type: String.self, guides: stringGuides)
        case .integer:
            var intGuides: [GenerationGuide<Int>] = []
            if let minimum = guides.integerMinimum {
                intGuides.append(.minimum(minimum))
            }
            if let maximum = guides.integerMaximum {
                intGuides.append(.maximum(maximum))
            }
            return DynamicGenerationSchema(type: Int.self, guides: intGuides)
        case .number:
            var numberGuides: [GenerationGuide<Double>] = []
            if let minimum = guides.minimum {
                numberGuides.append(.minimum(minimum))
            }
            if let maximum = guides.maximum {
                numberGuides.append(.maximum(maximum))
            }
            return DynamicGenerationSchema(type: Double.self, guides: numberGuides)
        case .boolean:
            return DynamicGenerationSchema(type: Bool.self)
        }
    case .array(let name, let description, let items, let minItems, let maxItems):
        let itemsSchema = try buildDynamic(items, name: "\(name)Element")
        return DynamicGenerationSchema(
            arrayOf: itemsSchema,
            minimumElements: minItems,
            maximumElements: maxItems
        )
    case .reference(let name):
        return DynamicGenerationSchema(referenceTo: name)
    case .null:
        #if AITHER_SDK_26_4 || AITHER_SDK_27
            if #available(macOS 26.4, iOS 26.4, *) {
                return DynamicGenerationSchema.null
            }
            throw BridgeError(message: "explicit null schemas require macOS/iOS 26.4")
        #else
            throw BridgeError(message: "explicit null schemas require SDK 26.4")
        #endif
    }
}
