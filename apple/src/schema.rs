//! `schemars` JSON Schema → wire schema conversion.
//!
//! `DynamicGenerationSchema` is far narrower than JSON Schema: objects with
//! named properties, string/integer/number/boolean primitives, arrays,
//! string enumerations, `anyOf` unions, named references and (26.4+) explicit
//! null. Anything outside that set is rejected with the schema path of the
//! offending node instead of being silently dropped or weakened into a
//! prompt hint.
//!
//! Annotation-only keywords (`title`, `description`, `default`, `examples`,
//! `deprecated`, `readOnly`, `writeOnly`, `$comment`, `$schema`, `format`)
//! carry no constraint and pass through or are ignored. `format` is an
//! annotation in the 2020-12 default vocabulary, not a validator, so it does
//! not fall under the reject-unsupported rule.

use std::collections::BTreeMap;

use serde_json::Value;

use crate::error::AppleError;
use crate::wire::{WireGuides, WireProperty, WireSchema, WireSchemaRoot};

/// Keywords that constrain values but have no `DynamicGenerationSchema`
/// representation. Presence of any of these rejects the schema.
const UNSUPPORTED_KEYWORDS: &[&str] = &[
    "exclusiveMinimum",
    "exclusiveMaximum",
    "multipleOf",
    "minLength",
    "maxLength",
    "minProperties",
    "maxProperties",
    "uniqueItems",
    "contains",
    "minContains",
    "maxContains",
    "patternProperties",
    "propertyNames",
    "dependentRequired",
    "dependentSchemas",
    "dependencies",
    "if",
    "then",
    "else",
    "not",
    "unevaluatedProperties",
    "unevaluatedItems",
    "prefixItems",
    "contentEncoding",
    "contentMediaType",
    "contentSchema",
];

/// Annotation keywords that carry no constraint.
const ANNOTATION_KEYWORDS: &[&str] = &[
    "title",
    "description",
    "default",
    "examples",
    "deprecated",
    "readOnly",
    "writeOnly",
    "$comment",
    "$schema",
    "$id",
    "format",
];

fn unsupported(path: &str, reason: impl Into<String>) -> AppleError {
    AppleError::UnsupportedSchema {
        path: path.to_string(),
        reason: reason.into(),
    }
}

/// Converts a schemars schema to the wire schema the Swift bridge builds a
/// `GenerationSchema` from.
///
/// # Errors
///
/// [`AppleError::UnsupportedSchema`] for constructs the framework cannot
/// express.
pub fn to_wire_schema(
    schema: &schemars::Schema,
    root_name: &str,
) -> Result<WireSchemaRoot, AppleError> {
    value_to_wire_schema(schema.as_value(), root_name)
}

/// Resolve reference identities using JSON pointers, including escaped names
/// and nested definitions. Each reference has one deterministic native name.
pub fn value_to_wire_schema(value: &Value, root_name: &str) -> Result<WireSchemaRoot, AppleError> {
    let mut normalized = value.clone();
    let mut references = BTreeMap::new();
    normalize_refs(&mut normalized, value, &mut references, "$")?;
    let schema = convert(&normalized, root_name, "$")?;
    let mut defs = BTreeMap::new();
    // Resolving a referenced definition may discover more references.
    loop {
        let pending = references
            .iter()
            .find(|(pointer, _)| !defs.contains_key(*pointer))
            .map(|(pointer, name)| (pointer.clone(), name.clone()));
        let Some((pointer, name)) = pending else {
            break;
        };
        let mut definition = value.pointer(&pointer).expect("validated pointer").clone();
        normalize_refs(&mut definition, value, &mut references, &pointer)?;
        defs.insert(pointer.clone(), convert(&definition, &name, &pointer)?);
    }
    let defs = defs
        .into_iter()
        .map(|(pointer, schema)| (references[&pointer].clone(), schema))
        .collect();
    Ok(WireSchemaRoot { schema, defs })
}

fn normalize_refs(
    value: &mut Value,
    root: &Value,
    references: &mut BTreeMap<String, String>,
    path: &str,
) -> Result<(), AppleError> {
    match value {
        Value::Object(map) => {
            if map.contains_key("$id") {
                return Err(unsupported(
                    path,
                    "$id changes reference scope and is not supported",
                ));
            }
            if let Some(reference) = map.get_mut("$ref") {
                let target = reference
                    .as_str()
                    .ok_or_else(|| unsupported(path, "non-string $ref"))?;
                let pointer = target
                    .strip_prefix('#')
                    .filter(|p| p.is_empty() || p.starts_with('/'))
                    .ok_or_else(|| {
                        unsupported(path, "only document-local JSON pointers are supported")
                    })?;
                if root.pointer(pointer).is_none() {
                    return Err(unsupported(
                        path,
                        format!("unresolved reference '{target}'"),
                    ));
                }
                let next = format!("definition{}", references.len());
                let name = references.entry(pointer.to_string()).or_insert(next);
                *reference = Value::String(name.clone());
            }
            for (key, child) in map {
                // Unused definitions do not constrain the instance.
                if key != "$defs" && key != "definitions" {
                    normalize_refs(child, root, references, &format!("{path}/{key}"))?;
                }
            }
        }
        Value::Array(items) => {
            for (index, child) in items.iter_mut().enumerate() {
                normalize_refs(child, root, references, &format!("{path}/{index}"))?;
            }
        }
        _ => {}
    }
    Ok(())
}

fn description_of(map: &serde_json::Map<String, Value>) -> Option<String> {
    map.get("description")
        .and_then(Value::as_str)
        .map(String::from)
}

fn convert_property(
    name: &str,
    value: &Value,
    required: bool,
    path: &str,
) -> Result<WireProperty, AppleError> {
    Ok(WireProperty {
        name: name.to_string(),
        description: value.as_object().and_then(description_of),
        schema: convert(value, path, path)?,
        optional: !required,
    })
}

/// Rejects unsupported keywords present on a schema object.
fn check_keywords(map: &serde_json::Map<String, Value>, path: &str) -> Result<(), AppleError> {
    for key in UNSUPPORTED_KEYWORDS {
        if map.contains_key(*key) {
            return Err(unsupported(
                path,
                format!("'{key}' has no DynamicGenerationSchema equivalent"),
            ));
        }
    }
    for (key, value) in map {
        let known = matches!(
            key.as_str(),
            "type"
                | "properties"
                | "required"
                | "additionalProperties"
                | "items"
                | "enum"
                | "const"
                | "anyOf"
                | "oneOf"
                | "allOf"
                | "$ref"
                | "$defs"
                | "definitions"
                | "minimum"
                | "maximum"
                | "pattern"
                | "minItems"
                | "maxItems"
        ) || ANNOTATION_KEYWORDS.contains(&key.as_str());
        if !known {
            return Err(unsupported(path, format!("unrecognized keyword '{key}'")));
        }
        check_shape(key, value, path)?;
    }
    Ok(())
}

fn check_shape(key: &str, value: &Value, path: &str) -> Result<(), AppleError> {
    let (valid, expected) = match key {
        "properties" => (value.is_object(), "an object"),
        "anyOf" | "oneOf" | "allOf" => (
            value.as_array().is_some_and(|items| !items.is_empty()),
            "a nonempty array of schemas",
        ),
        "minimum" | "maximum" => (value.as_f64().is_some(), "a number"),
        "pattern" => (value.is_string(), "a string"),
        "minItems" | "maxItems" => {
            return array_count(value, &format!("{path}/{key}")).map(|_| ());
        }
        "type" => {
            let valid_type = |value: &Value| {
                matches!(
                    value.as_str(),
                    Some("null" | "boolean" | "object" | "array" | "number" | "string" | "integer")
                )
            };
            let valid = valid_type(value)
                || value.as_array().is_some_and(|items| {
                    !items.is_empty()
                        && items
                            .iter()
                            .enumerate()
                            .all(|(index, item)| valid_type(item) && !items[..index].contains(item))
                });
            (
                valid,
                "a type name or nonempty array of distinct type names",
            )
        }
        _ => return Ok(()),
    };
    if valid {
        Ok(())
    } else {
        Err(unsupported(
            &format!("{path}/{key}"),
            format!("'{key}' must be {expected}"),
        ))
    }
}

fn reject_siblings(
    map: &serde_json::Map<String, Value>,
    allowed: &[&str],
    path: &str,
) -> Result<(), AppleError> {
    for key in map.keys() {
        if !allowed.contains(&key.as_str())
            && !ANNOTATION_KEYWORDS.contains(&key.as_str())
            && key != "$defs"
            && key != "definitions"
        {
            return Err(unsupported(
                path,
                format!("conjoined '{key}' cannot be represented"),
            ));
        }
    }
    Ok(())
}

fn reject_string_choice_siblings(
    map: &serde_json::Map<String, Value>,
    key: &str,
    path: &str,
) -> Result<(), AppleError> {
    reject_siblings(map, &[key, "type"], path)?;
    if map
        .get("type")
        .is_some_and(|t| t.as_str() != Some("string"))
    {
        return Err(unsupported(
            path,
            "string choices conflict with declared type",
        ));
    }
    Ok(())
}

fn convert(value: &Value, name: &str, path: &str) -> Result<WireSchema, AppleError> {
    let Value::Object(map) = value else {
        return Err(unsupported(path, "boolean schemas are not supported"));
    };
    check_keywords(map, path)?;
    if map.contains_key("anyOf") && map.contains_key("oneOf") {
        return Err(unsupported(
            path,
            "conjoined anyOf and oneOf cannot be represented",
        ));
    }

    if let Some(reference) = map.get("$ref") {
        let Some(target) = reference.as_str() else {
            return Err(unsupported(path, "non-string $ref"));
        };
        reject_siblings(map, &["$ref"], path)?;
        return Ok(WireSchema::Reference {
            name: target.to_string(),
        });
    }

    let description = description_of(map);

    if map.contains_key("enum") || map.contains_key("const") {
        return convert_choice(map, name, path, description);
    }

    if let Some(Value::Array(types)) = map.get("type") {
        let mut choices = Vec::new();
        for (index, ty) in types.iter().enumerate() {
            let mut branch = map.clone();
            branch.insert("type".into(), ty.clone());
            choices.push(convert(
                &Value::Object(branch),
                &format!("{name}Type{index}"),
                path,
            )?);
        }
        if choices.is_empty() {
            return Err(unsupported(path, "empty type union"));
        }
        return Ok(WireSchema::AnyOf {
            name: name.into(),
            description,
            choices,
        });
    }

    // oneOf can become anyOf only when branch types prove disjoint.
    if let Some(Value::Array(choices)) = map.get("oneOf") {
        let types: Option<Vec<_>> = choices
            .iter()
            .map(|c| c.get("type").and_then(Value::as_str))
            .collect();
        let Some(types) = types else {
            return Err(unsupported(path, "oneOf exclusivity cannot be proven"));
        };
        for (index, ty) in types.iter().enumerate() {
            if types[..index].contains(ty)
                || (*ty == "number" && types.contains(&"integer"))
                || (*ty == "integer" && types.contains(&"number"))
            {
                return Err(unsupported(path, "oneOf alternatives overlap"));
            }
        }
    }
    // anyOf / oneOf unions.
    let union = map.get("anyOf").or_else(|| map.get("oneOf"));
    if let Some(Value::Array(choices)) = union {
        reject_siblings(map, &["anyOf", "oneOf"], path)?;
        if choices.is_empty() {
            return Err(unsupported(path, "empty union"));
        }
        let mut converted = Vec::with_capacity(choices.len());
        for (index, choice) in choices.iter().enumerate() {
            converted.push(convert(
                choice,
                &format!("{name}Choice{index}"),
                &format!("{path}/anyOf/{index}"),
            )?);
        }
        return Ok(WireSchema::AnyOf {
            name: name.to_string(),
            description,
            choices: converted,
        });
    }
    if let Some(Value::Array(all_of)) = map.get("allOf") {
        reject_siblings(map, &["allOf"], path)?;
        if all_of.len() == 1 {
            return convert(&all_of[0], name, path);
        }
        return Err(unsupported(
            path,
            "allOf with more than one branch cannot be merged",
        ));
    }

    convert_type(map, name, path, description)
}

fn convert_choice(
    map: &serde_json::Map<String, Value>,
    name: &str,
    path: &str,
    description: Option<String>,
) -> Result<WireSchema, AppleError> {
    // enum / const become string enumerations; the framework only supports
    // string choice lists.
    if let Some(values) = map.get("enum") {
        reject_string_choice_siblings(map, "enum", path)?;
        let Value::Array(items) = values else {
            return Err(unsupported(path, "non-array enum"));
        };
        let mut strings = Vec::with_capacity(items.len());
        for item in items {
            match item.as_str() {
                Some(s) => strings.push(s.to_string()),
                None => {
                    return Err(unsupported(
                        path,
                        "enum values must be strings for guided generation",
                    ));
                }
            }
        }
        if strings.is_empty() {
            return Err(unsupported(path, "empty enum"));
        }
        return Ok(WireSchema::Enumeration {
            name: name.to_string(),
            description,
            values: strings,
        });
    }
    if let Some(constant) = map.get("const") {
        reject_string_choice_siblings(map, "const", path)?;
        let Some(text) = constant.as_str() else {
            return Err(unsupported(
                path,
                "only string const values are representable",
            ));
        };
        return Ok(WireSchema::Enumeration {
            name: name.to_string(),
            description,
            values: vec![text.to_string()],
        });
    }

    unreachable!("enum or const checked")
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "rounded integral float is checked against the exact i64 range before casting"
)]
fn integer_bound(value: &Value, path: &str, lower: bool) -> Result<i64, AppleError> {
    if let Some(integer) = value.as_i64() {
        return Ok(integer);
    }
    if value.is_u64() {
        return Err(unsupported(
            path,
            "integer bound is outside the native Int range",
        ));
    }
    let number = value
        .as_f64()
        .ok_or_else(|| unsupported(path, "bound must be numeric"))?;
    let rounded = if lower { number.ceil() } else { number.floor() };
    // These powers of two are exactly representable; the upper endpoint is exclusive.
    if !(-9_223_372_036_854_775_808.0..9_223_372_036_854_775_808.0).contains(&rounded) {
        return Err(unsupported(
            path,
            "integer bound is outside the native Int range",
        ));
    }
    Ok(rounded as i64)
}

fn integer_guides(
    map: &serde_json::Map<String, Value>,
    path: &str,
) -> Result<WireGuides, AppleError> {
    let integer_minimum = map
        .get("minimum")
        .map(|v| integer_bound(v, &format!("{path}/minimum"), true))
        .transpose()?;
    let integer_maximum = map
        .get("maximum")
        .map(|v| integer_bound(v, &format!("{path}/maximum"), false))
        .transpose()?;
    if integer_minimum
        .zip(integer_maximum)
        .is_some_and(|(min, max)| min > max)
    {
        return Err(unsupported(path, "integer bounds describe an empty range"));
    }
    Ok(WireGuides {
        integer_minimum,
        integer_maximum,
        ..WireGuides::default()
    })
}

fn array_count(value: &Value, path: &str) -> Result<usize, AppleError> {
    if value.as_f64().is_some_and(|number| number.fract() != 0.0) {
        return Err(unsupported(path, "array count must be an integer"));
    }
    // Keep exact JSON integers intact; the shared bound decoder checks Swift Int
    // representability before converting an integral floating-point value.
    let count = integer_bound(value, path, true)?;
    usize::try_from(count)
        .map_err(|_| unsupported(path, "array count must be nonnegative and fit usize"))
}

fn convert_type(
    map: &serde_json::Map<String, Value>,
    name: &str,
    path: &str,
    description: Option<String>,
) -> Result<WireSchema, AppleError> {
    let type_name = map.get("type").and_then(Value::as_str);
    let has_properties = map.contains_key("properties") || map.contains_key("required");
    match (type_name, has_properties) {
        (Some("object"), _) | (None, true) => convert_object(map, name, path, description),

        (Some("string"), false) => Ok(WireSchema::Primitive {
            name: name.to_string(),
            description,
            kind: "string",
            guides: WireGuides {
                pattern: map.get("pattern").and_then(Value::as_str).map(String::from),
                ..WireGuides::default()
            },
        }),
        (Some("integer"), false) => Ok(WireSchema::Primitive {
            name: name.to_string(),
            description,
            kind: "integer",
            guides: integer_guides(map, path)?,
        }),
        (Some("number"), false) => Ok(WireSchema::Primitive {
            name: name.to_string(),
            description,
            kind: "number",
            guides: WireGuides {
                minimum: map.get("minimum").and_then(Value::as_f64),
                maximum: map.get("maximum").and_then(Value::as_f64),
                ..WireGuides::default()
            },
        }),
        (Some("boolean"), false) => Ok(WireSchema::Primitive {
            name: name.to_string(),
            description,
            kind: "boolean",
            guides: WireGuides::default(),
        }),
        (Some("null"), false) => Ok(WireSchema::Null),
        (Some("array"), false) => {
            let items = map.get("items").unwrap_or(&Value::Null);
            let count = |key: &str| -> Result<Option<usize>, AppleError> {
                map.get(key)
                    .map(|value| array_count(value, &format!("{path}/{key}")))
                    .transpose()
            };
            let min_items = count("minItems")?;
            let max_items = count("maxItems")?;
            if min_items.zip(max_items).is_some_and(|(min, max)| min > max) {
                return Err(unsupported(
                    &format!("{path}/maxItems"),
                    "maxItems is less than minItems",
                ));
            }
            Ok(WireSchema::Array {
                name: name.to_string(),
                description,
                items: Box::new(convert(
                    items,
                    &format!("{name}Element"),
                    &format!("{path}/items"),
                )?),
                min_items,
                max_items,
            })
        }
        (Some(other), _) => Err(unsupported(
            path,
            format!("type '{other}' is not supported"),
        )),
        (None, false) => Err(unsupported(
            path,
            "schema has no type, properties, enum, or union to convert",
        )),
    }
}
fn convert_object(
    map: &serde_json::Map<String, Value>,
    name: &str,
    path: &str,
    description: Option<String>,
) -> Result<WireSchema, AppleError> {
    // additionalProperties as a schema is a map, which the framework
    // has no type for.
    match map.get("additionalProperties") {
        None | Some(Value::Bool(_)) => {}
        Some(Value::Object(_)) => {
            return Err(unsupported(
                path,
                "map-typed additionalProperties is not supported",
            ));
        }
        Some(_) => return Err(unsupported(path, "invalid additionalProperties")),
    }
    let props = map.get("properties").map(|value| {
        value
            .as_object()
            .expect("check_shape validated properties as an object")
    });
    let mut required = Vec::new();
    if let Some(value) = map.get("required") {
        let items = value
            .as_array()
            .ok_or_else(|| unsupported(&format!("{path}/required"), "required must be an array"))?;
        for (index, item) in items.iter().enumerate() {
            let key = item.as_str().ok_or_else(|| {
                unsupported(
                    &format!("{path}/required/{index}"),
                    "required entries must be strings",
                )
            })?;
            if required.contains(&key) {
                return Err(unsupported(
                    &format!("{path}/required/{index}"),
                    "duplicate required key",
                ));
            }
            if !props.is_some_and(|props| props.contains_key(key)) {
                return Err(unsupported(
                    &format!("{path}/required/{index}"),
                    format!("required key '{key}' has no declared property"),
                ));
            }
            required.push(key);
        }
    }
    let mut properties = Vec::new();
    if let Some(props) = props {
        for (prop_name, prop_schema) in props {
            properties.push(convert_property(
                prop_name,
                prop_schema,
                required.contains(&prop_name.as_str()),
                &format!("{path}/properties/{prop_name}"),
            )?);
        }
    }
    Ok(WireSchema::Object {
        name: name.to_string(),
        description,
        properties,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unsupported_reason(value: &Value) -> String {
        match value_to_wire_schema(value, "root") {
            Err(AppleError::UnsupportedSchema { path, reason }) => format!("{path}: {reason}"),
            other => panic!("expected UnsupportedSchema, got {other:?}"),
        }
    }

    #[test]
    fn malformed_constraint_shapes_are_rejected_at_the_keyword() {
        use serde_json::json;
        for (schema, keyword) in [
            (json!({"type":"object","properties":"x"}), "properties"),
            (json!({"type":"object","properties":null}), "properties"),
            (json!({"type":"string","anyOf":{}}), "anyOf"),
            (json!({"type":"string","oneOf":true}), "oneOf"),
            (json!({"type":"string","allOf":null}), "allOf"),
            (json!({"type":"number","minimum":"1"}), "minimum"),
            (json!({"type":"number","maximum":null}), "maximum"),
            (json!({"type":"string","pattern":42}), "pattern"),
            (
                json!({"type":"array","items":{"type":"string"},"minItems":-1}),
                "minItems",
            ),
            (
                json!({"type":"array","items":{"type":"string"},"maxItems":1.5}),
                "maxItems",
            ),
            (json!({"type":42,"properties":{}}), "type"),
            (json!({"type":["object",null],"properties":{}}), "type"),
            (json!({"type":[["string"]]}), "type"),
            (json!({"type":[]}), "type"),
            (json!({"type":["string","string"]}), "type"),
            (json!({"type":"unknown"}), "type"),
        ] {
            let error = unsupported_reason(&schema);
            assert!(
                error.starts_with(&format!("$/{keyword}:")),
                "{schema}: {error}"
            );
        }
        assert!(value_to_wire_schema(&json!({"type":"object"}), "root").is_ok());
        assert!(
            value_to_wire_schema(
                &json!({"type":"array","items":{"type":"string"},"minItems":0,"maxItems":2}),
                "root"
            )
            .is_ok()
        );
    }

    #[test]
    fn array_counts_normalize_mathematical_integers_with_native_bounds() {
        use serde_json::json;
        for (minimum, maximum, expected_min, expected_max) in [
            (json!(1.0), json!(2.0), 1, 2),
            (json!(1_i64), json!(2_u64), 1, 2),
            (json!(-0.0), json!(i64::MAX), 0, i64::MAX),
        ] {
            let root = value_to_wire_schema(&json!({"type":"array","items":{"type":"string"},"minItems":minimum,"maxItems":maximum}), "root").unwrap();
            let wire = serde_json::to_value(root.schema).unwrap();
            assert_eq!(wire["min_items"].as_i64(), Some(expected_min));
            assert_eq!(wire["max_items"].as_i64(), Some(expected_max));
        }
        for value in [
            json!(1.5),
            json!(-1),
            json!(-1.0),
            json!(u64::MAX),
            json!(9_223_372_036_854_775_808_u64),
            json!(9_223_372_036_854_775_808.0),
            json!(1e100),
        ] {
            for keyword in ["minItems", "maxItems"] {
                let error = unsupported_reason(
                    &json!({"type":"array","items":{"type":"string"},keyword:value}),
                );
                assert!(error.starts_with(&format!("$/{keyword}:")), "{error}");
            }
        }
        let error = unsupported_reason(
            &json!({"type":"array","items":{"type":"string"},"minItems":3,"maxItems":2}),
        );
        assert!(error.starts_with("$/maxItems:"), "{error}");
    }

    #[test]
    fn integer_bounds_preserve_exact_values_and_round_inward() {
        for (minimum, maximum, expected_min, expected_max) in [
            (serde_json::json!(1.5), serde_json::json!(5.9), 2, 5),
            (serde_json::json!(-5.9), serde_json::json!(-1.5), -5, -2),
            (
                serde_json::json!(i64::MIN),
                serde_json::json!(i64::MAX),
                i64::MIN,
                i64::MAX,
            ),
        ] {
            let root = value_to_wire_schema(
                &serde_json::json!({"type":"integer", "minimum":minimum,"maximum":maximum}),
                "root",
            )
            .unwrap();
            let WireSchema::Primitive { guides, .. } = root.schema else {
                panic!("expected primitive")
            };
            assert_eq!(guides.integer_minimum, Some(expected_min));
            assert_eq!(guides.integer_maximum, Some(expected_max));
            let wire = serde_json::to_value(guides).unwrap();
            assert_eq!(wire["integer_minimum"].as_i64(), Some(expected_min));
            assert_eq!(wire["integer_maximum"].as_i64(), Some(expected_max));
        }
        for bound in [
            serde_json::json!(1e100),
            serde_json::json!(-1e100),
            serde_json::json!(u64::MAX),
            serde_json::json!(9_223_372_036_854_775_808.0),
            serde_json::json!("invalid"),
        ] {
            for keyword in ["minimum", "maximum"] {
                let error =
                    unsupported_reason(&serde_json::json!({"type":"integer", keyword:bound}));
                assert!(error.starts_with(&format!("$/{keyword}:")), "{error}");
            }
        }
        for (min, max) in [(1.5, 1.9), (5.0, 4.0)] {
            assert!(
                unsupported_reason(
                    &serde_json::json!({"type":"integer","minimum":min,"maximum":max})
                )
                .contains("empty range")
            );
        }
    }

    #[test]
    fn required_entries_must_be_declared_unique_strings() {
        for required in [
            serde_json::json!(["missing"]),
            serde_json::json!("known"),
            serde_json::json!([42]),
            serde_json::json!(["known", "known"]),
        ] {
            let error = unsupported_reason(
                &serde_json::json!({"type":"object","properties":{"known":{"type":"string"}},"required":required}),
            );
            assert!(error.starts_with("$/required"), "{error}");
        }
        assert!(
            unsupported_reason(&serde_json::json!({"type":"object","required":["missing"]}))
                .contains("no declared property")
        );
    }

    #[test]
    fn conjoined_union_keywords_are_rejected() {
        let error = unsupported_reason(
            &serde_json::json!({"anyOf":[{"type":"string"}],"oneOf":[{"type":"number"}]}),
        );
        assert!(error.contains("conjoined anyOf and oneOf"), "{error}");
    }

    #[test]
    fn object_with_required_and_optional_properties() {
        let value = serde_json::json!({
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "full name"},
                "age": {"type": "integer", "minimum": 0, "maximum": 150},
                "nick": {"type": ["string", "null"]}
            },
            "required": ["name"]
        });
        let root = value_to_wire_schema(&value, "person").expect("converts");
        let WireSchema::Object { properties, .. } = root.schema else {
            panic!("expected object")
        };
        assert_eq!(properties.len(), 3);
        let by_name = |n: &str| properties.iter().find(|p| p.name == n).expect(n);
        assert!(!by_name("name").optional);
        assert!(by_name("age").optional);
        assert!(by_name("nick").optional);
        match &by_name("name").schema {
            WireSchema::Primitive { kind: "string", .. } => {}
            other => panic!("expected string, got {other:?}"),
        }
        match &by_name("age").schema {
            WireSchema::Primitive {
                kind: "integer",
                guides,
                ..
            } => {
                assert_eq!(guides.integer_minimum, Some(0));
                assert_eq!(guides.integer_maximum, Some(150));
            }
            other => panic!("expected integer, got {other:?}"),
        }
    }

    #[test]
    fn string_enum_and_refs() {
        let value = serde_json::json!({
            "$defs": {
                "Color": {"type": "string", "enum": ["red", "green"]}
            },
            "type": "object",
            "properties": {
                "color": {"$ref": "#/$defs/Color"}
            },
            "required": ["color"]
        });
        let root = value_to_wire_schema(&value, "root").expect("converts");
        assert_eq!(root.defs.len(), 1);
        assert!(
            matches!(&root.defs["definition0"], WireSchema::Enumeration { values, .. } if values.len() == 2)
        );
        let WireSchema::Object { properties, .. } = root.schema else {
            panic!("expected object")
        };
        assert!(
            matches!(&properties[0].schema, WireSchema::Reference { name } if name == "definition0")
        );
    }

    #[test]
    fn unsupported_constraints_fail_with_path() {
        let err = unsupported_reason(&serde_json::json!({
            "type": "object",
            "properties": {"tag": {"type": "string", "minLength": 2}}
        }));
        assert!(err.contains("properties/tag"), "{err}");
        assert!(err.contains("minLength"), "{err}");

        let err = unsupported_reason(&serde_json::json!({
            "type": "object", "additionalProperties": {"type": "string"}
        }));
        assert!(err.contains("additionalProperties"), "{err}");

        let err = unsupported_reason(&serde_json::json!({
            "type": "integer", "multipleOf": 3
        }));
        assert!(err.contains("multipleOf"), "{err}");
    }

    #[test]
    fn one_of_and_nullable_anyof() {
        let value = serde_json::json!({
            "anyOf": [
                {"type": "integer"},
                {"type": "string"}
            ]
        });
        let root = value_to_wire_schema(&value, "root").expect("converts");
        assert!(matches!(root.schema, WireSchema::AnyOf { .. }));

        // Required nullable fields must still be present, with explicit null.
        let value = serde_json::json!({
            "type": "object",
            "properties": {
                "maybe": {"anyOf": [{"type": "string"}, {"type": "null"}]}
            },
            "required": ["maybe"]
        });
        let root = value_to_wire_schema(&value, "root").expect("converts");
        let WireSchema::Object { properties, .. } = root.schema else {
            panic!("expected object")
        };
        assert!(!properties[0].optional);
        assert!(
            matches!(&properties[0].schema, WireSchema::AnyOf { choices, .. } if choices.len() == 2)
        );
    }

    #[test]
    fn conjunctions_and_overlapping_oneof_are_rejected() {
        for schema in [
            serde_json::json!({"$defs":{"X":{"type":"string"}},"$ref":"#/$defs/X","maxLength":4}),
            serde_json::json!({"enum":["a","b"],"pattern":"^a$"}),
            serde_json::json!({"type":"string","allOf":[{"type":"integer"}]}),
            serde_json::json!({"oneOf":[{"type":"number"},{"type":"integer"}]}),
            serde_json::json!({"oneOf":[{"type":"string"},{"type":"string"}]}),
            serde_json::json!({"$ref":"#/$defs/missing"}),
        ] {
            assert!(value_to_wire_schema(&schema, "Root").is_err(), "{schema}");
        }
    }

    #[test]
    fn nested_escaped_definitions_have_distinct_identities() {
        let schema = serde_json::json!({"$defs": {
            "a/b~c":{"type":"string"},
            "outer":{"$defs":{"a/b~c":{"type":"integer"}},"$ref":"#/$defs/outer/$defs/a~1b~0c"}
        },"type":"object","properties":{
            "a":{"$ref":"#/$defs/a~1b~0c"},"b":{"$ref":"#/$defs/outer"}
        }});
        let wire = value_to_wire_schema(&schema, "Root").unwrap();
        assert_eq!(wire.defs.len(), 3);
        let json = serde_json::to_value(wire).unwrap();
        assert!(json["defs"].is_object());
        assert_eq!(
            json["schema"]["properties"][0]["schema"]["ref"],
            "definition0"
        );
    }
}
