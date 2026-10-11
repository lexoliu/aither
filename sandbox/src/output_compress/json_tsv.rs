//! Structure transform stage: JSON table to TSV.
//!
//! This stage is lossy -- the TSV form drops the distinction between strings,
//! numbers, booleans and null, and flattens how arrays nest -- so the
//! orchestrator only applies it when it can store the original JSON and name
//! it in the text the model reads.

use std::collections::{HashMap, HashSet};

use serde_json::Value;
use tracing::debug;

/// Converts `text` to TSV when it is a JSON array of records or a single JSON
/// object, and when the TSV form is actually smaller.
///
/// Returns `None` for anything else: scalars, empty arrays, non-JSON text, or
/// a table whose TSV encoding would not save anything.
#[must_use]
pub fn to_tsv(text: &str) -> Option<String> {
    let trimmed = text.trim();
    let value: Value = serde_json::from_str(trimmed).ok()?;

    let rows = match &value {
        Value::Array(array) if !array.is_empty() => array
            .iter()
            .map(|element| flatten_value(element, ""))
            .collect::<Vec<_>>(),
        Value::Object(_) => vec![flatten_value(&value, "")],
        // Scalars and empty arrays gain nothing from a table.
        _ => return None,
    };

    if rows.is_empty() {
        return None;
    }

    // Column headers in first-seen order, so the table keeps the document's
    // field order rather than an arbitrary one.
    let mut columns: Vec<String> = Vec::new();
    let mut seen = HashSet::new();
    for row in &rows {
        for (key, _) in row {
            if seen.insert(key.clone()) {
                columns.push(key.clone());
            }
        }
    }

    if columns.is_empty() {
        return None;
    }

    let mut tsv = String::new();
    push_row(&mut tsv, columns.iter().map(String::as_str));
    for row in &rows {
        let cells: HashMap<&str, &str> = row
            .iter()
            .map(|(key, value)| (key.as_str(), value.as_str()))
            .collect();
        push_row(
            &mut tsv,
            columns
                .iter()
                .map(|column| cells.get(column.as_str()).copied().unwrap_or_default()),
        );
    }

    if tsv.len() >= trimmed.len() {
        return None;
    }
    debug!(
        json_len = trimmed.len(),
        tsv_len = tsv.len(),
        "compressed JSON to TSV"
    );
    Some(tsv)
}

/// Appends one tab-separated row, escaping each field.
fn push_row<'a>(tsv: &mut String, fields: impl Iterator<Item = &'a str>) {
    for (index, field) in fields.enumerate() {
        if index > 0 {
            tsv.push('\t');
        }
        tsv.push_str(&escape_tsv_field(field));
    }
    tsv.push('\n');
}

/// Flattens a JSON value into `(dotted_key, string_value)` pairs.
fn flatten_value(value: &Value, prefix: &str) -> Vec<(String, String)> {
    let mut result = Vec::new();
    match value {
        Value::Object(map) => {
            for (key, child) in map {
                let full_key = if prefix.is_empty() {
                    key.clone()
                } else {
                    let mut joined = String::with_capacity(prefix.len() + 1 + key.len());
                    joined.push_str(prefix);
                    joined.push('.');
                    joined.push_str(key);
                    joined
                };
                result.extend(flatten_value(child, &full_key));
            }
        }
        Value::Array(_) => {
            // Nested arrays keep their JSON encoding: there is no column
            // layout that would survive a ragged shape.
            let encoded = serde_json::to_string(value)
                .expect("re-serializing a parsed JSON value cannot fail");
            result.push((prefix.to_owned(), encoded));
        }
        Value::String(text) => result.push((prefix.to_owned(), text.clone())),
        Value::Number(number) => result.push((prefix.to_owned(), number.to_string())),
        Value::Bool(flag) => result.push((prefix.to_owned(), flag.to_string())),
        Value::Null => result.push((prefix.to_owned(), String::new())),
    }
    result
}

/// Escapes a TSV field: the separators become spaces so the table stays
/// rectangular.
fn escape_tsv_field(field: &str) -> String {
    field.replace(['\t', '\n', '\r'], " ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn converts_an_array_of_records() {
        let json = r#"[
            {"name": "Alice", "age": 30, "address": {"city": "NYC", "zip": "10001"}},
            {"name": "Bob", "age": 25, "address": {"city": "LA", "zip": "90001"}}
        ]"#;
        let tsv = to_tsv(json).expect("array of records converts");
        let header = tsv.lines().next().expect("header row");
        for column in ["name", "age", "address.city", "address.zip"] {
            assert!(header.contains(column), "header missing {column}: {header}");
        }
        for cell in ["Alice", "Bob", "NYC", "LA"] {
            assert!(tsv.contains(cell), "missing {cell}: {tsv}");
        }
    }

    #[test]
    fn flattens_a_single_object() {
        let json = r#"{"name": "Alice Example Jones", "nested": {"x": 1}}"#;
        let tsv = to_tsv(json).expect("single object converts");
        assert!(tsv.contains("name\tnested.x"), "{tsv}");
    }

    #[test]
    fn rejects_scalars() {
        assert!(to_tsv("42").is_none());
        assert!(to_tsv("\"hello\"").is_none());
        assert!(to_tsv("true").is_none());
        assert!(to_tsv("[]").is_none());
    }

    #[test]
    fn rejects_non_json() {
        assert!(to_tsv("error: could not compile `aither`").is_none());
    }

    #[test]
    fn rejects_a_table_that_would_grow() {
        // Records with disjoint keys make a wide, mostly empty table: every
        // row pays a tab for every column.
        let records: Vec<String> = (0..20)
            .map(|index| format!(r#"{{"k{index}":{index}}}"#))
            .collect();
        let json = format!("[{}]", records.join(","));
        assert!(to_tsv(&json).is_none(), "{json}");
    }

    #[test]
    fn flattens_nested_objects_with_dotted_keys() {
        let value: Value =
            serde_json::from_str(r#"{"a": {"b": {"c": 1}}, "d": "hello"}"#).expect("valid json");
        let flat: HashMap<String, String> = flatten_value(&value, "").into_iter().collect();
        assert_eq!(flat.get("a.b.c").map(String::as_str), Some("1"));
        assert_eq!(flat.get("d").map(String::as_str), Some("hello"));
    }

    #[test]
    fn escapes_separators_in_fields() {
        assert_eq!(escape_tsv_field("hello\tworld"), "hello world");
        assert_eq!(escape_tsv_field("line1\nline2"), "line1 line2");
    }
}
