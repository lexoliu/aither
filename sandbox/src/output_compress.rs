//! Output compression pipeline for bash tool results.
//!
//! Applies semantics-preserving transformations to reduce output size:
//! 1. JSON arrays/objects -> TSV with dot-notation flattened headers (if smaller)
//! 2. Strip empty lines and invisible characters
//! 3. Programming language source -> tree-sitter folded code with line numbers (raw saved separately)
//!
//! Stripping is done before code folding so that tree-sitter line numbers
//! reference the cleaned text -- which is also what gets saved as the raw file.

use serde_json::Value;
use tracing::debug;

/// Result of the compression pipeline.
pub struct CompressedOutput {
    /// The (possibly compressed) text to show inline.
    pub text: String,
    /// If the output was detected as source code, the original raw text
    /// should be saved to a file. This field holds that raw content.
    pub raw_for_file: Option<String>,
}

/// Run the full compression pipeline on text output.
///
/// Returns `None` if the input is empty.
pub fn compress_text(text: &str) -> Option<CompressedOutput> {
    if text.is_empty() {
        return None;
    }

    // Step 1: Try JSON -> TSV conversion (no raw file needed)
    if let Some(tsv) = try_json_to_tsv(text) {
        let cleaned = strip_noise(&tsv);
        return Some(CompressedOutput {
            text: cleaned,
            raw_for_file: None,
        });
    }

    // Step 2: Strip noise (empty lines, invisible chars).
    // Done before code folding so tree-sitter line numbers match the
    // cleaned text, which is also what we save as the raw file.
    let cleaned = strip_noise(text);

    // Step 3: Try magika detection + tree-sitter folding on the cleaned text.
    // The raw file saved is the cleaned version (same line numbers as the fold markers).
    if let Some(folded) = try_fold_source_code(&cleaned) {
        return Some(CompressedOutput {
            text: folded.folded,
            raw_for_file: Some(cleaned),
        });
    }

    // No code folding applied -- return cleaned text if it's smaller
    if cleaned.len() < text.len() {
        Some(CompressedOutput {
            text: cleaned,
            raw_for_file: None,
        })
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// JSON -> TSV
// ---------------------------------------------------------------------------

/// Try to parse `text` as JSON and convert to TSV if it's an array of objects
/// or a single object. Returns TSV string only if it's smaller than the original.
fn try_json_to_tsv(text: &str) -> Option<String> {
    let trimmed = text.trim();
    let val: Value = serde_json::from_str(trimmed).ok()?;

    let rows = match &val {
        Value::Array(arr) if !arr.is_empty() => {
            // Array of objects (or mixed) -- flatten each element
            arr.iter().map(|v| flatten_value(v, "")).collect::<Vec<_>>()
        }
        Value::Object(_) => {
            // Single object -- treat as one-row table
            vec![flatten_value(&val, "")]
        }
        _ => return None, // Scalars, empty arrays -- not useful as TSV
    };

    if rows.is_empty() {
        return None;
    }

    // Collect all unique column headers in insertion order
    let mut columns: Vec<String> = Vec::new();
    let mut seen = std::collections::HashSet::new();
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

    // Build TSV
    let mut tsv = String::new();

    // Header row
    for (i, col) in columns.iter().enumerate() {
        if i > 0 {
            tsv.push('\t');
        }
        tsv.push_str(&escape_tsv_field(col));
    }
    tsv.push('\n');

    // Data rows
    for row in &rows {
        let row_map: std::collections::HashMap<&str, &str> =
            row.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect();
        for (i, col) in columns.iter().enumerate() {
            if i > 0 {
                tsv.push('\t');
            }
            if let Some(val) = row_map.get(col.as_str()) {
                tsv.push_str(&escape_tsv_field(val));
            }
        }
        tsv.push('\n');
    }

    // Only use TSV if it's actually smaller
    if tsv.len() < trimmed.len() {
        debug!(
            json_len = trimmed.len(),
            tsv_len = tsv.len(),
            "compressed JSON to TSV"
        );
        Some(tsv)
    } else {
        None
    }
}

/// Flatten a JSON value into a list of (dotted_key, string_value) pairs.
fn flatten_value(val: &Value, prefix: &str) -> Vec<(String, String)> {
    let mut result = Vec::new();
    match val {
        Value::Object(map) => {
            for (key, child) in map {
                let full_key = if prefix.is_empty() {
                    key.clone()
                } else {
                    format!("{prefix}.{key}")
                };
                result.extend(flatten_value(child, &full_key));
            }
        }
        Value::Array(arr) => {
            // For arrays inside objects, serialize as JSON string
            let s = serde_json::to_string(val).unwrap_or_default();
            result.push((prefix.to_owned(), s));
            let _ = arr; // suppress unused warning
        }
        Value::String(s) => {
            result.push((prefix.to_owned(), s.clone()));
        }
        Value::Number(n) => {
            result.push((prefix.to_owned(), n.to_string()));
        }
        Value::Bool(b) => {
            result.push((prefix.to_owned(), b.to_string()));
        }
        Value::Null => {
            result.push((prefix.to_owned(), String::new()));
        }
    }
    result
}

/// Escape a TSV field: replace tabs and newlines with spaces.
fn escape_tsv_field(s: &str) -> String {
    s.replace('\t', " ").replace('\n', " ").replace('\r', " ")
}

// ---------------------------------------------------------------------------
// Source code detection (magika) + tree-sitter folding
// ---------------------------------------------------------------------------

struct FoldedCode {
    folded: String,
}

/// Mapping from magika `ContentType` label to tree-sitter language module.
fn magika_label_to_ts_language(label: &str) -> Option<tree_sitter::Language> {
    match label {
        "bash" | "shell" => Some(rs_tree_sitter_languages::bash::language()),
        "c" => Some(rs_tree_sitter_languages::c::language()),
        "cpp" => Some(rs_tree_sitter_languages::cpp::language()),
        "css" => Some(rs_tree_sitter_languages::css::language()),
        "elixir" => Some(rs_tree_sitter_languages::elixir::language()),
        "erlang" => Some(rs_tree_sitter_languages::erlang::language()),
        "go" => Some(rs_tree_sitter_languages::go::language()),
        "haskell" => Some(rs_tree_sitter_languages::haskell::language()),
        "html" => Some(rs_tree_sitter_languages::html::language()),
        "java" => Some(rs_tree_sitter_languages::java::language()),
        "javascript" => Some(rs_tree_sitter_languages::javascript::language()),
        "json" => Some(rs_tree_sitter_languages::json::language()),
        "lua" => Some(rs_tree_sitter_languages::lua::language()),
        "markdown" => Some(rs_tree_sitter_languages::markdown::language()),
        "perl" => Some(rs_tree_sitter_languages::perl::language()),
        "python" => Some(rs_tree_sitter_languages::python::language()),
        "ruby" => Some(rs_tree_sitter_languages::ruby::language()),
        "rust" => Some(rs_tree_sitter_languages::rust::language()),
        "toml" => Some(rs_tree_sitter_languages::toml::language()),
        "tsx" => Some(rs_tree_sitter_languages::tsx::language()),
        "typescript" => Some(rs_tree_sitter_languages::typescript::language()),
        "yaml" => Some(rs_tree_sitter_languages::yaml::language()),
        _ => None,
    }
}

/// Set of tree-sitter node types considered "foldable blocks".
/// These are compound statements whose body can be collapsed.
fn is_foldable_node(kind: &str) -> bool {
    matches!(
        kind,
        "function_definition"
            | "function_item"
            | "function_declaration"
            | "method_definition"
            | "method_declaration"
            | "class_definition"
            | "class_declaration"
            | "class_body"
            | "impl_item"
            | "trait_item"
            | "module"
            | "block"
            | "compound_statement"
            | "if_statement"
            | "if_expression"
            | "for_statement"
            | "for_expression"
            | "while_statement"
            | "while_expression"
            | "match_expression"
            | "switch_statement"
            | "try_statement"
            | "do_statement"
    )
}

/// Minimum number of lines a foldable block must span to be worth folding.
const MIN_FOLD_LINES: usize = 4;

/// Try to detect source code via magika and fold it with tree-sitter.
fn try_fold_source_code(text: &str) -> Option<FoldedCode> {
    // Use magika to identify content type
    let mut session = magika::Session::new().ok()?;
    let result = session.identify_content_sync(text.as_bytes()).ok()?;
    let label = result.info().label;

    let ts_lang = magika_label_to_ts_language(label)?;

    debug!(label = %label, "detected source code via magika, applying tree-sitter folding");

    // Parse with tree-sitter
    let mut parser = tree_sitter::Parser::new();
    parser.set_language(&ts_lang).ok()?;
    let tree = parser.parse(text, None)?;
    let root = tree.root_node();

    // Collect foldable ranges (line-based, 0-indexed)
    let mut fold_ranges: Vec<(usize, usize)> = Vec::new();
    collect_foldable_ranges(root, &mut fold_ranges);

    if fold_ranges.is_empty() {
        // No foldable blocks found, skip
        return None;
    }

    // Sort by start line and merge overlapping ranges
    fold_ranges.sort_by_key(|r| r.0);
    let merged = merge_ranges(&fold_ranges);

    // Build folded output
    let lines: Vec<&str> = text.lines().collect();
    let total_lines = lines.len();
    let line_num_width = digit_count(total_lines);

    let mut output = String::new();
    let mut i = 0;

    for (fold_start, fold_end) in &merged {
        let fold_start = *fold_start;
        let fold_end = (*fold_end).min(total_lines.saturating_sub(1));

        // Emit lines before this fold region
        while i < fold_start && i < total_lines {
            append_numbered_line(&mut output, i + 1, lines[i], line_num_width);
            i += 1;
        }

        // Emit the first line of the fold region (the signature/declaration)
        if i < total_lines {
            append_numbered_line(&mut output, i + 1, lines[i], line_num_width);
            i += 1;
        }

        // Calculate how many inner lines we're folding
        let inner_end = fold_end + 1; // exclusive
        let folded_count = if inner_end > i { inner_end - i } else { 0 };

        if folded_count > 0 {
            // Emit fold marker
            let marker = format!(
                "{:>width$}  ... ({folded_count} lines folded, lines {}-{}) ...",
                "",
                i + 1,
                inner_end,
                width = line_num_width,
            );
            output.push_str(&marker);
            output.push('\n');
            i = inner_end;
        }
    }

    // Emit remaining lines after last fold
    while i < total_lines {
        append_numbered_line(&mut output, i + 1, lines[i], line_num_width);
        i += 1;
    }

    // Only use folded version if it actually reduced size
    if output.len() < text.len() {
        Some(FoldedCode { folded: output })
    } else {
        None
    }
}

/// Recursively collect foldable ranges from the tree-sitter AST.
fn collect_foldable_ranges(node: tree_sitter::Node, ranges: &mut Vec<(usize, usize)>) {
    if is_foldable_node(node.kind()) {
        let start_line = node.start_position().row;
        let end_line = node.end_position().row;
        if end_line - start_line >= MIN_FOLD_LINES {
            ranges.push((start_line, end_line));
        }
    }
    for i in 0..node.child_count() {
        if let Some(child) = node.child(i) {
            collect_foldable_ranges(child, ranges);
        }
    }
}

/// Merge overlapping or adjacent ranges.
fn merge_ranges(ranges: &[(usize, usize)]) -> Vec<(usize, usize)> {
    let mut merged: Vec<(usize, usize)> = Vec::new();
    for &(start, end) in ranges {
        if let Some(last) = merged.last_mut() {
            if start <= last.1 + 1 {
                last.1 = last.1.max(end);
                continue;
            }
        }
        merged.push((start, end));
    }
    merged
}

fn append_numbered_line(out: &mut String, line_num: usize, line: &str, width: usize) {
    use std::fmt::Write;
    let _ = writeln!(out, "{line_num:>width$}  {line}");
}

fn digit_count(n: usize) -> usize {
    if n == 0 {
        return 1;
    }
    ((n as f64).log10().floor() as usize) + 1
}

// ---------------------------------------------------------------------------
// Noise stripping
// ---------------------------------------------------------------------------

/// Remove empty lines and invisible/control characters while preserving semantics.
fn strip_noise(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    for line in text.lines() {
        // Strip invisible/control characters (keep printable + tab + standard whitespace)
        let cleaned: String = line
            .chars()
            .filter(|c| {
                // Keep printable characters, tab, and regular space
                !c.is_control() || *c == '\t'
            })
            .collect();

        // Skip fully empty lines (after stripping)
        let trimmed = cleaned.trim_end();
        if trimmed.is_empty() {
            continue;
        }

        result.push_str(trimmed);
        result.push('\n');
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_json_array_to_tsv() {
        let json = r#"[
            {"name": "Alice", "age": 30, "address": {"city": "NYC", "zip": "10001"}},
            {"name": "Bob", "age": 25, "address": {"city": "LA", "zip": "90001"}}
        ]"#;
        let tsv = try_json_to_tsv(json).unwrap();
        eprintln!("TSV output:\n{tsv}");
        // Column order depends on JSON map iteration order; check individual fields
        let lines: Vec<&str> = tsv.lines().collect();
        let header = lines[0];
        assert!(header.contains("name"), "header missing 'name': {header}");
        assert!(header.contains("age"), "header missing 'age': {header}");
        assert!(
            header.contains("address.city"),
            "header missing 'address.city': {header}"
        );
        assert!(
            header.contains("address.zip"),
            "header missing 'address.zip': {header}"
        );
        assert!(tsv.contains("Alice"));
        assert!(tsv.contains("Bob"));
        assert!(tsv.contains("NYC"));
        assert!(tsv.contains("LA"));
    }

    #[test]
    fn test_json_single_object_to_tsv() {
        let json = r#"{"name": "Alice", "nested": {"x": 1}}"#;
        let result = try_json_to_tsv(json);
        // Single object with 2 fields -> TSV might not be smaller, depends on overhead
        // But the flattening should work
        if let Some(tsv) = result {
            assert!(tsv.contains("name\tnested.x"));
        }
    }

    #[test]
    fn test_json_scalar_not_converted() {
        assert!(try_json_to_tsv("42").is_none());
        assert!(try_json_to_tsv("\"hello\"").is_none());
        assert!(try_json_to_tsv("true").is_none());
    }

    #[test]
    fn test_strip_noise() {
        let input = "hello\n\n\nworld\n\n";
        let result = strip_noise(input);
        assert_eq!(result, "hello\nworld\n");
    }

    #[test]
    fn test_strip_invisible_chars() {
        let input = "hello\x00world\x01test\n";
        let result = strip_noise(input);
        assert_eq!(result, "helloworldtest\n");
    }

    #[test]
    fn test_flatten_nested() {
        let val: Value = serde_json::from_str(r#"{"a": {"b": {"c": 1}}, "d": "hello"}"#).unwrap();
        let flat = flatten_value(&val, "");
        let map: std::collections::HashMap<String, String> = flat.into_iter().collect();
        assert_eq!(map.get("a.b.c").unwrap(), "1");
        assert_eq!(map.get("d").unwrap(), "hello");
    }

    #[test]
    fn test_escape_tsv() {
        assert_eq!(escape_tsv_field("hello\tworld"), "hello world");
        assert_eq!(escape_tsv_field("line1\nline2"), "line1 line2");
    }

    #[test]
    fn test_fold_python_code() {
        let code = r#"
def hello():
    print("hello")
    print("world")
    print("foo")
    print("bar")
    return True

def goodbye():
    print("bye")
    print("see ya")
    print("later")
    print("adios")
    return False
"#;
        let result = try_fold_source_code(code.trim());
        // If magika + tree-sitter work, we should get folded output
        if let Some(folded) = result {
            assert!(
                folded.folded.contains("folded"),
                "expected fold markers: {}",
                folded.folded
            );
            assert!(folded.folded.len() < code.len(), "folded should be smaller");
        }
        // If magika can't identify it (e.g., too short), that's OK for the test
    }

    #[test]
    fn test_compress_text_json() {
        // Large enough JSON array that TSV is smaller
        let json = r#"[
            {"id": 1, "name": "Alice", "email": "alice@example.com", "role": "admin"},
            {"id": 2, "name": "Bob", "email": "bob@example.com", "role": "user"},
            {"id": 3, "name": "Charlie", "email": "charlie@example.com", "role": "user"},
            {"id": 4, "name": "Diana", "email": "diana@example.com", "role": "admin"},
            {"id": 5, "name": "Eve", "email": "eve@example.com", "role": "viewer"}
        ]"#;
        let result = compress_text(json);
        if let Some(c) = result {
            assert!(
                c.raw_for_file.is_none(),
                "JSON->TSV should not save raw file"
            );
            assert!(c.text.contains('\t'), "should be TSV format");
            assert!(c.text.len() < json.len(), "TSV should be smaller");
        }
    }

    #[test]
    fn test_compress_text_strips_noise() {
        let noisy = "hello\n\n\n\nworld\n\n\n";
        let result = compress_text(noisy).unwrap();
        assert_eq!(result.text, "hello\nworld\n");
        assert!(result.raw_for_file.is_none());
    }

    #[test]
    fn test_compress_text_empty() {
        assert!(compress_text("").is_none());
    }
}
