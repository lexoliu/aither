//! Search strategies for deferred tool lookup.
//!
//! When many tools are registered, the agent can use tool search to reduce
//! context usage by only loading relevant tools on-demand.

use std::collections::HashMap;

use aither_core::llm::tool::ToolDefinition;

use crate::config::SearchStrategy;

/// Search for tools matching a query.
///
/// Returns indices of matching tools sorted by relevance.
pub fn search_tools(
    query: &str,
    tools: &[ToolDefinition],
    strategy: SearchStrategy,
    top_k: usize,
) -> Vec<usize> {
    match strategy {
        SearchStrategy::Bm25 => bm25_search(query, tools, top_k),
        SearchStrategy::Regex => regex_search(query, tools, top_k),
    }
}

/// BM25-based keyword search.
///
/// Scores tools based on term frequency and inverse document frequency.
fn bm25_search(query: &str, tools: &[ToolDefinition], top_k: usize) -> Vec<usize> {
    if tools.is_empty() {
        return Vec::new();
    }

    let query_terms: Vec<&str> = tokenize(query);
    if query_terms.is_empty() {
        return Vec::new();
    }

    // Calculate IDF for each term
    let idf = calculate_idf(&query_terms, tools);

    // BM25 parameters
    const K1: f64 = 1.2;
    const B: f64 = 0.75;

    // Calculate average document length
    let avg_doc_len: f64 =
        tools.iter().map(|t| doc_length(t) as f64).sum::<f64>() / tools.len() as f64;

    // Score each tool
    let mut scores: Vec<(usize, f64)> = tools
        .iter()
        .enumerate()
        .map(|(idx, tool)| {
            let doc_text = format!("{} {}", tool.name(), tool.description());
            let doc_len = doc_length(tool) as f64;
            let term_freqs = term_frequencies(&doc_text);

            let score: f64 = query_terms
                .iter()
                .map(|term| {
                    let tf = *term_freqs.get(*term).unwrap_or(&0) as f64;
                    let term_idf = idf.get(*term).copied().unwrap_or(0.0);

                    // BM25 formula
                    let numerator = tf * (K1 + 1.0);
                    let denominator = tf + K1 * (1.0 - B + B * doc_len / avg_doc_len);

                    term_idf * numerator / denominator
                })
                .sum();

            (idx, score)
        })
        .collect();

    // Sort by score descending
    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Return top-k indices with non-zero scores
    scores
        .into_iter()
        .filter(|(_, score)| *score > 0.0)
        .take(top_k)
        .map(|(idx, _)| idx)
        .collect()
}

/// Regex-based pattern search.
///
/// Matches tools where name or description contains the query pattern.
fn regex_search(query: &str, tools: &[ToolDefinition], top_k: usize) -> Vec<usize> {
    let pattern = match regex::RegexBuilder::new(query)
        .case_insensitive(true)
        .build()
    {
        Ok(p) => p,
        Err(_) => {
            // Fall back to literal search if regex is invalid
            return literal_search(query, tools, top_k);
        }
    };

    let mut matches: Vec<(usize, usize)> = tools
        .iter()
        .enumerate()
        .filter_map(|(idx, tool)| {
            let text = format!("{} {}", tool.name(), tool.description());
            let match_count = pattern.find_iter(&text).count();
            if match_count > 0 {
                Some((idx, match_count))
            } else {
                None
            }
        })
        .collect();

    // Sort by match count descending
    matches.sort_by(|a, b| b.1.cmp(&a.1));

    matches
        .into_iter()
        .take(top_k)
        .map(|(idx, _)| idx)
        .collect()
}

/// Simple literal substring search (fallback).
fn literal_search(query: &str, tools: &[ToolDefinition], top_k: usize) -> Vec<usize> {
    let query_lower = query.to_lowercase();

    let mut matches: Vec<(usize, usize)> = tools
        .iter()
        .enumerate()
        .filter_map(|(idx, tool)| {
            let text = format!("{} {}", tool.name(), tool.description()).to_lowercase();
            let match_count = text.matches(&query_lower).count();
            if match_count > 0 {
                Some((idx, match_count))
            } else {
                None
            }
        })
        .collect();

    matches.sort_by(|a, b| b.1.cmp(&a.1));

    matches
        .into_iter()
        .take(top_k)
        .map(|(idx, _)| idx)
        .collect()
}

/// Tokenize text into lowercase words.
fn tokenize(text: &str) -> Vec<&str> {
    text.split(|c: char| !c.is_alphanumeric() && c != '_')
        .filter(|s| !s.is_empty() && s.len() > 1)
        .collect()
}

/// Calculate term frequencies in a document.
fn term_frequencies(text: &str) -> HashMap<String, usize> {
    let mut freqs = HashMap::new();
    for word in text
        .split(|c: char| !c.is_alphanumeric() && c != '_')
        .filter(|s| !s.is_empty())
    {
        *freqs.entry(word.to_lowercase()).or_insert(0) += 1;
    }
    freqs
}

/// Calculate IDF (inverse document frequency) for terms.
fn calculate_idf(terms: &[&str], tools: &[ToolDefinition]) -> HashMap<String, f64> {
    let n = tools.len() as f64;
    let mut idf = HashMap::new();

    for term in terms {
        let term_lower = term.to_lowercase();
        let doc_count = tools
            .iter()
            .filter(|tool| {
                let text = format!("{} {}", tool.name(), tool.description()).to_lowercase();
                text.contains(&term_lower)
            })
            .count() as f64;

        // IDF formula with smoothing
        let value = ((n - doc_count + 0.5) / (doc_count + 0.5) + 1.0).ln();
        idf.insert(term_lower, value);
    }

    idf
}

/// Calculate document length (number of terms).
fn doc_length(tool: &ToolDefinition) -> usize {
    let text = format!("{} {}", tool.name(), tool.description());
    tokenize(&text).len()
}

#[cfg(test)]
mod tests {
    use super::*;
    use aither_core::llm::tool::Tool;
    use schemars::JsonSchema;
    use serde::Deserialize;
    use std::borrow::Cow;

    #[derive(Debug, JsonSchema, Deserialize)]
    struct EmptyArgs;

    struct MockTool {
        name: &'static str,
        description: &'static str,
    }

    impl Tool for MockTool {
        fn name(&self) -> Cow<'static, str> {
            self.name.into()
        }

        fn description(&self) -> Cow<'static, str> {
            self.description.into()
        }

        type Arguments = EmptyArgs;

        async fn call(&self, _args: Self::Arguments) -> aither_core::Result {
            Ok("ok".into())
        }
    }

    fn make_tool(name: &'static str, description: &'static str) -> ToolDefinition {
        ToolDefinition::new(&MockTool { name, description })
    }

    #[test]
    fn test_bm25_search() {
        let tools = vec![
            make_tool("read_file", "Read contents of a file from the filesystem"),
            make_tool("write_file", "Write content to a file"),
            make_tool("list_directory", "List files in a directory"),
            make_tool("run_command", "Execute a shell command"),
        ];

        let results = bm25_search("file", &tools, 3);
        assert!(!results.is_empty());
        // Should find read_file and write_file
        assert!(results.contains(&0) || results.contains(&1));
    }

    #[test]
    fn test_regex_search() {
        let tools = vec![
            make_tool("read_file", "Read contents of a file"),
            make_tool("write_file", "Write content to a file"),
            make_tool("run_command", "Execute a shell command"),
        ];

        let results = regex_search("file", &tools, 2);
        assert_eq!(results.len(), 2);
        assert!(results.contains(&0));
        assert!(results.contains(&1));
    }

    #[test]
    fn test_empty_query() {
        let tools = vec![make_tool("test", "Test tool")];

        let results = bm25_search("", &tools, 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_no_matches() {
        let tools = vec![make_tool("calculator", "Perform math operations")];

        let results = bm25_search("database", &tools, 5);
        assert!(results.is_empty());
    }
}
