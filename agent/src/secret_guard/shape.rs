//! The shapes of a key or token.
//!
//! Each rule is built to match a credential and nothing that merely looks
//! like one. A rule whose prefix is a vendor's (`sk-`, `ghp_`, `AKIA`, …)
//! matches only a whole token: the characters on both sides must not
//! continue it, so a run inside base64 data or a longer identifier never
//! matches. A rule that keys on context rather than a prefix — a bearer
//! token, a value assigned to a secret-named key — also requires the value
//! to mix letters and digits, which prose, paths and identifiers rarely do.
//! Hashes, UUIDs and identifiers carry no vendor prefix and no secret-named
//! key, so no rule reaches them.

use std::ops::Range;

use regex::Regex;

/// The shape of a key or token found in text.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SecretShape {
    /// A vendor API key or access token with a vendor prefix, such as an
    /// `OpenAI`, Anthropic, GitHub, GitLab, Slack, Google, Stripe, npm or
    /// Hugging Face key.
    ProviderKey,
    /// An AWS access key ID.
    AwsAccessKeyId,
    /// A JSON web token.
    Jwt,
    /// A PEM private key block.
    PrivateKey,
    /// The credential of an HTTP `Bearer` authorization.
    BearerToken,
    /// A value assigned to a key named as a secret, such as `DB_PASSWORD=…`
    /// or `"api_key": "…"`.
    Assignment,
}

impl std::fmt::Display for SecretShape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::ProviderKey => "provider API key",
            Self::AwsAccessKeyId => "AWS access key ID",
            Self::Jwt => "JSON web token",
            Self::PrivateKey => "private key",
            Self::BearerToken => "bearer token",
            Self::Assignment => "value of a secret-named key",
        })
    }
}

/// What a match must satisfy beyond its pattern.
#[derive(Debug, Clone, Copy)]
enum Gate {
    /// Nothing further: the pattern alone is specific enough.
    None,
    /// A whole token, mixing upper case, lower case and digits, as a
    /// randomly generated key does.
    RandomToken,
    /// A whole token of letters and digits.
    AlphanumericToken,
    /// A whole token of letters and digits, assigned to a key whose name
    /// ends in a secret word.
    SecretKey,
}

#[derive(Debug)]
struct Rule {
    shape: SecretShape,
    regex: Regex,
    gate: Gate,
}

/// Vendor key prefixes, each with the length and alphabet its vendor
/// documents.
const PROVIDER_KEY: &str = concat!(
    r"sk-(?:ant-|proj-|svcacct-|admin-)?[A-Za-z0-9_-]{20,}",
    r"|gh[pousr]_[A-Za-z0-9]{36}",
    r"|github_pat_[A-Za-z0-9_]{22,}",
    r"|glpat-[A-Za-z0-9_-]{20,}",
    r"|xox[abposr]-[A-Za-z0-9-]{10,}",
    r"|AIza[A-Za-z0-9_-]{35}",
    r"|[sr]k_(?:live|test)_[A-Za-z0-9]{16,}",
    r"|npm_[A-Za-z0-9]{36}",
    r"|hf_[A-Za-z0-9]{34,}",
);
const AWS_ACCESS_KEY_ID: &str = r"(?:AKIA|ASIA)[A-Z0-9]{16}";
const JWT: &str = r"eyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{16,}";
/// A block without its end marker, as truncated output leaves it, runs to
/// the end of the text.
const PRIVATE_KEY: &str =
    r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----(?s:.)*?(?:-----END [A-Z0-9 ]*PRIVATE KEY-----|\z)";
const BEARER_TOKEN: &str = r"[Bb]earer[ \t]+([A-Za-z0-9._~+/=-]{16,})";
const ASSIGNMENT: &str =
    r#"([A-Za-z_][A-Za-z0-9_.-]*)["']?[ \t]*[:=][ \t]*["']?([A-Za-z0-9_./+=~-]{16,})"#;

/// Key-name endings, in lower case, that name the assigned value a secret.
const SECRET_KEY_SUFFIXES: &[&str] = &[
    "apikey",
    "api_key",
    "api-key",
    "secret",
    "secretkey",
    "secret_key",
    "token",
    "password",
    "passwd",
    "accesskey",
    "access_key",
    "privatekey",
    "private_key",
];

/// The compiled shape rules.
#[derive(Debug)]
pub(super) struct SecretShapes {
    rules: Vec<Rule>,
}

impl SecretShapes {
    pub(super) fn new() -> Self {
        let rule = |shape, pattern: &str, gate| Rule {
            shape,
            regex: Regex::new(pattern).expect("secret shape patterns are valid"),
            gate,
        };
        Self {
            rules: vec![
                rule(SecretShape::PrivateKey, PRIVATE_KEY, Gate::None),
                rule(SecretShape::ProviderKey, PROVIDER_KEY, Gate::RandomToken),
                rule(
                    SecretShape::AwsAccessKeyId,
                    AWS_ACCESS_KEY_ID,
                    Gate::AlphanumericToken,
                ),
                rule(SecretShape::Jwt, JWT, Gate::AlphanumericToken),
                rule(
                    SecretShape::BearerToken,
                    BEARER_TOKEN,
                    Gate::AlphanumericToken,
                ),
                rule(SecretShape::Assignment, ASSIGNMENT, Gate::SecretKey),
            ],
        }
    }

    /// Calls `found` with each span of `text` in the shape of a key or
    /// token, rule by rule; spans of different rules may overlap.
    pub(super) fn find(&self, text: &str, mut found: impl FnMut(SecretShape, Range<usize>)) {
        for rule in &self.rules {
            match rule.gate {
                Gate::None => {
                    for hit in rule.regex.find_iter(text) {
                        found(rule.shape, hit.range());
                    }
                }
                Gate::RandomToken | Gate::AlphanumericToken => {
                    let random = matches!(rule.gate, Gate::RandomToken);
                    for captures in rule.regex.captures_iter(text) {
                        // The credential is the last group: the whole match
                        // for a prefix rule, the value for a bearer token.
                        let hit = captures
                            .iter()
                            .flatten()
                            .last()
                            .expect("a match has its whole-match group");
                        let range = hit.range();
                        if is_whole_token(text, &range) && mixes(hit.as_str(), random) {
                            found(rule.shape, range);
                        }
                    }
                }
                Gate::SecretKey => {
                    for captures in rule.regex.captures_iter(text) {
                        let (Some(key), Some(value)) = (captures.get(1), captures.get(2)) else {
                            unreachable!("the assignment pattern captures a key and a value");
                        };
                        let range = value.range();
                        if is_secret_key(key.as_str())
                            && is_whole_token(text, &range)
                            && mixes(value.as_str(), false)
                        {
                            found(rule.shape, range);
                        }
                    }
                }
            }
        }
    }
}

/// Bytes that continue a token: a key or token never starts or ends next
/// to one.
const fn continues_token(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'+' | b'/')
}

/// Whether `range` is not part of a longer run of token characters. Token
/// characters are ASCII, so a byte check is exact on UTF-8 text.
fn is_whole_token(text: &str, range: &Range<usize>) -> bool {
    let bytes = text.as_bytes();
    let before = range
        .start
        .checked_sub(1)
        .and_then(|index| bytes.get(index));
    let after = bytes.get(range.end);
    !before.is_some_and(|&byte| continues_token(byte))
        && !after.is_some_and(|&byte| continues_token(byte))
}

/// Whether `token` mixes letters and digits; with `random`, upper and lower
/// case letters and digits.
fn mixes(token: &str, random: bool) -> bool {
    let bytes = token.as_bytes();
    let digit = bytes.iter().any(u8::is_ascii_digit);
    if random {
        digit
            && bytes.iter().any(u8::is_ascii_uppercase)
            && bytes.iter().any(u8::is_ascii_lowercase)
    } else {
        digit && bytes.iter().any(u8::is_ascii_alphabetic)
    }
}

fn is_secret_key(key: &str) -> bool {
    let key = key.to_ascii_lowercase();
    SECRET_KEY_SUFFIXES
        .iter()
        .any(|suffix| key.ends_with(suffix))
}
