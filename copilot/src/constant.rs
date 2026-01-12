//! Constants for GitHub Copilot API integration.

/// Base URL for GitHub Copilot API.
pub const COPILOT_BASE_URL: &str = "https://api.githubcopilot.com";

/// GitHub OAuth device authorization endpoint.
pub const GITHUB_DEVICE_CODE_URL: &str = "https://github.com/login/device/code";

/// GitHub OAuth token endpoint.
pub const GITHUB_TOKEN_URL: &str = "https://github.com/login/oauth/access_token";

/// GitHub API endpoint to exchange OAuth token for Copilot session token.
pub const COPILOT_TOKEN_URL: &str = "https://api.github.com/copilot_internal/v2/token";

/// OAuth client ID for GitHub Copilot (VS Code's client ID).
pub const COPILOT_CLIENT_ID: &str = "Iv1.b507a08c87ecfe98";

/// Default model for chat completions.
pub const DEFAULT_MODEL: &str = "gpt-4o";

// Available models through GitHub Copilot
/// GPT-4 model.
pub const GPT4: &str = "gpt-4";
/// GPT-4o model.
pub const GPT4O: &str = "gpt-4o";
/// GPT-4o mini model.
pub const GPT4O_MINI: &str = "gpt-4o-mini";
/// Claude 3.5 Sonnet (available via Copilot Enterprise).
pub const CLAUDE_SONNET: &str = "claude-3.5-sonnet";

/// Editor version header value (identifies as VS Code).
pub const EDITOR_VERSION: &str = "vscode/1.96.0";

/// Copilot integration ID header value.
pub const COPILOT_INTEGRATION_ID: &str = "vscode-chat";
