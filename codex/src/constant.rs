//! Constants for the `ChatGPT` Codex backend and its OAuth flow.

/// `OpenAI` auth issuer base URL.
pub const ISSUER: &str = "https://auth.openai.com";

/// Public OAuth client ID shared by Codex-compatible third-party clients.
///
/// Registered for the `http://localhost:1455` loopback redirect; the port and
/// path are fixed by the client's registration and cannot be changed.
pub const CLIENT_ID: &str = "app_EMoamEEZ73f0CkXaXp7hrann";

/// OAuth authorization endpoint (browser flow).
pub const AUTHORIZE_URL: &str = "https://auth.openai.com/oauth/authorize";

/// OAuth token endpoint (authorization-code exchange and refresh grants).
pub const TOKEN_URL: &str = "https://auth.openai.com/oauth/token";

/// Device authorization request endpoint (headless flow).
pub const DEVICE_CODE_URL: &str = "https://auth.openai.com/api/accounts/deviceauth/usercode";

/// Device authorization token polling endpoint.
pub const DEVICE_TOKEN_URL: &str = "https://auth.openai.com/api/accounts/deviceauth/token";

/// Page where the user enters the device code.
pub const DEVICE_VERIFICATION_URL: &str = "https://auth.openai.com/codex/device";

/// Local port for the OAuth loopback redirect listener.
pub const OAUTH_PORT: u16 = 1455;

/// Redirect URI registered for [`CLIENT_ID`].
pub const REDIRECT_URI: &str = "http://localhost:1455/auth/callback";

/// OAuth scopes requested at login.
pub const OAUTH_SCOPES: &str = "openid profile email offline_access";

/// Base URL for the subscription Codex Responses backend.
pub const CODEX_BASE_URL: &str = "https://chatgpt.com/backend-api/codex";

/// Client version sent to the models endpoint and as request metadata.
pub const CLIENT_VERSION: &str = "0.160.0";

/// `originator` header value identifying this client to the backend.
pub const ORIGINATOR: &str = "may";

/// The header carrying the `ChatGPT` account identifier.
pub const ACCOUNT_ID_HEADER: &str = "ChatGPT-Account-Id";

/// Optional residency hint header; only sent when the token carries a real
/// constraint (`no_constraint` is omitted).
pub const RESIDENCY_HEADER: &str = "x-openai-internal-codex-residency";
