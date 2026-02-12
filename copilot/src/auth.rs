//! OAuth device flow authentication for GitHub Copilot.
//!
//! GitHub Copilot uses OAuth 2.0 device flow for authentication. The flow is:
//!
//! 1. Request a device code from GitHub
//! 2. Display the user code and verification URL to the user
//! 3. Poll the token endpoint until the user completes authorization
//!
//! # Example
//!
//! ```no_run
//! use aither_copilot::auth;
//!
//! # async fn demo() -> Result<(), aither_copilot::CopilotError> {
//! // Step 1: Get device code
//! let device = auth::request_device_code().await?;
//!
//! // Step 2: Show user the code
//! println!("Visit {} and enter code: {}", device.verification_uri, device.user_code);
//!
//! // Step 3: Poll for token
//! let token = auth::poll_for_token(&device.device_code, device.interval).await?;
//!
//! println!("Got token: {}", &token.access_token[..20]);
//! # Ok(())
//! # }
//! ```

use crate::{
    CopilotError,
    constant::{COPILOT_CLIENT_ID, GITHUB_DEVICE_CODE_URL, GITHUB_TOKEN_URL},
};
use serde::{Deserialize, Serialize};
use std::time::{Duration, SystemTime};
use zenwave::{Client, client};

/// Response from the device code request.
#[derive(Debug, Clone, Deserialize)]
pub struct DeviceCodeResponse {
    /// The device verification code.
    pub device_code: String,
    /// The user-facing code to enter at the verification URL.
    pub user_code: String,
    /// The URL where the user should enter the code.
    pub verification_uri: String,
    /// Number of seconds until the device code expires.
    pub expires_in: u64,
    /// Minimum number of seconds between polling requests.
    pub interval: u64,
}

/// Token obtained from successful authentication.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CopilotToken {
    /// The access token for API requests.
    pub access_token: String,
    /// Token type (usually "bearer").
    pub token_type: String,
    /// OAuth scope granted.
    pub scope: String,
    /// When the token expires (if known).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expires_at: Option<SystemTime>,
}

/// Request body for device code endpoint.
#[derive(Serialize)]
struct DeviceCodeRequest<'a> {
    client_id: &'a str,
    scope: &'a str,
}

/// Request body for token endpoint.
#[derive(Serialize)]
struct TokenRequest<'a> {
    client_id: &'a str,
    device_code: &'a str,
    grant_type: &'a str,
}

/// Response from token endpoint.
#[derive(Deserialize)]
struct TokenResponse {
    access_token: Option<String>,
    token_type: Option<String>,
    scope: Option<String>,
    error: Option<String>,
    error_description: Option<String>,
}

/// Request a device code from GitHub to start the OAuth flow.
///
/// Returns a [`DeviceCodeResponse`] containing the user code and verification URL.
///
/// # Errors
///
/// Returns an error if the HTTP request fails or the response is malformed.
pub async fn request_device_code() -> Result<DeviceCodeResponse, CopilotError> {
    let mut backend = client();

    let request = DeviceCodeRequest {
        client_id: COPILOT_CLIENT_ID,
        scope: "copilot",
    };

    let response: DeviceCodeResponse = backend
        .post(GITHUB_DEVICE_CODE_URL)
        .map_err(CopilotError::Http)?
        .header("Accept", "application/json")
        .map_err(CopilotError::Http)?
        .json_body(&request)
        .map_err(CopilotError::Http)?
        .json()
        .await
        .map_err(CopilotError::Http)?;

    Ok(response)
}

/// Poll the token endpoint until the user completes authorization.
///
/// This function blocks (with async sleep) until:
/// - The user completes authorization (returns token)
/// - The device code expires (returns error)
/// - The user denies access (returns error)
///
/// # Arguments
///
/// * `device_code` - The device code from [`request_device_code`]
/// * `interval` - Minimum seconds between poll attempts (from device code response)
///
/// # Errors
///
/// Returns an error if authorization fails, expires, or is denied.
pub async fn poll_for_token(
    device_code: &str,
    interval: u64,
) -> Result<CopilotToken, CopilotError> {
    let poll_interval = Duration::from_secs(interval.max(5)); // At least 5 seconds

    loop {
        // Wait before polling
        sleep(poll_interval).await;

        match try_get_token(device_code).await {
            Ok(token) => return Ok(token),
            Err(CopilotError::AuthorizationPending) => {
                // User hasn't completed auth yet, keep polling
                tracing::debug!("Authorization pending, polling again...");
                continue;
            }
            Err(e) => return Err(e),
        }
    }
}

/// Attempt to get a token (single poll attempt).
///
/// Returns `Ok(token)` if the user has completed authorization.
/// Returns `Err(CopilotError::AuthorizationPending)` if still waiting.
/// Returns other errors for failures.
///
/// This is useful when you want to implement your own polling loop
/// with custom timeout handling.
pub async fn try_get_token(device_code: &str) -> Result<CopilotToken, CopilotError> {
    let mut backend = client();

    let request = TokenRequest {
        client_id: COPILOT_CLIENT_ID,
        device_code,
        grant_type: "urn:ietf:params:oauth:grant-type:device_code",
    };

    let response: TokenResponse = backend
        .post(GITHUB_TOKEN_URL)
        .map_err(CopilotError::Http)?
        .header("Accept", "application/json")
        .map_err(CopilotError::Http)?
        .json_body(&request)
        .map_err(CopilotError::Http)?
        .json()
        .await
        .map_err(CopilotError::Http)?;

    // Check for errors
    if let Some(error) = response.error {
        return match error.as_str() {
            "authorization_pending" => Err(CopilotError::AuthorizationPending),
            "slow_down" => Err(CopilotError::AuthorizationPending), // Treat as pending, caller will wait
            "expired_token" => Err(CopilotError::DeviceCodeExpired),
            "access_denied" => Err(CopilotError::AccessDenied),
            _ => Err(CopilotError::Api(
                response.error_description.unwrap_or(error),
            )),
        };
    }

    // Extract token
    let access_token = response
        .access_token
        .ok_or_else(|| CopilotError::Api("Missing access_token in response".to_string()))?;
    let token_type = response.token_type.unwrap_or_else(|| "bearer".to_string());
    let scope = response.scope.unwrap_or_else(|| "copilot".to_string());

    Ok(CopilotToken {
        access_token,
        token_type,
        scope,
        expires_at: None, // GitHub doesn't return expiration for Copilot tokens
    })
}

/// Sleep for the given duration (runtime-agnostic).
async fn sleep(duration: Duration) {
    #[cfg(not(target_arch = "wasm32"))]
    {
        async_io::Timer::after(duration).await;
    }
    #[cfg(target_arch = "wasm32")]
    {
        gloo_timers::future::TimeoutFuture::new(duration.as_millis() as u32).await;
    }
}

/// Session token for Copilot API calls.
///
/// This is obtained by exchanging an OAuth token with the Copilot API.
/// Session tokens are short-lived (typically ~30 minutes).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionToken {
    /// The session token for API calls.
    pub token: String,
    /// The API endpoint to use (varies per user type).
    pub api_endpoint: String,
    /// When the token expires (Unix timestamp).
    pub expires_at: i64,
}

/// Endpoints returned from the token exchange.
#[derive(Debug, Deserialize)]
struct EndpointsResponse {
    api: String,
}

/// Response from the Copilot token exchange endpoint.
#[derive(Debug, Deserialize)]
struct SessionTokenResponse {
    token: String,
    endpoints: EndpointsResponse,
    expires_at: i64,
}

/// Exchange an OAuth token for a Copilot session token.
///
/// The OAuth token is the `access_token` from [`poll_for_token`].
/// The session token is what you use for actual API calls.
///
/// Session tokens are short-lived (typically ~30 minutes), so you may need
/// to call this again when the token expires.
///
/// # Example
///
/// ```no_run
/// use aither_copilot::auth;
///
/// # async fn demo() -> Result<(), aither_copilot::CopilotError> {
/// let oauth_token = "gho_xxxx"; // from poll_for_token
/// let session = auth::get_session_token(oauth_token).await?;
/// println!("API endpoint: {}", session.api_endpoint);
/// println!("Session token expires at: {}", session.expires_at);
/// # Ok(())
/// # }
/// ```
pub async fn get_session_token(oauth_token: &str) -> Result<SessionToken, CopilotError> {
    use crate::constant::{COPILOT_TOKEN_URL, EDITOR_VERSION};

    let mut backend = client();

    let response: SessionTokenResponse = backend
        .get(COPILOT_TOKEN_URL)
        .map_err(CopilotError::Http)?
        .header("Authorization", format!("token {oauth_token}"))
        .map_err(CopilotError::Http)?
        .header("Accept", "application/json")
        .map_err(CopilotError::Http)?
        .header(
            "User-Agent",
            format!("aither-copilot/0.1 ({EDITOR_VERSION})"),
        )
        .map_err(CopilotError::Http)?
        .json()
        .await
        .map_err(CopilotError::Http)?;

    Ok(SessionToken {
        token: response.token,
        api_endpoint: response.endpoints.api,
        expires_at: response.expires_at,
    })
}
