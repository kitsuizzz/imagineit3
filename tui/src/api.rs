//! HTTP client for the public ImagineIt v3 API.
//!
//! All work runs on a background Tokio runtime; the TUI receives `AppEvent`s
//! through an unbounded channel.

use std::time::Duration;

use anyhow::{bail, Context, Result};
use image::DynamicImage;
use reqwest::{Client, Response, StatusCode};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::mpsc::UnboundedSender;

use crate::crypto::{unwrap_private_key, wrap_private_key, Identity};

const POLL_INTERVAL: Duration = Duration::from_millis(500);

#[derive(Clone)]
pub struct Account {
    pub server: String,
    pub token: String,
    pub account_id: String,
    pub identity: Identity,
}

#[derive(Debug, Clone, Serialize)]
pub struct GenerationParams {
    pub prompt: String,
    pub negative_prompt: Option<String>,
    pub width: u16,
    pub height: u16,
    pub cfg: f32,
    pub steps: u16,
    pub seed: u64,
    pub clip_skip: u8,
    pub scheduler: String,
}

#[derive(Debug, Clone)]
pub struct ProgressUpdate {
    pub status: String,
    pub steps_completed: u64,
    pub steps_total: Option<u64>,
    pub error: Option<String>,
}

#[derive(Debug, Clone)]
pub struct GeneratedImage {
    pub promise_id: String,
    pub image: DynamicImage,
    pub jpeg: Vec<u8>,
    pub seed: u64,
    pub scheduler: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct WorkerInfo {
    pub worker_id: String,
    pub name: String,
    pub model: String,
    pub device: String,
    #[serde(default)]
    pub kind: String,
    pub status: String,
    pub connected_seconds: u64,
    pub last_seen_seconds_ago: u64,
    pub has_active_job: bool,
}

#[derive(Clone, Deserialize)]
pub struct WorkerKeyMaterial {
    pub worker_id: String,
    pub key_id: String,
    pub key: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct KaggleProfile {
    pub id: String,
    #[serde(default)]
    pub kind: String,
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub worker_id: String,
    #[serde(default)]
    pub has_key: bool,
    #[serde(default)]
    pub created_at: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KaggleAction {
    Refresh,
    Create,
    Start,
    Stop,
    Delete,
}

pub enum AppEvent {
    AccountCreated {
        account: Account,
        token: String,
        warning: Option<String>,
    },
    SignedIn {
        account: Account,
    },
    AuthFailed(String),
    Validated {
        valid: bool,
        message: Option<String>,
    },
    Notice {
        message: String,
        error: bool,
    },
    Submitted {
        promise_id: String,
    },
    Progress {
        promise_id: String,
        update: ProgressUpdate,
    },
    ImageReady(GeneratedImage),
    Failed {
        promise_id: Option<String>,
        message: String,
    },
    WorkersUpdated {
        account_id: String,
        workers: Vec<WorkerInfo>,
    },
    WorkersFailed {
        account_id: String,
        message: String,
    },
    WorkerKeyReady {
        account_id: String,
        material: WorkerKeyMaterial,
    },
    WorkerKeyFailed {
        account_id: String,
        message: String,
    },
    KaggleProfilesUpdated {
        account_id: String,
        action: KaggleAction,
        profiles: Vec<KaggleProfile>,
        preferred_id: Option<String>,
        message: Option<String>,
    },
    KaggleFailed {
        account_id: String,
        action: KaggleAction,
        message: String,
    },
}

#[derive(Debug, Deserialize)]
struct AccountCreatedResponse {
    account_id: String,
    token: String,
}

#[derive(Debug, Deserialize)]
struct AccountInfo {
    account_id: String,
}

#[derive(Debug, Deserialize)]
struct InferenceCreated {
    promise_id: String,
}

#[derive(Debug, Deserialize)]
struct WrappedKeyResponse {
    wrapped_key: String,
}

#[derive(Debug, Deserialize)]
struct WorkerListResponse {
    #[serde(default)]
    workers: Vec<WorkerInfo>,
}

#[derive(Debug, Deserialize)]
struct ProfileListResponse {
    #[serde(default)]
    profiles: Vec<KaggleProfile>,
}

#[derive(Debug, Deserialize)]
struct ProfileCreatedResponse {
    profile: KaggleProfile,
}

#[derive(Debug, Deserialize)]
struct KaggleReportResponse {
    #[serde(default)]
    report: Option<String>,
}

#[derive(Debug, Deserialize)]
struct InferenceStatus {
    status: String,
    #[serde(default)]
    steps_completed: u64,
    steps_total: Option<u64>,
    error: Option<String>,
}

pub fn build_client() -> Result<Client> {
    Client::builder()
        .user_agent(concat!("imagineit-tui/", env!("CARGO_PKG_VERSION")))
        .connect_timeout(Duration::from_secs(10))
        .timeout(Duration::from_secs(120))
        .build()
        .context("could not build the HTTP client")
}

fn url(server: &str, path: &str) -> String {
    format!("{}{}", server.trim_end_matches('/'), path)
}

async fn check(response: Response) -> Result<Response> {
    if response.status().is_success() {
        return Ok(response);
    }

    let status = response.status();
    let body = response.text().await.unwrap_or_default();
    let json = serde_json::from_str::<Value>(&body).ok();
    let field = |name: &str| {
        json.as_ref()
            .and_then(|value| value.get(name))
            .and_then(Value::as_str)
            .map(str::to_owned)
    };
    let detail_value = json.as_ref().and_then(|value| value.get("detail"));
    let detail = detail_value
        .and_then(Value::as_str)
        .map(str::to_owned)
        .or_else(|| {
            detail_value
                .and_then(|value| value.get("error"))
                .and_then(Value::as_str)
                .map(str::to_owned)
        })
        .map(|value| match value.as_str() {
            "kaggle_api_key_required" => "a Kaggle API key is required".to_string(),
            "kaggle_username_unknown" => {
                "Kaggle could not detect your username from the API key".to_string()
            }
            _ => value,
        });
    let error = field("error");
    let hint = field("hint");

    let message = if let Some(detail) = detail {
        detail
    } else if let Some(error) = error {
        match hint {
            Some(hint) => format!("{error} ({hint})"),
            None => error,
        }
    } else {
        let body = body.trim();
        if body.is_empty() {
            format!("server returned {status}")
        } else {
            format!("server returned {status}: {}", truncate(body, 240))
        }
    };
    bail!("{message}")
}

/// Public deployments sometimes gate the app API behind a reverse-proxy
/// bridge (the web client uses the same-origin `/ui-api` prefix). Detect that
/// automatically so a bare hostname works as well as an explicit prefix.
pub async fn resolve_server(server: &str, client: &Client) -> String {
    let server = server.trim_end_matches('/');
    let Some((status, final_base)) = probe_status(server, client).await else {
        return server.to_string();
    };
    let final_base = final_base.unwrap_or_else(|| server.to_string());

    if status.is_success() {
        return final_base;
    }

    if status == StatusCode::UNAUTHORIZED {
        let bridge = format!("{final_base}/ui-api");
        if let Some((bridge_status, bridge_base)) = probe_status(&bridge, client).await {
            if bridge_status.is_success() {
                return bridge_base.unwrap_or(bridge);
            }
        }
    }

    final_base
}

async fn probe_status(server: &str, client: &Client) -> Option<(StatusCode, Option<String>)> {
    let response = client
        .get(url(server, "/api/v3/status"))
        .timeout(Duration::from_secs(4))
        .send()
        .await
        .ok()?;
    let status = response.status();
    let base = response
        .url()
        .as_str()
        .strip_suffix("/api/v3/status")
        .map(str::to_owned);
    Some((status, base))
}

fn truncate(text: &str, max_chars: usize) -> String {
    let mut chars = text.chars();
    let result: String = chars.by_ref().take(max_chars).collect();
    if chars.next().is_some() {
        format!("{result}…")
    } else {
        result
    }
}

pub async fn create_account(server: String, client: Client, tx: UnboundedSender<AppEvent>) {
    let result = create_account_inner(&server, &client).await;
    match result {
        Ok((account, token, warning)) => {
            let _ = tx.send(AppEvent::AccountCreated {
                account,
                token,
                warning,
            });
        }
        Err(err) => {
            let _ = tx.send(AppEvent::AuthFailed(format!("{err:#}")));
        }
    }
}

async fn create_account_inner(
    server: &str,
    client: &Client,
) -> Result<(Account, String, Option<String>)> {
    let identity = Identity::generate();
    let public_key = identity.public_key_b64();

    let response = client
        .post(url(server, "/api/v3/accounts"))
        .json(&serde_json::json!({ "public_key": public_key }))
        .send()
        .await
        .context("could not reach the ImagineIt server")?;
    let response = check(response).await?;
    let created: AccountCreatedResponse = response
        .json()
        .await
        .context("signup returned an unexpected response")?;

    let account = Account {
        server: server.trim_end_matches('/').to_string(),
        token: created.token.clone(),
        account_id: created.account_id,
        identity,
    };

    // Escrow the private key so the token can restore it elsewhere. This is a
    // convenience, not a requirement for using the account on this machine.
    let warning = match wrap_private_key(&account.token, &account.identity) {
        Ok(wrapped) => match client
            .put(url(server, "/api/v3/account/wrapped-key"))
            .bearer_auth(&account.token)
            .json(&serde_json::json!({ "wrapped_key": wrapped }))
            .send()
            .await
        {
            Ok(response) if response.status().is_success() => None,
            Ok(response) => Some(format!(
                "account saved locally, but key escrow failed ({}). \
                 The token may not work on another machine.",
                response.status()
            )),
            Err(err) => Some(format!(
                "account saved locally, but key escrow failed ({err}). \
                 The token may not work on another machine."
            )),
        },
        Err(err) => Some(format!(
            "account saved locally, but the key could not be wrapped ({err:#}). \
             The token may not work on another machine."
        )),
    };

    Ok((account, created.token, warning))
}

pub async fn sign_in(server: String, token: String, client: Client, tx: UnboundedSender<AppEvent>) {
    let result = sign_in_inner(&server, &token, &client).await;
    match result {
        Ok(account) => {
            let _ = tx.send(AppEvent::SignedIn { account });
        }
        Err(err) => {
            let _ = tx.send(AppEvent::AuthFailed(format!("{err:#}")));
        }
    }
}

async fn sign_in_inner(server: &str, token: &str, client: &Client) -> Result<Account> {
    let token = token.trim();
    if token.is_empty() {
        bail!("paste your account token first");
    }

    let response = client
        .get(url(server, "/api/v3/account"))
        .bearer_auth(token)
        .send()
        .await
        .context("could not reach the ImagineIt server")?;
    if response.status() == StatusCode::UNAUTHORIZED {
        bail!("that token is not valid");
    }
    let response = check(response).await?;
    let account_info: AccountInfo = response
        .json()
        .await
        .context("account check returned an unexpected response")?;

    let response = client
        .get(url(server, "/api/v3/account/wrapped-key"))
        .bearer_auth(token)
        .send()
        .await
        .context("could not fetch the escrowed key")?;
    if response.status() == StatusCode::NOT_FOUND {
        bail!(
            "this token has no escrowed encryption key on the server. \
             Sign in on the machine where the account was created, or create a new account."
        );
    }
    let response = check(response).await?;
    let wrapped: WrappedKeyResponse = response
        .json()
        .await
        .context("key escrow returned an unexpected response")?;
    let identity = unwrap_private_key(token, &wrapped.wrapped_key)
        .context("the escrowed key could not be opened with this token")?;

    Ok(Account {
        server: server.trim_end_matches('/').to_string(),
        token: token.to_string(),
        account_id: account_info.account_id,
        identity,
    })
}

pub async fn validate_account(account: Account, client: Client, tx: UnboundedSender<AppEvent>) {
    let response = client
        .get(url(&account.server, "/api/v3/account"))
        .bearer_auth(&account.token)
        .send()
        .await;

    match response {
        Ok(response) if response.status() == StatusCode::UNAUTHORIZED => {
            let _ = tx.send(AppEvent::Validated {
                valid: false,
                message: Some("Your saved token is no longer valid.".to_string()),
            });
        }
        Ok(response) => match check(response).await {
            Ok(response) => match response.json::<AccountInfo>().await {
                Ok(info) if info.account_id == account.account_id => {}
                Ok(_) => {
                    let _ = tx.send(AppEvent::Notice {
                        message: "The server reports a different account for this token.".into(),
                        error: true,
                    });
                }
                Err(err) => {
                    let _ = tx.send(AppEvent::Notice {
                        message: format!("Could not validate the saved account: {err}"),
                        error: true,
                    });
                }
            },
            Err(err) => {
                let _ = tx.send(AppEvent::Notice {
                    message: format!("Could not validate the saved account: {err:#}"),
                    error: false,
                });
            }
        },
        Err(err) => {
            let _ = tx.send(AppEvent::Notice {
                message: format!("Server unreachable; staying signed in. ({err})"),
                error: false,
            });
        }
    }
}

pub async fn fetch_workers(account: Account, client: Client, tx: UnboundedSender<AppEvent>) {
    let account_id = account.account_id.clone();
    let result = fetch_workers_inner(&account, &client).await;
    match result {
        Ok(workers) => {
            let _ = tx.send(AppEvent::WorkersUpdated {
                account_id,
                workers,
            });
        }
        Err(err) => {
            let _ = tx.send(AppEvent::WorkersFailed {
                account_id,
                message: format!("{err:#}"),
            });
        }
    }
}

async fn fetch_workers_inner(account: &Account, client: &Client) -> Result<Vec<WorkerInfo>> {
    let response = client
        .get(url(&account.server, "/api/v3/workers"))
        .bearer_auth(&account.token)
        .timeout(Duration::from_secs(10))
        .send()
        .await
        .context("could not fetch connected GPU workers")?;
    let response = check(response).await?;
    let list: WorkerListResponse = response
        .json()
        .await
        .context("the worker list returned an unexpected response")?;
    Ok(list.workers)
}

/// Fetch the newest worker key, creating one if the account has none. The key
/// is returned in plaintext only to the authenticated account.
pub async fn ensure_worker_key(account: Account, client: Client, tx: UnboundedSender<AppEvent>) {
    let account_id = account.account_id.clone();
    let result = ensure_worker_key_inner(&account, &client).await;
    match result {
        Ok(material) => {
            let _ = tx.send(AppEvent::WorkerKeyReady {
                account_id,
                material,
            });
        }
        Err(err) => {
            let _ = tx.send(AppEvent::WorkerKeyFailed {
                account_id,
                message: format!("{err:#}"),
            });
        }
    }
}

async fn ensure_worker_key_inner(account: &Account, client: &Client) -> Result<WorkerKeyMaterial> {
    let response = client
        .get(url(&account.server, "/api/v3/workers/keys/current"))
        .bearer_auth(&account.token)
        .timeout(Duration::from_secs(20))
        .send()
        .await
        .context("could not fetch a GPU worker key")?;
    let response = check(response).await?;
    response
        .json()
        .await
        .context("the worker key endpoint returned an unexpected response")
}

pub async fn fetch_kaggle_profiles(
    account: Account,
    client: Client,
    tx: UnboundedSender<AppEvent>,
) {
    let account_id = account.account_id.clone();
    match fetch_kaggle_profiles_inner(&account, &client).await {
        Ok(profiles) => {
            let _ = tx.send(AppEvent::KaggleProfilesUpdated {
                account_id,
                action: KaggleAction::Refresh,
                profiles,
                preferred_id: None,
                message: None,
            });
        }
        Err(err) => {
            let _ = tx.send(AppEvent::KaggleFailed {
                account_id,
                action: KaggleAction::Refresh,
                message: format!("{err:#}"),
            });
        }
    }
}

pub async fn create_kaggle_profile(
    account: Account,
    client: Client,
    api_key: String,
    tx: UnboundedSender<AppEvent>,
) {
    let account_id = account.account_id.clone();
    match create_kaggle_profile_inner(&account, &client, &api_key).await {
        Ok((profiles, profile_id)) => {
            let _ = tx.send(AppEvent::KaggleProfilesUpdated {
                account_id,
                action: KaggleAction::Create,
                profiles,
                preferred_id: Some(profile_id),
                message: Some("Kaggle profile saved. Press F5 to start it on Kaggle.".to_string()),
            });
        }
        Err(err) => {
            let _ = tx.send(AppEvent::KaggleFailed {
                account_id,
                action: KaggleAction::Create,
                message: format!("{err:#}"),
            });
        }
    }
}

pub async fn run_kaggle_profile_action(
    account: Account,
    client: Client,
    action: KaggleAction,
    profile_id: String,
    tx: UnboundedSender<AppEvent>,
) {
    let account_id = account.account_id.clone();
    let result = if action == KaggleAction::Delete {
        delete_kaggle_profile_inner(&account, &client, &profile_id)
            .await
            .map(|profiles| (profiles, "Kaggle profile deleted.".to_string()))
    } else {
        kaggle_profile_action_inner(&account, &client, action, &profile_id).await
    };
    match result {
        Ok((profiles, message)) => {
            let preferred_id = if action == KaggleAction::Delete {
                None
            } else {
                Some(profile_id)
            };
            let _ = tx.send(AppEvent::KaggleProfilesUpdated {
                account_id,
                action,
                profiles,
                preferred_id,
                message: Some(message),
            });
        }
        Err(err) => {
            let _ = tx.send(AppEvent::KaggleFailed {
                account_id,
                action,
                message: format!("{err:#}"),
            });
        }
    }
}

async fn fetch_kaggle_profiles_inner(
    account: &Account,
    client: &Client,
) -> Result<Vec<KaggleProfile>> {
    let response = client
        .get(url(&account.server, "/api/v3/gpu/profiles"))
        .bearer_auth(&account.token)
        .timeout(Duration::from_secs(20))
        .send()
        .await
        .context("could not fetch GPU profiles")?;
    let response = check(response).await?;
    let data: ProfileListResponse = response
        .json()
        .await
        .context("the GPU profile list returned an unexpected response")?;
    Ok(data.profiles)
}

async fn create_kaggle_profile_inner(
    account: &Account,
    client: &Client,
    api_key: &str,
) -> Result<(Vec<KaggleProfile>, String)> {
    let response = client
        .post(url(&account.server, "/api/v3/gpu/profiles"))
        .bearer_auth(&account.token)
        .timeout(Duration::from_secs(30))
        .json(&serde_json::json!({
            "kind": "kaggle",
            "api_key": api_key,
        }))
        .send()
        .await
        .context("could not create the Kaggle GPU profile")?;
    let response = check(response).await?;
    let created: ProfileCreatedResponse = response
        .json()
        .await
        .context("the GPU profile endpoint returned an unexpected response")?;
    let profiles = fetch_kaggle_profiles_inner(account, client).await?;
    Ok((profiles, created.profile.id))
}

async fn kaggle_profile_action_inner(
    account: &Account,
    client: &Client,
    action: KaggleAction,
    profile_id: &str,
) -> Result<(Vec<KaggleProfile>, String)> {
    let path = format!(
        "/api/v3/gpu/profiles/{profile_id}/{}",
        match action {
            KaggleAction::Start => "start",
            KaggleAction::Stop => "stop",
            _ => bail!("unsupported Kaggle profile action"),
        }
    );
    let timeout = if action == KaggleAction::Start {
        Duration::from_secs(240)
    } else {
        Duration::from_secs(30)
    };
    let response = client
        .post(url(&account.server, &path))
        .bearer_auth(&account.token)
        .timeout(timeout)
        .json(&serde_json::json!({}))
        .send()
        .await
        .context("could not reach the Kaggle profile through the server")?;
    let response = check(response).await?;
    let data: KaggleReportResponse = response
        .json()
        .await
        .context("the Kaggle profile endpoint returned an unexpected response")?;
    let profiles = fetch_kaggle_profiles_inner(account, client).await?;
    let message = data.report.unwrap_or_else(|| match action {
        KaggleAction::Stop => "Kaggle stop requested.".to_string(),
        _ => "Kaggle profile started.".to_string(),
    });
    Ok((profiles, message))
}

async fn delete_kaggle_profile_inner(
    account: &Account,
    client: &Client,
    profile_id: &str,
) -> Result<Vec<KaggleProfile>> {
    let response = client
        .delete(url(
            &account.server,
            &format!("/api/v3/gpu/profiles/{profile_id}"),
        ))
        .bearer_auth(&account.token)
        .timeout(Duration::from_secs(20))
        .send()
        .await
        .context("could not delete the Kaggle GPU profile")?;
    let response = check(response).await?;
    let _: Value = response.json().await.unwrap_or(Value::Null);
    fetch_kaggle_profiles_inner(account, client).await
}

pub async fn generate(
    account: Account,
    client: Client,
    params: GenerationParams,
    tx: UnboundedSender<AppEvent>,
) {
    let promise_id = match submit(&account, &client, &params).await {
        Ok(promise_id) => {
            let _ = tx.send(AppEvent::Submitted {
                promise_id: promise_id.clone(),
            });
            promise_id
        }
        Err(err) => {
            let _ = tx.send(AppEvent::Failed {
                promise_id: None,
                message: format!("{err:#}"),
            });
            return;
        }
    };

    match watch(&account, &client, &params, &promise_id, &tx).await {
        Ok(generated) => {
            let _ = tx.send(AppEvent::ImageReady(generated));
        }
        Err(err) => {
            let _ = tx.send(AppEvent::Failed {
                promise_id: Some(promise_id),
                message: format!("{err:#}"),
            });
        }
    }
}

async fn submit(account: &Account, client: &Client, params: &GenerationParams) -> Result<String> {
    let response = client
        .post(url(&account.server, "/api/v3/images/inferences"))
        .bearer_auth(&account.token)
        .json(params)
        .send()
        .await
        .context("could not reach the ImagineIt server")?;
    let response = check(response).await?;
    let created: InferenceCreated = response
        .json()
        .await
        .context("the inference request returned an unexpected response")?;
    Ok(created.promise_id)
}

async fn watch(
    account: &Account,
    client: &Client,
    params: &GenerationParams,
    promise_id: &str,
    tx: &UnboundedSender<AppEvent>,
) -> Result<GeneratedImage> {
    loop {
        let response = client
            .get(url(
                &account.server,
                &format!("/api/v3/images/inferences/{promise_id}"),
            ))
            .bearer_auth(&account.token)
            .send()
            .await
            .context("lost contact with the ImagineIt server while polling")?;
        let response = check(response).await?;
        let status: InferenceStatus = response
            .json()
            .await
            .context("the inference status returned an unexpected response")?;

        let update = ProgressUpdate {
            status: status.status.clone(),
            steps_completed: status.steps_completed,
            steps_total: status.steps_total,
            error: status.error.clone(),
        };
        let _ = tx.send(AppEvent::Progress {
            promise_id: promise_id.to_string(),
            update: update.clone(),
        });

        match status.status.as_str() {
            "succeeded" => break,
            "failed" | "cancelled" => {
                bail!(
                    "{}",
                    update
                        .error
                        .unwrap_or_else(|| format!("inference {}", update.status))
                )
            }
            _ => tokio::time::sleep(POLL_INTERVAL).await,
        }
    }

    let response = client
        .get(url(
            &account.server,
            &format!("/api/v3/images/{promise_id}"),
        ))
        .bearer_auth(&account.token)
        .send()
        .await
        .context("could not download the generated image")?;
    let response = check(response).await?;
    let sealed = response
        .bytes()
        .await
        .context("could not read the generated image")?;
    let jpeg = account
        .identity
        .open(&sealed)
        .context("the generated image could not be decrypted")?;
    let image =
        image::load_from_memory(&jpeg).context("the generated image could not be decoded")?;

    Ok(GeneratedImage {
        promise_id: promise_id.to_string(),
        image,
        jpeg,
        seed: params.seed,
        scheduler: params.scheduler.clone(),
    })
}
