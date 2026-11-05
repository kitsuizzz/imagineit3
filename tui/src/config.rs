//! Local account storage.
//!
//! The account token and P-256 private key are written to a single JSON file
//! with owner-only permissions. Losing that file means losing the ability to
//! decrypt this account's images, so the key is also escrowed server-side,
//! wrapped with the token.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use base64::{engine::general_purpose::STANDARD, Engine};
use directories::BaseDirs;
use serde::{Deserialize, Serialize};

use crate::api::{Account, WorkerKeyMaterial};
use crate::crypto::Identity;

const CONFIG_VERSION: u32 = 1;

#[derive(Debug, Serialize, Deserialize)]
pub struct StoredAccount {
    #[serde(default = "config_version")]
    pub version: u32,
    pub server: String,
    pub token: String,
    pub account_id: String,
    pub private_key: String,
    #[serde(default)]
    pub public_key: String,
}

fn config_version() -> u32 {
    CONFIG_VERSION
}

pub fn default_path() -> PathBuf {
    if let Some(path) = std::env::var_os("IMAGINEIT_CONFIG") {
        return PathBuf::from(path);
    }
    if let Some(dirs) = BaseDirs::new() {
        return dirs.config_dir().join("imagineit-tui").join("account.json");
    }
    PathBuf::from(".imagineit-tui-account.json")
}

pub fn load(path: &Path) -> Result<Option<Account>> {
    let data = match fs::read(path) {
        Ok(data) => data,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(err) => return Err(err).with_context(|| format!("could not read {}", path.display())),
    };

    let stored: StoredAccount = serde_json::from_slice(&data)
        .with_context(|| format!("{} is not a valid account file", path.display()))?;
    let secret = STANDARD
        .decode(stored.private_key.trim())
        .context("account file contains an invalid private key")?;
    let identity = Identity::from_secret_bytes(&secret)?;

    // Prefer a freshly derived key over a stale public key in the file.
    let expected_public = identity.public_key_b64();
    if !stored.public_key.is_empty() && stored.public_key != expected_public {
        return Err(anyhow::anyhow!(
            "account file public and private keys do not match"
        ));
    }

    Ok(Some(Account {
        server: stored.server.trim_end_matches('/').to_string(),
        token: stored.token,
        account_id: stored.account_id,
        identity,
    }))
}

pub fn save(path: &Path, account: &Account) -> Result<()> {
    let stored = StoredAccount {
        version: CONFIG_VERSION,
        server: account.server.clone(),
        token: account.token.clone(),
        account_id: account.account_id.clone(),
        private_key: STANDARD.encode(account.identity.secret_bytes()),
        public_key: account.identity.public_key_b64(),
    };
    let data = serde_json::to_vec_pretty(&stored).context("could not serialize account")?;

    if let Some(parent) = path.parent() {
        let existed = parent.exists();
        fs::create_dir_all(parent)
            .with_context(|| format!("could not create {}", parent.display()))?;
        if !existed {
            set_dir_permissions(parent);
        }
    }

    let tmp = path.with_extension("json.tmp");
    fs::write(&tmp, data).with_context(|| format!("could not write {}", tmp.display()))?;
    set_file_permissions(&tmp);
    fs::rename(&tmp, path).with_context(|| format!("could not replace {}", path.display()))?;
    set_file_permissions(path);
    Ok(())
}

/// Save a worker key next to the account file, in a `workers/` subdirectory.
/// The returned path can be passed directly to `gpu_worker.py --key-file`.
pub fn save_worker_key(account_path: &Path, material: &WorkerKeyMaterial) -> Result<PathBuf> {
    let parent = account_path.parent().unwrap_or_else(|| Path::new("."));
    let dir = parent.join("workers");
    let existed = dir.exists();
    fs::create_dir_all(&dir).with_context(|| format!("could not create {}", dir.display()))?;
    if !existed {
        set_dir_permissions(&dir);
    }

    let safe_id: String = material
        .worker_id
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || character == '-' || character == '_' {
                character
            } else {
                '_'
            }
        })
        .collect();
    let path = dir.join(format!("{safe_id}.worker.json"));
    let data = serde_json::to_vec_pretty(&serde_json::json!({
        "worker_id": material.worker_id,
        "key_id": material.key_id,
        "key": material.key,
    }))
    .context("could not serialize the worker key")?;

    let tmp = path.with_extension("json.tmp");
    fs::write(&tmp, data).with_context(|| format!("could not write {}", tmp.display()))?;
    set_file_permissions(&tmp);
    fs::rename(&tmp, &path).with_context(|| format!("could not replace {}", path.display()))?;
    set_file_permissions(&path);
    Ok(path)
}

#[cfg(unix)]
fn set_file_permissions(path: &Path) {
    use std::os::unix::fs::PermissionsExt;
    let _ = fs::set_permissions(path, fs::Permissions::from_mode(0o600));
}

#[cfg(not(unix))]
fn set_file_permissions(_path: &Path) {}

#[cfg(unix)]
fn set_dir_permissions(path: &Path) {
    use std::os::unix::fs::PermissionsExt;
    let _ = fs::set_permissions(path, fs::Permissions::from_mode(0o700));
}

#[cfg(not(unix))]
fn set_dir_permissions(_path: &Path) {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::Account;

    #[test]
    fn account_round_trips() {
        let dir =
            std::env::temp_dir().join(format!("imagineit-tui-config-test-{}", std::process::id()));
        let path = dir.join("account.json");
        let account = Account {
            server: "http://localhost:4600".to_string(),
            token: "imi_test".to_string(),
            account_id: "acct_123".to_string(),
            identity: Identity::generate(),
        };

        save(&path, &account).unwrap();
        let loaded = load(&path).unwrap().expect("account file");
        assert_eq!(loaded.server, account.server);
        assert_eq!(loaded.token, account.token);
        assert_eq!(loaded.account_id, account.account_id);
        assert_eq!(
            loaded.identity.secret_bytes(),
            account.identity.secret_bytes()
        );

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn worker_key_file_round_trips() {
        let dir = std::env::temp_dir().join(format!(
            "imagineit-tui-worker-key-test-{}",
            std::process::id()
        ));
        let account_path = dir.join("account.json");
        let material = WorkerKeyMaterial {
            worker_id: "gpu-test-01".to_string(),
            key_id: "k1".to_string(),
            key: "c2VjcmV0".to_string(),
        };

        let path = save_worker_key(&account_path, &material).unwrap();
        let stored: serde_json::Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        assert_eq!(stored["worker_id"], "gpu-test-01");
        assert_eq!(stored["key_id"], "k1");
        assert_eq!(stored["key"], "c2VjcmV0");
        assert!(path.ends_with("workers/gpu-test-01.worker.json"));

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = fs::metadata(&path).unwrap().permissions().mode() & 0o777;
            assert_eq!(mode, 0o600);
        }

        let _ = fs::remove_dir_all(dir);
    }
}
