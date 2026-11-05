//! Account identity and the client half of ImagineIt's E2E1 envelope.
//!
//! The server stores an uncompressed P-256 public key and seals prompts and
//! images to it. The private key never leaves this process except when it is
//! wrapped with the account token and escrowed on the server so the same token
//! can restore it on another machine.

use aes_gcm::{
    aead::{Aead, KeyInit},
    Aes256Gcm, Nonce,
};
use anyhow::{bail, Context, Result};
use base64::{
    engine::general_purpose::{STANDARD, URL_SAFE_NO_PAD},
    Engine,
};
use hkdf::Hkdf;
use p256::{ecdh::diffie_hellman, elliptic_curve::sec1::ToEncodedPoint, PublicKey, SecretKey};
use rand::{rngs::OsRng, RngCore};
use serde_json::json;
use sha2::Sha256;

pub const E2E_MAGIC: &[u8; 4] = b"E2E1";
pub const E2E_INFO: &[u8] = b"imagineit-e2e-v1";
const PUBLIC_KEY_LEN: usize = 65;
const NONCE_LEN: usize = 12;
const TAG_LEN: usize = 16;
const WRAP_SALT: &[u8] = b"imagineit-key-wrap-v1";
const WRAP_INFO: &[u8] = b"private-key";

#[derive(Clone)]
pub struct Identity {
    secret: SecretKey,
}

impl Identity {
    pub fn generate() -> Self {
        Self {
            secret: SecretKey::random(&mut OsRng),
        }
    }

    pub fn from_secret_bytes(bytes: &[u8]) -> Result<Self> {
        let secret =
            SecretKey::from_slice(bytes).context("stored private key is not a P-256 scalar")?;
        Ok(Self { secret })
    }

    pub fn secret_bytes(&self) -> [u8; 32] {
        let bytes = self.secret.to_bytes();
        let mut out = [0u8; 32];
        out.copy_from_slice(&bytes);
        out
    }

    pub fn public_key_b64(&self) -> String {
        let encoded = self.secret.public_key().to_encoded_point(false);
        STANDARD.encode(encoded.as_bytes())
    }

    /// Open an `E2E1` envelope: magic | ephemeral public key | nonce | ciphertext.
    pub fn open(&self, envelope: &[u8]) -> Result<Vec<u8>> {
        if envelope.len() < E2E_MAGIC.len() + PUBLIC_KEY_LEN + NONCE_LEN + TAG_LEN {
            bail!("sealed payload is too short");
        }
        if &envelope[..E2E_MAGIC.len()] != E2E_MAGIC {
            bail!("unknown sealed payload version");
        }

        let mut offset = E2E_MAGIC.len();
        let ephemeral_raw = &envelope[offset..offset + PUBLIC_KEY_LEN];
        offset += PUBLIC_KEY_LEN;
        let nonce = &envelope[offset..offset + NONCE_LEN];
        offset += NONCE_LEN;
        let ciphertext = &envelope[offset..];

        let ephemeral = PublicKey::from_sec1_bytes(ephemeral_raw)
            .context("sealed payload has an invalid ephemeral key")?;
        let shared = diffie_hellman(self.secret.to_nonzero_scalar(), ephemeral.as_affine());

        let hkdf = Hkdf::<Sha256>::new(Some(ephemeral_raw), shared.raw_secret_bytes());
        let mut key = [0u8; 32];
        hkdf.expand(E2E_INFO, &mut key)
            .map_err(|_| anyhow::anyhow!("could not derive the payload key"))?;

        let mut nonce_array = [0u8; NONCE_LEN];
        nonce_array.copy_from_slice(nonce);
        let cipher = Aes256Gcm::new_from_slice(&key).expect("32-byte AES key");
        cipher
            .decrypt(&Nonce::from(nonce_array), ciphertext)
            .map_err(|_| anyhow::anyhow!("sealed payload failed authentication"))
    }
}

/// Wrap our private key as a JWK for the server's key escrow. The derivation
/// matches the web client: HKDF-SHA256(token, salt, info) -> AES-256-GCM.
pub fn wrap_private_key(token: &str, identity: &Identity) -> Result<String> {
    let hkdf = Hkdf::<Sha256>::new(Some(WRAP_SALT), token.as_bytes());
    let mut key = [0u8; 32];
    hkdf.expand(WRAP_INFO, &mut key)
        .map_err(|_| anyhow::anyhow!("could not derive the key-wrap key"))?;

    let point = identity.secret.public_key().to_encoded_point(false);
    let point = point.as_bytes();
    let (x, y) = (&point[1..33], &point[33..65]);
    let jwk = json!({
        "kty": "EC",
        "crv": "P-256",
        "x": URL_SAFE_NO_PAD.encode(x),
        "y": URL_SAFE_NO_PAD.encode(y),
        "d": URL_SAFE_NO_PAD.encode(identity.secret.to_bytes()),
        "key_ops": ["deriveBits"],
        "ext": true,
    });
    let plaintext = serde_json::to_vec(&jwk).context("could not serialize the private key")?;

    let mut nonce = [0u8; NONCE_LEN];
    OsRng.fill_bytes(&mut nonce);
    let cipher = Aes256Gcm::new_from_slice(&key).expect("32-byte AES key");
    let ciphertext = cipher
        .encrypt(&Nonce::from(nonce), plaintext.as_slice())
        .map_err(|_| anyhow::anyhow!("could not wrap the private key"))?;

    let mut blob = Vec::with_capacity(NONCE_LEN + ciphertext.len());
    blob.extend_from_slice(&nonce);
    blob.extend_from_slice(&ciphertext);
    Ok(STANDARD.encode(blob))
}

/// Recover the private key from the server's escrowed, token-wrapped JWK.
pub fn unwrap_private_key(token: &str, wrapped: &str) -> Result<Identity> {
    let blob = STANDARD
        .decode(wrapped.trim())
        .context("wrapped key is not valid base64")?;
    if blob.len() < NONCE_LEN + TAG_LEN {
        bail!("wrapped key is too short");
    }

    let hkdf = Hkdf::<Sha256>::new(Some(WRAP_SALT), token.as_bytes());
    let mut key = [0u8; 32];
    hkdf.expand(WRAP_INFO, &mut key)
        .map_err(|_| anyhow::anyhow!("could not derive the key-wrap key"))?;

    let mut nonce = [0u8; NONCE_LEN];
    nonce.copy_from_slice(&blob[..NONCE_LEN]);
    let cipher = Aes256Gcm::new_from_slice(&key).expect("32-byte AES key");
    let plaintext = cipher
        .decrypt(&Nonce::from(nonce), &blob[NONCE_LEN..])
        .map_err(|_| anyhow::anyhow!("wrapped key could not be opened; the token may not match"))?;

    let jwk: serde_json::Value =
        serde_json::from_slice(&plaintext).context("wrapped key is not a JSON web key")?;
    let encoded_d = jwk
        .get("d")
        .and_then(|value| value.as_str())
        .context("wrapped key has no private scalar")?;
    let d = URL_SAFE_NO_PAD
        .decode(encoded_d.trim_end_matches('='))
        .context("wrapped key scalar is not valid base64url")?;
    Identity::from_secret_bytes(&d)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Generated with the server's app.services.e2e implementation.
    const PRIVATE_KEY_B64: &str = "l5djQGLHrj0SplhOc4yIUM7AyeSpoxfdnDSq9thRGMI=";
    const SEALED_B64: &str = "RTJFMQSom+yZ2mfpigUeURXvB7d2peF/9aSIsmE72uM1w/oucE9Knvqgew2VvopH21rrra+S2dcLyeu/F0/5LPtIE+i2adZD5BQaycDDMgDUON/ToxyioRj4QaPtBcc9foObD2UwlamAbzh1EKX5EJrtbO8iTVgvNIwUA+lG4IYr3k2XdYVJ";
    const WRAPPED_KEY_B64: &str = "bwIfhmRZDvo1jgZm60AVpQyYlaU/3mxZOLoSLCm+WbADFno8HTjfhypXfby3nT/0neUjejpf020JckDBctqT03QoAzxQs9+xY6fADfYg1OGZ1BdcHtsHov63OGgz0/pEpekWwR8qapPLLZGaolZsMncMFxYJW8EoSlrFkduUGCN+pfXoFLcDSf8yTSpBMcfZqJd/PjssyOcfsijNt4CizgNp+mhHF5BOnU5xVVr1UcOREU+BXFQ7zuTcQvYXG6A/KgpnGakLEB2bXet8O0J9jMfilPmwfc3MPM/rxK37ne5wJK/dkZSEf4KVPJB5YBfY";
    const TOKEN: &str = "imi_test_token_1234567890";

    fn identity() -> Identity {
        Identity::from_secret_bytes(&STANDARD.decode(PRIVATE_KEY_B64).unwrap()).unwrap()
    }

    #[test]
    fn opens_a_server_sealed_envelope() {
        let envelope = STANDARD.decode(SEALED_B64).unwrap();
        let plaintext = identity().open(&envelope).unwrap();
        assert_eq!(
            String::from_utf8(plaintext).unwrap(),
            r#"{"prompt":"a tiny teal fox","steps":4}"#
        );
    }

    #[test]
    fn rejects_a_tampered_envelope() {
        let mut envelope = STANDARD.decode(SEALED_B64).unwrap();
        *envelope.last_mut().unwrap() ^= 0xff;
        assert!(identity().open(&envelope).is_err());
    }

    #[test]
    fn unwraps_the_web_client_key_escrow() {
        let restored = unwrap_private_key(TOKEN, WRAPPED_KEY_B64).unwrap();
        assert_eq!(restored.secret_bytes(), identity().secret_bytes());
    }

    #[test]
    fn key_escrow_round_trips() {
        let original = Identity::generate();
        let wrapped = wrap_private_key(TOKEN, &original).unwrap();
        let restored = unwrap_private_key(TOKEN, &wrapped).unwrap();
        assert_eq!(restored.secret_bytes(), original.secret_bytes());
        assert_eq!(restored.public_key_b64(), original.public_key_b64());
        assert!(unwrap_private_key("wrong token", &wrapped).is_err());
    }
}
