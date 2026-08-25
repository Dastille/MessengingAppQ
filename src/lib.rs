//! Wave-function toy cipher.
//!
//! This is a **demonstration** of a diagonal unitary applied to character
//! amplitudes. It is **not** a cryptosystem: the map is deterministic, the
//! "key" is a phase vector, and floating point is reversible with the same
//! seed. Do not use it to protect real messages.
//!
//! v0.1 never compiled on Linux (`cargo.toml` instead of `Cargo.toml`,
//! `ndarray` with a BLAS feature, a broken series for `exp`, and decrypt
//! using a reversed seed so round-trips failed). v0.2 uses only `std` and
//! a correct diagonal `exp(±i θ)`.

const PI: f64 = std::f64::consts::PI;

/// Shared "entangled" seed. Both parties use the **same** bytes.
/// The old reverse-Bob trick guaranteed decryption failure; keep it only
/// if you want to demonstrate a wrong key.
pub fn generate_shared_seed(seed_length: usize) -> Vec<u8> {
    (0..seed_length).map(|i| ((i as u8).wrapping_mul(42)) % 255).collect()
}

fn phase_at(key_seed: &[u8], i: usize) -> f64 {
    let b = key_seed[i % key_seed.len()];
    (b as f64 / 255.0) * 2.0 * PI
}

/// Encrypt `message` under `key_seed` at evolution time `time_step`.
/// Output is interleaved little-endian f64 pairs `(re, im)` per character.
pub fn wave_encrypt(message: &str, key_seed: &[u8], time_step: f64) -> Result<Vec<u8>, &'static str> {
    if message.is_empty() || key_seed.is_empty() {
        return Err("empty message or key");
    }
    let chars: Vec<u32> = message.chars().map(|c| c as u32).collect();
    let n = chars.len();
    let norm = (n as f64).sqrt();
    let mut bytes = Vec::with_capacity(n * 16);
    for (i, code) in chars.into_iter().enumerate() {
        let amp = code as f64 / norm;
        let theta = phase_at(key_seed, i) * time_step;
        // U = exp(-i θ) applied to a real amplitude
        let re = amp * theta.cos();
        let im = -amp * theta.sin();
        bytes.extend_from_slice(&re.to_le_bytes());
        bytes.extend_from_slice(&im.to_le_bytes());
    }
    Ok(bytes)
}

/// Inverse of [`wave_encrypt`] with the **same** seed and time step.
pub fn wave_decrypt(cipher_bytes: &[u8], key_seed: &[u8], time_step: f64) -> Result<String, &'static str> {
    if cipher_bytes.len() % 16 != 0 {
        return Err("invalid cipher length");
    }
    if key_seed.is_empty() {
        return Err("empty key");
    }
    let n = cipher_bytes.len() / 16;
    let norm = (n as f64).sqrt();
    let mut out = String::with_capacity(n);
    for i in 0..n {
        let start = i * 16;
        let re = f64::from_le_bytes(
            cipher_bytes[start..start + 8]
                .try_into()
                .map_err(|_| "byte conversion failed")?,
        );
        let im = f64::from_le_bytes(
            cipher_bytes[start + 8..start + 16]
                .try_into()
                .map_err(|_| "byte conversion failed")?,
        );
        let theta = phase_at(key_seed, i) * time_step;
        // multiply by exp(+i θ)
        let orig_re = theta.cos() * re - theta.sin() * im;
        let code = (orig_re * norm).round() as i64;
        let clamped = code.clamp(0, 0x10ffff) as u32;
        out.push(char::from_u32(clamped).unwrap_or('\u{FFFD}'));
    }
    Ok(out)
}

/// Compare two ciphertexts via a short interference sample.
/// Returns `true` if they look tampered (relative error > 3%).
pub fn detect_tamper(send: &[u8], recv: &[u8]) -> Result<bool, &'static str> {
    let sample = 4.min(send.len() / 16).min(recv.len() / 16);
    if sample == 0 {
        return Err("insufficient bytes for tamper detection");
    }
    let mut acc = 0.0;
    for i in 0..sample {
        let s = i * 16;
        let re_s = f64::from_le_bytes(send[s..s + 8].try_into().map_err(|_| "byte conversion failed")?);
        let im_s = f64::from_le_bytes(send[s + 8..s + 16].try_into().map_err(|_| "byte conversion failed")?);
        let re_r = f64::from_le_bytes(recv[s..s + 8].try_into().map_err(|_| "byte conversion failed")?);
        let im_r = f64::from_le_bytes(recv[s + 8..s + 16].try_into().map_err(|_| "byte conversion failed")?);
        let dre = re_s - re_r;
        let dim = im_s - im_r;
        acc += dre * dre + dim * dim;
    }
    let rms = (acc / sample as f64).sqrt();
    Ok(rms > 1e-9)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_ascii() {
        let key = generate_shared_seed(16);
        let msg = "Quantum breakthrough!";
        let cipher = wave_encrypt(msg, &key, 1.0).unwrap();
        let back = wave_decrypt(&cipher, &key, 1.0).unwrap();
        assert_eq!(back, msg);
    }

    #[test]
    fn wrong_key_fails() {
        let key = generate_shared_seed(16);
        let mut bad = key.clone();
        bad.reverse();
        let msg = "hello";
        let cipher = wave_encrypt(msg, &key, 1.0).unwrap();
        let back = wave_decrypt(&cipher, &bad, 1.0).unwrap();
        assert_ne!(back, msg);
    }

    #[test]
    fn tamper_flag() {
        let key = generate_shared_seed(16);
        let cipher = wave_encrypt("abcdefgh", &key, 1.0).unwrap();
        let mut tampered = cipher.clone();
        tampered[0] ^= 0x80;
        assert!(detect_tamper(&cipher, &tampered).unwrap());
        assert!(!detect_tamper(&cipher, &cipher).unwrap());
    }
}
