use nalgebra::{Complex, DMatrix, Matrix};
use ndarray::Array1;
use std::f64::consts::PI;

// Generate correlated seeds simulating entanglement (replace with sensor-based entropy in production)
fn generate_entangled_seed(seed_length: usize) -> (Vec<u8>, Vec<u8>) {
    let mut alice_seed = vec![0u8; seed_length];
    for i in 0..seed_length {
        alice_seed[i] = ((i as u8) * 42) % 255; // Fixed for demo; use rand crate later
    }
    let mut bob_seed = alice_seed.clone();
    bob_seed.reverse(); // Simple correlation for testing
    (alice_seed, bob_seed)
}

// Encrypt message by simulating wave function evolution
fn wave_encrypt(message: &str, key_seed: &[u8], time_step: f64) -> Result<Vec<u8>, &'static str> {
    if message.is_empty() || key_seed.is_empty() {
        return Err("Empty message or key");
    }

    // Convert message to complex wave vector, normalized
    let psi_vec: Vec<Complex<f64>> = message.chars().map(|c| Complex::new(c as u32 as f64, 0.0)).collect();
    let len = psi_vec.len();
    let norm = (len as f64).sqrt();
    let psi = Array1::from_vec(psi_vec).map(|z| *z / norm);

    // Key-derived phases for Hamiltonian
    let mut phases: Vec<f64> = key_seed.iter().map(|&b| (b as f64 / 255.0) * 2.0 * PI).collect();
    phases.resize(len, 0.0);
    let h = DMatrix::from_diagonal(&DMatrix::from_vec(len, 1, phases).map(|x| Complex::new(x, 0.0)));

    // Evolve: U = exp(-i H t)
    let neg_i_ht = h * Complex::new(0.0, -1.0) * time_step;
    let u = matrix_exp(&neg_i_ht).map_err(|_| "Matrix exp failed")?;

    // Cipher wave = U * psi
    let psi_mat = DMatrix::from_vec(len, 1, psi.to_vec());
    let cipher_wave = u * psi_mat;

    // Serialize to bytes (re and im as f64)
    let mut bytes = Vec::with_capacity(len * 16);
    for c in cipher_wave.iter() {
        bytes.extend_from_slice(&c.re.to_le_bytes());
        bytes.extend_from_slice(&c.im.to_le_bytes());
    }
    Ok(bytes)
}

// Decrypt by reversing wave evolution
fn wave_decrypt(cipher_bytes: &[u8], key_seed: &[u8], time_step: f64) -> Result<String, &'static str> {
    if cipher_bytes.len() % 16 != 0 {
        return Err("Invalid cipher length");
    }
    let len = cipher_bytes.len() / 16;

    // Deserialize cipher_wave
    let mut cipher_wave_vec = Vec::with_capacity(len);
    for i in 0..len {
        let start = i * 16;
        let re = f64::from_le_bytes(cipher_bytes[start..start+8].try_into().map_err(|_| "Byte conversion failed")?);
        let im = f64::from_le_bytes(cipher_bytes[start+8..start+16].try_into().map_err(|_| "Byte conversion failed")?);
        cipher_wave_vec.push(Complex::new(re, im));
    }
    let cipher_wave = DMatrix::from_vec(len, 1, cipher_wave_vec);

    // Reverse Hamiltonian
    let mut phases: Vec<f64> = key_seed.iter().map(|&b| (b as f64 / 255.0) * 2.0 * PI).collect();
    phases.resize(len, 0.0);
    let h = DMatrix::from_diagonal(&DMatrix::from_vec(len, 1, phases).map(|x| Complex::new(x, 0.0)));

    // Inverse: U_inv = exp(i H t)
    let i_ht = h * Complex::new(0.0, 1.0) * time_step;
    let u_inv = matrix_exp(&i_ht).map_err(|_| "Matrix exp failed")?;

    // psi = U_inv * cipher_wave
    let psi = u_inv * cipher_wave;

    // Recover message
    let recovered: String = psi.iter()
        .map(|z| ((z.re.round() as u32 % 256) as u8 as char))
        .collect();
    Ok(recovered)
}

// Detect tampering via interference norm
fn detect_tamper(test_send_bytes: &[u8], test_recv_bytes: &[u8]) -> Result<bool, &'static str> {
    let sample_size = 4;
    let byte_size = sample_size * 16;
    if test_send_bytes.len() < byte_size || test_recv_bytes.len() < byte_size {
        return Err("Insufficient bytes for tamper detection");
    }

    let mut test_send = Vec::with_capacity(sample_size);
    let mut test_recv = Vec::with_capacity(sample_size);
    for i in 0..sample_size {
        let start = i * 16;
        let re = f64::from_le_bytes(test_send_bytes[start..start+8].try_into().map_err(|_| "Byte conversion failed")?);
        let im = f64::from_le_bytes(test_send_bytes[start+8..start+16].try_into().map_err(|_| "Byte conversion failed")?);
        test_send.push(Complex::new(re, im));

        let re = f64::from_le_bytes(test_recv_bytes[start..start+8].try_into().map_err(|_| "Byte conversion failed")?);
        let im = f64::from_le_bytes(test_recv_bytes[start+8..start+16].try_into().map_err(|_| "Byte conversion failed")?);
        test_recv.push(Complex::new(re, im));
    }

    let test_send_arr = Array1::from_vec(test_send);
    let test_recv_arr = Array1::from_vec(test_recv);
    let interf = &test_send_arr + &test_recv_arr;
    let interf_norm = interf.iter().fold(0.0, |acc, x| acc + x.norm_squared()).sqrt();
    let expected = (2.0 * sample_size as f64).sqrt();
    let error = (interf_norm - expected).abs() / expected;
    Ok(error > 0.03)
}

// Matrix exponential (simplified, assumes nalgebra support)
fn matrix_exp(mat: &DMatrix<Complex<f64>>) -> Result<DMatrix<Complex<f64>>, &'static str> {
    // In practice, use nalgebra's exp or external lib like sprs for efficiency
    // Placeholder: Compute via series (not production-ready)
    let n = mat.nrows();
    let mut result = DMatrix::identity(n, n);
    let mut term = DMatrix::identity(n, n);
    let mut factorial = 1.0;
    for i in 1..10 { // Limited terms for demo
        term = term * mat / i as f64;
        factorial *= i as f64;
        result += &term / factorial;
    }
    Ok(result)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (alice_seed, bob_seed) = generate_entangled_seed(16);
    let message = "Quantum breakthrough!";
    let cipher = wave_encrypt(message, &alice_seed, 1.0)?;
    let decrypted = wave_decrypt(&cipher, &bob_seed, 1.0)?;
    println!("Decrypted: {}", decrypted);

    let mut tampered = cipher.clone();
    tampered[0] ^= 1;
    if detect_tamper(&cipher, &tampered)? {
        println!("Tampering detected!");
    }
    Ok(())
}
