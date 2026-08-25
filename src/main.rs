use indeterminate_messenger::{detect_tamper, generate_shared_seed, wave_decrypt, wave_encrypt};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let key = generate_shared_seed(16);
    let message = std::env::args().nth(1).unwrap_or_else(|| "Quantum breakthrough!".into());
    let t = 1.0;

    let cipher = wave_encrypt(&message, &key, t)?;
    let decrypted = wave_decrypt(&cipher, &key, t)?;
    println!("plain     : {message}");
    println!("cipher B  : {} bytes", cipher.len());
    println!("decrypted : {decrypted}");
    println!("round-trip: {}", if decrypted == message { "ok" } else { "FAIL" });

    let mut tampered = cipher.clone();
    if !tampered.is_empty() {
        tampered[0] ^= 1;
    }
    if detect_tamper(&cipher, &tampered)? {
        println!("tamper    : detected on flipped first byte");
    }
    Ok(())
}
