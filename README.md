# MessengingAppQ

A **research prototype** of a wave-function toy cipher: each character becomes a complex amplitude, a key-derived diagonal Hamiltonian evolves it, and the inverse recovers the text. Tamper detection is a residual on the ciphertext.

This is **not encryption**. It is a lab demo. Do not send secrets through it.

## What was broken (v0.1)

| Bug | Effect |
| --- | --- |
| File named `cargo.toml` | Cargo on Linux/macOS looks for `Cargo.toml` and ignores the crate |
| `ndarray` with `blas` | Needs a system BLAS; the crate would not build in a clean environment |
| Series `exp` divided by factorial twice | Unitary was garbage |
| Decrypt used a **reversed** seed | Round-trip always failed |
| Amplitudes never un-normalized | Recovered characters were wrong even with the right key |
| Unused `Matrix` import | Dead code |

v0.2 is `std`-only, `Cargo.toml` is correctly named, encrypt/decrypt share one seed, and `cargo test` covers round-trip, wrong-key, and tamper.

## Build

```sh
cargo test
cargo run -- "hello from the wave"
```

No extra crates. Edition 2021.

## How the toy works

For character `i` with code `c`, length `n`, key byte `k`:

```
amp   = c / sqrt(n)
θ     = (k/255) · 2π · t
cipher = amp · exp(-i θ)
plain  = Re( cipher · exp(+i θ) ) · sqrt(n)
```

Because `H` is diagonal, `exp(±i H t)` is a per-component multiply. The old dense-matrix series was unnecessary and wrong.

## License

[AGPL-3.0](LICENSE)
