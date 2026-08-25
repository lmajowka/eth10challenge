//! Ethereum address helpers: Keccak-256, target parsing, EIP-55 formatting, and
//! the CPU reference derivation (BIP-39 seed -> m/44'/60'/0'/0/0 -> address).
//!
//! This is also the reference the GPU is checked against in `--selftest`.

use anyhow::{bail, Context, Result};
use bitcoin::bip32::{DerivationPath, Xpriv};
use bitcoin::secp256k1::{Secp256k1, Signing};
use bitcoin::Network;
use sha3::{Digest, Keccak256};

/// BIP-44 path for the first account of the default Ethereum wallet (MetaMask,
/// Ledger Live, Trust, ...). Coin type 60 is the only difference from Bitcoin.
pub const ETH_PATH: &str = "m/44'/60'/0'/0/0";

/// Ethereum's hash. Note this is original Keccak, *not* NIST SHA-3 — they differ
/// only in a padding byte, but every digest differs.
pub fn keccak256(data: &[u8]) -> [u8; 32] {
    let mut h = Keccak256::new();
    h.update(data);
    h.finalize().into()
}

/// Parses a target address: 40 hex characters, `0x` prefix optional.
///
/// A mixed-case input carries an EIP-55 checksum, so it is verified rather than
/// merely lowercased. Silently accepting a typo would send an exhaustive search
/// after an address that no mnemonic can produce, and it would run to completion
/// before saying so.
pub fn parse_address(s: &str) -> Result<[u8; 20]> {
    let body = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")).unwrap_or(s);
    if body.len() != 40 || !body.chars().all(|c| c.is_ascii_hexdigit()) {
        bail!("target must be a 40-character hex Ethereum address (got {s:?})");
    }

    let bytes = hex::decode(body).context("decoding target address")?;
    let mut addr = [0u8; 20];
    addr.copy_from_slice(&bytes);

    let has_upper = body.chars().any(|c| c.is_ascii_uppercase());
    let has_lower = body.chars().any(|c| c.is_ascii_lowercase());
    if has_upper && has_lower {
        let want = to_eip55(&addr);
        // Compare only the body: the caller's `0x` prefix casing is irrelevant.
        if want[2..] != *body {
            bail!("EIP-55 checksum mismatch: {s} is not a valid address (did you mean {want}?)");
        }
    }
    Ok(addr)
}

/// Formats an address with the EIP-55 mixed-case checksum.
pub fn to_eip55(addr: &[u8; 20]) -> String {
    let lower = hex::encode(addr);
    let hash = keccak256(lower.as_bytes());
    let mut out = String::with_capacity(42);
    out.push_str("0x");
    for (i, c) in lower.chars().enumerate() {
        // Nibble i of the hash decides the case of hex character i.
        let nibble = if i % 2 == 0 { hash[i / 2] >> 4 } else { hash[i / 2] & 0x0f };
        if c.is_ascii_alphabetic() && nibble >= 8 {
            out.push(c.to_ascii_uppercase());
        } else {
            out.push(c);
        }
    }
    out
}

/// 64-byte BIP-39 seed -> 20-byte Ethereum address at [`ETH_PATH`].
///
/// The address is the low 20 bytes of `keccak256(X || Y)`, where X||Y is the
/// uncompressed public key *without* its `0x04` prefix.
pub fn address_from_seed<C: Signing>(
    secp: &Secp256k1<C>,
    path: &DerivationPath,
    seed: &[u8; 64],
) -> Result<[u8; 20]> {
    // The network only selects xprv version bytes; derived keys are identical.
    let master = Xpriv::new_master(Network::Bitcoin, seed)?;
    let child = master.derive_priv(secp, path)?;
    let pubkey = child.private_key.public_key(secp);
    let uncompressed = pubkey.serialize_uncompressed(); // 0x04 || X || Y
    let hash = keccak256(&uncompressed[1..]);
    let mut addr = [0u8; 20];
    addr.copy_from_slice(&hash[12..]);
    Ok(addr)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Vectors below were produced by an independent pure-Python implementation
    // (hand-rolled Keccak-f[1600] and secp256k1), not by this code path.
    const ABANDON: &str = "abandon abandon abandon abandon abandon abandon \
                           abandon abandon abandon abandon abandon about";

    #[test]
    fn keccak_known_answers() {
        assert_eq!(
            hex::encode(keccak256(b"")),
            "c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470"
        );
        assert_eq!(
            hex::encode(keccak256(b"abc")),
            "4e03657aea45a94fc7d47ba826c8d667c0d1e6e33a64a036ec44f58fa12d6c45"
        );
    }

    #[test]
    fn eip55_round_trip() {
        // The four canonical EIP-55 examples.
        for a in [
            "0x5aAeb6053F3E94C9b9A09f33669435E7Ef1BeAed",
            "0xfB6916095ca1df60bB79Ce92cE3Ea74c37c5d359",
            "0xdbF03B407c01E7cD3CBea99509d93f8DDDC8C6FB",
            "0xD1220A0cf47c7B9Be7A2E6BA89F429762e7b9aDb",
        ] {
            let parsed = parse_address(a).unwrap();
            assert_eq!(to_eip55(&parsed), a);
        }
    }

    #[test]
    fn parse_accepts_any_uniform_case_and_rejects_bad_checksum() {
        let want = parse_address("0x5aAeb6053F3E94C9b9A09f33669435E7Ef1BeAed").unwrap();
        // All-lower, all-upper and no-prefix carry no checksum, so all are accepted.
        assert_eq!(parse_address("0x5aaeb6053f3e94c9b9a09f33669435e7ef1beaed").unwrap(), want);
        assert_eq!(parse_address("0X5AAEB6053F3E94C9B9A09F33669435E7EF1BEAED").unwrap(), want);
        assert_eq!(parse_address("5aaeb6053f3e94c9b9a09f33669435e7ef1beaed").unwrap(), want);
        // Mixed case with a flipped letter must be rejected.
        assert!(parse_address("0x5aAeb6053F3E94C9b9A09f33669435E7Ef1BeAeD").is_err());
        assert!(parse_address("0xdeadbeef").is_err());
    }

    #[test]
    fn seed_to_address_matches_independent_reference() {
        use bip39::{Language, Mnemonic};
        let secp = Secp256k1::new();
        let path: DerivationPath = ETH_PATH.parse().unwrap();
        for (phrase, want) in [
            (ABANDON, "0x9858EfFD232B4033E47d90003D41EC34EcaEda94"),
            (
                "legal winner thank year wave sausage worth useful legal winner thank yellow",
                "0x58A57ed9d8d624cBD12e2C467D34787555bB1b25",
            ),
            (
                "letter advice cage absurd amount doctor acoustic avoid letter advice cage above",
                "0x3061750d3dF69ef7B8d4407CB7f3F879Fd9d2398",
            ),
            (
                "zoo zoo zoo zoo zoo zoo zoo zoo zoo zoo zoo wrong",
                "0xfc2077CA7F403cBECA41B1B0F62D91B5EA631B5E",
            ),
        ] {
            let m = Mnemonic::parse_in_normalized(Language::English, phrase).unwrap();
            let addr = address_from_seed(&secp, &path, &m.to_seed("")).unwrap();
            assert_eq!(to_eip55(&addr), want, "phrase: {phrase}");
        }
    }
}
