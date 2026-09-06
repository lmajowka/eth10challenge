//! Host side of the CUDA port: context/module management and the per-primitive
//! selftest harness that verifies each device function against the CPU crates.

use anyhow::{Context as _, Result};
use cust::context::Context;
use cust::launch;
use cust::memory::{CopyDestination, DeviceBuffer};
use cust::module::Module;
use cust::stream::{Stream, StreamFlags};

/// PTX emitted by `build.rs` (nvcc compiling `src/cuda/kernels.cu`).
const PTX: &str = include_str!(env!("KERNELS_PTX"));

/// Owns the CUDA context + loaded module for the lifetime of a run.
pub struct Gpu {
    _context: Context,
    module: Module,
    stream: Stream,
}

impl Gpu {
    pub fn new() -> Result<Self> {
        let _context = cust::quick_init().context("CUDA init failed (no device / driver?)")?;
        let module = Module::from_ptx(PTX, &[]).context("loading kernels PTX")?;
        let stream =
            Stream::new(StreamFlags::NON_BLOCKING, None).context("creating CUDA stream")?;

        // Build the fixed-base window table of multiples of G. Every kernel that
        // derives a public key reads it, so it must run before anything else.
        let init = module
            .get_function("k_init_gtable")
            .context("loading k_init_gtable")?;
        unsafe {
            launch!(init<<<(64 * 15 + 255) / 256, 256, 0, stream>>>())
                .context("launching k_init_gtable")?;
        }
        stream.synchronize().context("building G table")?;

        Ok(Self {
            _context,
            module,
            stream,
        })
    }

    /// Runs a one-message-per-thread hash kernel over `inputs`, returning a
    /// `digest_len`-byte digest per input. The kernel signature must be
    /// `(const u8* msgs, const u32* lens, u32 stride, u8* out, u32 n)`.
    fn hash_batch(&self, kernel: &str, inputs: &[Vec<u8>], digest_len: usize) -> Result<Vec<Vec<u8>>> {
        let n = inputs.len();
        let (packed, lens, stride) = pack(inputs);

        let d_msgs = DeviceBuffer::from_slice(&packed)?;
        let d_lens = DeviceBuffer::from_slice(&lens)?;
        let d_out = DeviceBuffer::from_slice(&vec![0u8; n * digest_len])?;

        let func = self.module.get_function(kernel)?;
        let (grid, block) = launch_dims(n);
        let stream = &self.stream;
        unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                d_msgs.as_device_ptr(),
                d_lens.as_device_ptr(),
                stride as u32,
                d_out.as_device_ptr(),
                n as u32
            ))?;
        }
        stream.synchronize()?;

        let mut out = vec![0u8; n * digest_len];
        d_out.copy_to(&mut out)?;
        Ok(out.chunks(digest_len).map(|c| c.to_vec()).collect())
    }

    /// One HMAC-SHA512 (64-byte output) per (key, msg) pair.
    fn hmac_batch(&self, keys: &[Vec<u8>], msgs: &[Vec<u8>]) -> Result<Vec<Vec<u8>>> {
        let n = keys.len();
        let (pk, klens, kstride) = pack(keys);
        let (pm, mlens, mstride) = pack(msgs);

        let d_keys = DeviceBuffer::from_slice(&pk)?;
        let d_klens = DeviceBuffer::from_slice(&klens)?;
        let d_msgs = DeviceBuffer::from_slice(&pm)?;
        let d_mlens = DeviceBuffer::from_slice(&mlens)?;
        let d_out = DeviceBuffer::from_slice(&vec![0u8; n * 64])?;

        let func = self.module.get_function("k_hmac_sha512")?;
        let (grid, block) = launch_dims(n);
        let stream = &self.stream;
        unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                d_keys.as_device_ptr(), d_klens.as_device_ptr(), kstride as u32,
                d_msgs.as_device_ptr(), d_mlens.as_device_ptr(), mstride as u32,
                d_out.as_device_ptr(), n as u32
            ))?;
        }
        stream.synchronize()?;
        let mut out = vec![0u8; n * 64];
        d_out.copy_to(&mut out)?;
        Ok(out.chunks(64).map(|c| c.to_vec()).collect())
    }

    /// One compressed (33-byte) public key per 32-byte big-endian private key.
    fn pubkey_batch(&self, privs: &[[u8; 32]]) -> Result<Vec<[u8; 33]>> {
        let n = privs.len();
        let flat: Vec<u8> = privs.iter().flatten().copied().collect();
        let d_priv = DeviceBuffer::from_slice(&flat)?;
        let d_out = DeviceBuffer::from_slice(&vec![0u8; n * 33])?;

        let func = self.module.get_function("k_pubkey")?;
        let (grid, block) = launch_dims(n);
        let stream = &self.stream;
        unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                d_priv.as_device_ptr(), d_out.as_device_ptr(), n as u32
            ))?;
        }
        stream.synchronize()?;
        let mut out = vec![0u8; n * 33];
        d_out.copy_to(&mut out)?;
        Ok(out.chunks(33).map(|c| c.try_into().unwrap()).collect())
    }

    /// One 64-byte uncompressed public key (X||Y) per 32-byte private key.
    fn pubkey_xy_batch(&self, privs: &[[u8; 32]]) -> Result<Vec<[u8; 64]>> {
        let n = privs.len();
        let flat: Vec<u8> = privs.iter().flatten().copied().collect();
        let d_priv = DeviceBuffer::from_slice(&flat)?;
        let d_out = DeviceBuffer::from_slice(&vec![0u8; n * 64])?;

        let func = self.module.get_function("k_pubkey_xy")?;
        let (grid, block) = launch_dims(n);
        let stream = &self.stream;
        unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                d_priv.as_device_ptr(), d_out.as_device_ptr(), n as u32
            ))?;
        }
        stream.synchronize()?;
        let mut out = vec![0u8; n * 64];
        d_out.copy_to(&mut out)?;
        Ok(out.chunks(64).map(|c| c.try_into().unwrap()).collect())
    }

    /// One 20-byte Ethereum address (path m/44'/60'/0'/0/0) per 64-byte seed.
    fn seed_to_eth_batch(&self, seeds: &[[u8; 64]]) -> Result<Vec<[u8; 20]>> {
        let n = seeds.len();
        let flat: Vec<u8> = seeds.iter().flatten().copied().collect();
        let d_seed = DeviceBuffer::from_slice(&flat)?;
        let d_out = DeviceBuffer::from_slice(&vec![0u8; n * 20])?;

        let func = self.module.get_function("k_seed_to_eth")?;
        let (grid, block) = launch_dims(n);
        let stream = &self.stream;
        unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                d_seed.as_device_ptr(), d_out.as_device_ptr(), n as u32
            ))?;
        }
        stream.synchronize()?;
        let mut out = vec![0u8; n * 20];
        d_out.copy_to(&mut out)?;
        Ok(out.chunks(20).map(|c| c.try_into().unwrap()).collect())
    }

    /// One (a + b) mod n per pair, all 32-byte big-endian.
    fn scalar_add_batch(&self, a: &[[u8; 32]], b: &[[u8; 32]]) -> Result<Vec<[u8; 32]>> {
        let n = a.len();
        let fa: Vec<u8> = a.iter().flatten().copied().collect();
        let fb: Vec<u8> = b.iter().flatten().copied().collect();
        let d_a = DeviceBuffer::from_slice(&fa)?;
        let d_b = DeviceBuffer::from_slice(&fb)?;
        let d_out = DeviceBuffer::from_slice(&vec![0u8; n * 32])?;

        let func = self.module.get_function("k_scalar_add")?;
        let (grid, block) = launch_dims(n);
        let stream = &self.stream;
        unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                d_a.as_device_ptr(), d_b.as_device_ptr(), d_out.as_device_ptr(), n as u32
            ))?;
        }
        stream.synchronize()?;
        let mut out = vec![0u8; n * 32];
        d_out.copy_to(&mut out)?;
        Ok(out.chunks(32).map(|c| c.try_into().unwrap()).collect())
    }

    /// One PBKDF2-HMAC-SHA512 (dkLen=64) per (password, salt) pair.
    fn pbkdf2_batch(&self, pws: &[Vec<u8>], salts: &[Vec<u8>], iters: u32) -> Result<Vec<Vec<u8>>> {
        let n = pws.len();
        let (pp, pwlens, pwstride) = pack(pws);
        let (ps, slens, sstride) = pack(salts);

        let d_pw = DeviceBuffer::from_slice(&pp)?;
        let d_pwlens = DeviceBuffer::from_slice(&pwlens)?;
        let d_salt = DeviceBuffer::from_slice(&ps)?;
        let d_slens = DeviceBuffer::from_slice(&slens)?;
        let d_out = DeviceBuffer::from_slice(&vec![0u8; n * 64])?;

        let func = self.module.get_function("k_pbkdf2")?;
        let (grid, block) = launch_dims(n);
        let stream = &self.stream;
        unsafe {
            launch!(func<<<grid, block, 0, stream>>>(
                d_pw.as_device_ptr(), d_pwlens.as_device_ptr(), pwstride as u32,
                d_salt.as_device_ptr(), d_slens.as_device_ptr(), sstride as u32,
                iters, d_out.as_device_ptr(), n as u32
            ))?;
        }
        stream.synchronize()?;
        let mut out = vec![0u8; n * 64];
        d_out.copy_to(&mut out)?;
        Ok(out.chunks(64).map(|c| c.to_vec()).collect())
    }
}

/// Packs variable-length byte vectors into a fixed-stride buffer plus lengths.
/// Returns (packed, lens, stride). Stride is at least 1 so device pointers stay valid.
fn pack(inputs: &[Vec<u8>]) -> (Vec<u8>, Vec<u32>, usize) {
    let n = inputs.len();
    let stride = inputs.iter().map(|m| m.len()).max().unwrap_or(0).max(1);
    let mut packed = vec![0u8; n * stride];
    let mut lens = vec![0u32; n];
    for (i, m) in inputs.iter().enumerate() {
        packed[i * stride..i * stride + m.len()].copy_from_slice(m);
        lens[i] = m.len() as u32;
    }
    (packed, lens, stride)
}

fn launch_dims(n: usize) -> (u32, u32) {
    let block = 256u32;
    let grid = ((n as u32) + block - 1) / block;
    (grid.max(1), block)
}

/// A wordlist prepared for the GPU: NFKD bytes packed at a fixed stride plus a
/// per-word byte length. BIP-39 wordlists are already NFKD-normalized, so the
/// canonical word strings can be used verbatim.
pub struct GpuWordlist {
    packed: Vec<u8>,
    lens: Vec<u8>,
    stride: usize,
}

impl GpuWordlist {
    pub fn new(words: &[&str]) -> Result<Self> {
        let stride = words.iter().map(|w| w.len()).max().unwrap_or(1).max(1);
        // Kernel's mnemonic buffer is 512 bytes: 12 words + 11 spaces must fit.
        anyhow::ensure!(
            stride * 12 + 11 <= 512,
            "wordlist word too long for GPU mnemonic buffer (stride {stride})"
        );
        anyhow::ensure!(stride < 256, "word length exceeds u8 length field");
        let mut packed = vec![0u8; words.len() * stride];
        let mut lens = vec![0u8; words.len()];
        for (i, w) in words.iter().enumerate() {
            let b = w.as_bytes();
            packed[i * stride..i * stride + b.len()].copy_from_slice(b);
            lens[i] = b.len() as u8;
        }
        Ok(Self { packed, lens, stride })
    }
}

/// Result of a successful GPU search: the global candidate index and its 12
/// word indices.
pub struct SearchHit {
    pub global_index: usize,
    pub indices: [u16; 12],
}

impl Gpu {
    /// Searches `candidates` (an iterator of 12 word-index arrays) for one whose
    /// derived Ethereum address equals `target`. Streams in batches so memory
    /// stays flat.
    pub fn search(
        &self,
        candidates: impl Iterator<Item = [u16; 12]> + Send + 'static,
        wordlist: &GpuWordlist,
        target: &[u8; 20],
        batch_size: usize,
        checksum: bool,
    ) -> Result<Option<SearchHit>> {
        let d_wordlist = DeviceBuffer::from_slice(&wordlist.packed)?;
        let d_lens = DeviceBuffer::from_slice(&wordlist.lens)?;
        let d_target = DeviceBuffer::from_slice(target)?;
        let filter = self.module.get_function("k_filter")?;
        let pipeline = self.module.get_function("k_pipeline")?;

        // All device buffers are allocated once and reused. Allocating inside
        // the loop cost four cudaMalloc/cudaFree pairs per batch, each of which
        // implicitly synchronizes the device.
        let d_survivors = unsafe { DeviceBuffer::<u32>::uninitialized(batch_size)? };
        let d_cand = unsafe { DeviceBuffer::<u16>::uninitialized(batch_size * 12)? };
        let mut d_counter = DeviceBuffer::from_slice(&[0u32])?;
        let d_found_flag = DeviceBuffer::from_slice(&[0u32])?;
        let d_found_idx = DeviceBuffer::from_slice(&[0u32])?;

        // Generating a batch takes ~10-15% of the time the GPU spends on it, and
        // it used to run between launches with the device idle. A producer thread
        // fills the next batch while the current one is in flight; buffers cycle
        // back over `empty` so nothing is reallocated.
        let (full_tx, full_rx) = std::sync::mpsc::sync_channel::<Vec<u16>>(1);
        let (empty_tx, empty_rx) = std::sync::mpsc::channel::<Vec<u16>>();
        for _ in 0..2 {
            let _ = empty_tx.send(Vec::with_capacity(batch_size * 12));
        }
        let producer = std::thread::spawn(move || {
            let mut it = candidates;
            while let Ok(mut buf) = empty_rx.recv() {
                buf.clear();
                for _ in 0..batch_size {
                    match it.next() {
                        Some(c) => buf.extend_from_slice(&c),
                        None => break,
                    }
                }
                let last = buf.len() < batch_size * 12;
                // A send error just means the consumer stopped (hit found).
                if full_tx.send(buf).is_err() || last {
                    return;
                }
            }
        });

        let mut batch_start: usize = 0;
        let mut checked: usize = 0;
        let block = 64u32;
        let stream = &self.stream;

        let result = loop {
            let buf = match full_rx.recv() {
                Ok(b) => b,
                Err(_) => break None, // producer finished
            };
            if buf.is_empty() {
                break None;
            }
            let n = buf.len() / 12;

            d_cand.index(0..buf.len()).copy_from(&buf[..])?;
            d_counter.copy_from(&[0u32])?;

            // Pass 1: checksum filter -> compacted survivors.
            let grid = ((n as u32) + block - 1) / block;
            unsafe {
                launch!(filter<<<grid, block, 0, stream>>>(
                    d_cand.as_device_ptr(), n as u32,
                    d_survivors.as_device_ptr(), d_counter.as_device_ptr(),
                    checksum as u32
                ))?;
            }
            stream.synchronize()?;
            let mut counter = [0u32];
            d_counter.copy_to(&mut counter)?;
            let count = counter[0];

            // Pass 2: heavy derivation over survivors only.
            if count > 0 {
                let grid2 = (count + block - 1) / block;
                unsafe {
                    launch!(pipeline<<<grid2, block, 0, stream>>>(
                        d_cand.as_device_ptr(),
                        d_survivors.as_device_ptr(),
                        count,
                        d_wordlist.as_device_ptr(),
                        d_lens.as_device_ptr(),
                        wordlist.stride as u32,
                        d_target.as_device_ptr(),
                        d_found_flag.as_device_ptr(),
                        d_found_idx.as_device_ptr()
                    ))?;
                }
                stream.synchronize()?;
            }

            let mut found_flag = [0u32];
            d_found_flag.copy_to(&mut found_flag)?;
            if found_flag[0] != 0 {
                let mut found_idx = [0u32];
                d_found_idx.copy_to(&mut found_idx)?;
                let local = found_idx[0] as usize;
                let mut indices = [0u16; 12];
                indices.copy_from_slice(&buf[local * 12..local * 12 + 12]);
                break Some(SearchHit {
                    global_index: batch_start + local,
                    indices,
                });
            }

            checked += n;
            batch_start += n;
            let short_batch = n < batch_size;
            let _ = empty_tx.send(buf);
            println!("Checked {} candidates...", crate::format_number(checked));
            use std::io::Write;
            let _ = std::io::stdout().flush();
            if short_batch {
                break None; // last (partial) batch
            }
        };

        // Break out of the loop early (a hit) and the producer may be parked in
        // either direction, so both of its peers have to go before the join:
        // dropping the receiver fails its pending send, dropping the sender ends
        // its wait for the next empty buffer.
        drop(full_rx);
        drop(empty_tx);
        let _ = producer.join();
        Ok(result)
    }
}

/// Runs all primitive selftests, printing PASS/FAIL per primitive. Returns
/// `Ok(true)` iff every check passed.
pub fn run_selftest() -> Result<bool> {
    use bitcoin::hashes::{sha256, sha512, Hash};

    let gpu = Gpu::new()?;
    let mut all_ok = true;

    // Messages chosen to exercise: empty, short, multi-block boundaries.
    let mut msgs: Vec<Vec<u8>> = vec![
        vec![],
        b"abc".to_vec(),
        b"message digest".to_vec(),
        b"The quick brown fox jumps over the lazy dog".to_vec(),
        vec![0x61u8; 55], // one-block-1-byte boundary for sha256
    ];
    msgs.push(vec![0x5au8; 64]);
    msgs.push(vec![0xa5u8; 119]);
    // Keccak's rate is 136 bytes, so straddle it in both directions: 135 leaves
    // exactly one byte of padding (where the two pad marks collide into 0x81),
    // 136 forces an entire extra padding block, and 137 spills into a second.
    msgs.push(vec![0x61u8; 135]);
    msgs.push(vec![0x61u8; 136]);
    msgs.push(vec![0x61u8; 137]);
    msgs.push(vec![0u8; 200]);

    // --- SHA-256 ---
    let got = gpu.hash_batch("k_sha256", &msgs, 32)?;
    let sha256_ok = msgs.iter().zip(&got).all(|(m, g)| {
        let want = sha256::Hash::hash(m).to_byte_array();
        g.as_slice() == want
    });
    report("SHA-256", sha256_ok, &mut all_ok);

    // --- SHA-512 ---
    let got = gpu.hash_batch("k_sha512", &msgs, 64)?;
    let sha512_ok = msgs.iter().zip(&got).all(|(m, g)| {
        let want = sha512::Hash::hash(m).to_byte_array();
        g.as_slice() == want
    });
    report("SHA-512", sha512_ok, &mut all_ok);

    // --- Keccak-256 (vs the sha3 crate, plus a hardcoded known-answer test) ---
    let got = gpu.hash_batch("k_keccak256", &msgs, 32)?;
    let mut keccak_ok = msgs
        .iter()
        .zip(&got)
        .all(|(m, g)| g.as_slice() == crate::eth::keccak256(m));
    // Independent of both implementations: if the sha3 crate were somehow the
    // NIST variant, every comparison above would still agree while every digest
    // was wrong. These two constants are the published Keccak-256 vectors.
    keccak_ok &= hex::encode(&got[0]) == "c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470";
    keccak_ok &= hex::encode(&got[1]) == "4e03657aea45a94fc7d47ba826c8d667c0d1e6e33a64a036ec44f58fa12d6c45";
    report("Keccak-256 (incl. rate-boundary + KAT)", keccak_ok, &mut all_ok);

    // --- HMAC-SHA512 (vs bitcoin_hashes HmacEngine) ---
    // Keys chosen to cross the 128-byte block boundary (short, exactly-block, oversized).
    let hkeys: Vec<Vec<u8>> = vec![
        b"key".to_vec(),
        b"Bitcoin seed".to_vec(),
        vec![0x0bu8; 20],
        vec![0xaau8; 131], // > block size -> key gets hashed first
    ];
    let hmsgs: Vec<Vec<u8>> = vec![
        b"The quick brown fox jumps over the lazy dog".to_vec(),
        vec![0x00u8; 64],
        b"Hi There".to_vec(),
        vec![0xddu8; 200],
    ];
    let got = gpu.hmac_batch(&hkeys, &hmsgs)?;
    let hmac_ok = hkeys.iter().zip(&hmsgs).zip(&got).all(|((k, m), g)| {
        use bitcoin::hashes::{Hmac, HmacEngine};
        use bitcoin::hashes::HashEngine;
        let mut eng = HmacEngine::<sha512::Hash>::new(k);
        eng.input(m);
        let want = Hmac::<sha512::Hash>::from_engine(eng).to_byte_array();
        g.as_slice() == want
    });
    report("HMAC-SHA512", hmac_ok, &mut all_ok);

    // --- PBKDF2-HMAC-SHA512 / BIP-39 seed (vs bip39 crate Mnemonic::to_seed) ---
    use bip39::{Language, Mnemonic};
    let phrases = [
        "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about",
        "legal winner thank year wave sausage worth useful legal winner thank yellow",
        "letter advice cage absurd amount doctor acoustic avoid letter advice cage above",
    ];
    let mut pws = Vec::new();
    let mut salts = Vec::new();
    let mut want_seeds = Vec::new();
    for p in phrases {
        let m = Mnemonic::parse_in_normalized(Language::English, p)?;
        pws.push(m.to_string().into_bytes());
        salts.push(b"mnemonic".to_vec());
        want_seeds.push(m.to_seed("").to_vec());
    }
    // A Japanese mnemonic is ~220 bytes of UTF-8, past HMAC's 128-byte block, so
    // the key gets hashed before use. English phrases never reach that path.
    // Built from entropy rather than a literal: the wordlist is NFKD, in which
    // dakuten are separate code points, so a composed literal would not match.
    let m = Mnemonic::from_entropy_in(Language::Japanese, &[0u8; 16])?;
    let jp_bytes = m.to_string().into_bytes();
    anyhow::ensure!(
        jp_bytes.len() > 128,
        "expected the Japanese mnemonic to exceed one HMAC block, got {}",
        jp_bytes.len()
    );
    pws.push(jp_bytes);
    salts.push(b"mnemonic".to_vec());
    want_seeds.push(m.to_seed("").to_vec());

    let got = gpu.pbkdf2_batch(&pws, &salts, 2048)?;
    let pbkdf2_ok = got.iter().zip(&want_seeds).all(|(g, w)| g == w);
    report("PBKDF2-HMAC-SHA512 / BIP-39 seed", pbkdf2_ok, &mut all_ok);

    // --- secp256k1: priv -> compressed pubkey (vs secp256k1 crate) ---
    use bitcoin::secp256k1::{PublicKey, Scalar, Secp256k1, SecretKey};
    let secp = Secp256k1::new();
    let mut rng = SplitMix64::new(0x9E3779B97F4A7C15);
    let mut privs: Vec<[u8; 32]> = Vec::new();
    // Anchor with priv == 1 (pubkey == G) for a deterministic sanity point.
    let mut one = [0u8; 32];
    one[31] = 1;
    privs.push(one);
    while privs.len() < 512 {
        let cand = rng.fill32();
        if SecretKey::from_slice(&cand).is_ok() {
            privs.push(cand);
        }
    }
    let got = gpu.pubkey_batch(&privs)?;
    let pubkey_ok = privs.iter().zip(&got).all(|(p, g)| {
        let sk = SecretKey::from_slice(p).unwrap();
        let want = PublicKey::from_secret_key(&secp, &sk).serialize();
        g == &want
    });
    report("secp256k1 priv->compressed pubkey (512 keys)", pubkey_ok, &mut all_ok);

    // --- secp256k1: priv -> uncompressed X||Y ---
    // The compressed form above keeps only the parity of Y, so it cannot catch a
    // Y coordinate left in a non-canonical (unreduced) representation. Ethereum
    // hashes all 32 bytes of Y, so every bit of it has to be checked.
    let got = gpu.pubkey_xy_batch(&privs)?;
    let pubkey_xy_ok = privs.iter().zip(&got).all(|(p, g)| {
        let sk = SecretKey::from_slice(p).unwrap();
        let want = PublicKey::from_secret_key(&secp, &sk).serialize_uncompressed();
        g[..] == want[1..] // drop the 0x04 prefix, which Ethereum does not hash
    });
    report("secp256k1 priv->uncompressed X||Y (512 keys)", pubkey_xy_ok, &mut all_ok);

    // --- scalar add mod n (vs SecretKey::add_tweak) ---
    let mut avec: Vec<[u8; 32]> = Vec::new();
    let mut bvec: Vec<[u8; 32]> = Vec::new();
    let mut want_sum: Vec<[u8; 32]> = Vec::new();
    while avec.len() < 256 {
        let a = rng.fill32();
        let b = rng.fill32();
        let (sk, tw) = match (SecretKey::from_slice(&a), Scalar::from_be_bytes(b)) {
            (Ok(s), Ok(t)) => (s, t),
            _ => continue,
        };
        let sum = match sk.add_tweak(&tw) {
            Ok(s) => s,
            Err(_) => continue, // result was zero; vanishingly rare
        };
        avec.push(a);
        bvec.push(b);
        want_sum.push(sum.secret_bytes());
    }
    let got = gpu.scalar_add_batch(&avec, &bvec)?;
    let addn_ok = got.iter().zip(&want_sum).all(|(g, w)| g == w);
    report("secp256k1 scalar add mod n (256 pairs)", addn_ok, &mut all_ok);

    // --- BIP32 m/44'/60'/0'/0/0 seed -> Ethereum address (vs the CPU path) ---
    use bitcoin::bip32::DerivationPath;
    let path: DerivationPath = crate::eth::ETH_PATH.parse()?;
    let mut seeds: Vec<[u8; 64]> = Vec::new();
    let mut want_addr: Vec<[u8; 20]> = Vec::new();
    for p in phrases {
        let m = Mnemonic::parse_in_normalized(Language::English, p)?;
        let seed = m.to_seed("");
        want_addr.push(crate::eth::address_from_seed(&secp, &path, &seed)?);
        seeds.push(seed);
    }
    let got = gpu.seed_to_eth_batch(&seeds)?;
    let bip32_ok = got.iter().zip(&want_addr).all(|(g, w)| g == w);
    report("BIP32 m/44'/60'/0'/0/0 seed->ETH address", bip32_ok, &mut all_ok);

    // The CPU reference above and the GPU share no code, but they do share this
    // author's reading of BIP-44. Anchor the whole chain to a value produced
    // outside both: the canonical BIP-39 test mnemonic's first MetaMask account.
    let kat_seed = Mnemonic::parse_in_normalized(Language::English, phrases[0])?.to_seed("");
    let kat = gpu.seed_to_eth_batch(&[kat_seed])?;
    let kat_ok = crate::eth::to_eip55(&kat[0]) == "0x9858EfFD232B4033E47d90003D41EC34EcaEda94";
    report("BIP32 known-answer ('abandon...about' -> MetaMask #0)", kat_ok, &mut all_ok);

    Ok(all_ok)
}

/// Minimal SplitMix64 PRNG — deterministic test inputs without a rand dependency.
struct SplitMix64 {
    state: u64,
}
impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn fill32(&mut self) -> [u8; 32] {
        let mut out = [0u8; 32];
        for chunk in out.chunks_mut(8) {
            chunk.copy_from_slice(&self.next_u64().to_be_bytes());
        }
        out
    }
}

fn report(name: &str, ok: bool, all_ok: &mut bool) {
    println!("  [{}] {}", if ok { "PASS" } else { "FAIL" }, name);
    *all_ok &= ok;
}
