use anyhow::{Context, Result};
use bip39::{Language, Mnemonic};
use bitcoin::bip32::DerivationPath;
use bitcoin::secp256k1::Secp256k1;
use clap::Parser;
use rayon::iter::ParallelBridge;
use rayon::prelude::*;
use std::io::{self, Write};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

mod candidates;
mod eth;
mod gpu;

use candidates::Slot;

#[derive(Parser, Debug)]
#[command(about = "Try permutations of BIP-39 words (10-12) to match an Ethereum address \
(BIP-44 m/44'/60'/0'/0/0). Missing words (when 10 or 11 are given) are filled from the \
2048-word BIP-39 list.", version)]
struct Args {
    /// Target Ethereum address, e.g. 0x9858EfFD232B4033E47d90003D41EC34EcaEda94.
    /// Mixed-case input is checked against its EIP-55 checksum.
    /// Optional only when --selftest is given.
    target_address: Option<String>,

    /// 10, 11, or 12 words (unordered or partially ordered). Missing words are
    /// completed from the BIP-39 wordlist. Ignored when --pattern is given.
    words: Vec<String>,

    /// Positional template: 12 slots separated by spaces, `?` for a slot whose
    /// word is unknown. Words at the other slots are pinned there.
    /// Example: --pattern "dutch ? ? ? fog ? ? ? ? ? ? parrot"
    #[arg(long)]
    pattern: Option<String>,

    /// Words that fill the `?` slots of --pattern, each used at most once
    /// (space- or comma-separated). If there are more `?` slots than pool words,
    /// each leftover slot is drawn from the full BIP-39 list, which multiplies
    /// the search space by 2048 per slot.
    #[arg(long)]
    pool: Option<String>,

    /// Restricts what a `?` slot may hold once the pool is spent. Accepts
    /// literal words and `prefix*` patterns, space- or comma-separated.
    /// Defaults to the entire wordlist. Each unfilled slot multiplies the search
    /// by this set's size, so narrowing it is the strongest lever available:
    /// --fill "d*,f*" cuts 2048 down to 218.
    #[arg(long)]
    fill: Option<String>,

    /// Two-batch mode, post half: the 6 phrase words hidden in the blog post.
    /// `word@N` pins that word at position N (1-based) and marks it as coming
    /// from the post; bare words are candidates for the post's remaining slots.
    /// Requires --video; each batch supplies exactly 6 of the 12 words, and
    /// which open slots belong to which batch is enumerated automatically.
    /// Example: --post "dutch@1 fiber@4 forest dinner rib roast fresh cattle"
    #[arg(long, conflicts_with_all = ["pattern", "pool", "fill", "words"], requires = "video")]
    post: Option<String>,

    /// Two-batch mode, video half: the 6 phrase words hidden in the video,
    /// same syntax as --post. Example: --video "fog@5 parrot@12 expect easy there will lake"
    #[arg(long, requires = "post")]
    video: Option<String>,

    /// BIP-39 wordlist language (english, portuguese, spanish, french, italian, czech, korean, japanese, chinese-simplified, chinese-traditional)
    #[arg(long, short, default_value = "english")]
    language: String,

    /// Number of threads to use (defaults to number of CPU cores)
    #[arg(long, short, default_value_t = 0)]
    threads: usize,

    /// Verify each GPU crypto primitive against the CPU reference and exit.
    #[arg(long)]
    selftest: bool,

    /// Force the CPU (rayon) search instead of the GPU. The GPU is used by
    /// default when a CUDA device is available.
    #[arg(long)]
    cpu: bool,

    /// Skip the BIP-39 checksum pre-filter and derive an address for every
    /// candidate. A conforming mnemonic always has a valid checksum, so this
    /// only finds phrases produced by a non-conforming generator -- at 16x the
    /// cost, since the filter is what rejects 15/16 of candidates.
    #[arg(long)]
    no_checksum: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();

    if args.selftest {
        println!("Running GPU primitive selftests...");
        let ok = gpu::run_selftest()?;
        if ok {
            println!("All selftests passed.");
            return Ok(());
        } else {
            anyhow::bail!("One or more selftests FAILED");
        }
    }

    let target_address = args
        .target_address
        .as_deref()
        .context("Missing target address")?;

    let target = eth::parse_address(target_address)?;
    let language = parse_language(&args.language)?;
    let wordlist: &'static [&'static str] = language.words_by_prefix("");

    let plan = build_plan(&args, wordlist)?;

    println!("Target: {} ({})", eth::to_eip55(&target), eth::ETH_PATH);
    plan.describe(wordlist);

    let start = Instant::now();

    // GPU by default; fall back to CPU if no CUDA device or on --cpu.
    let found = if args.cpu {
        run_cpu_search(&args, &plan, wordlist, language, &target)?
    } else {
        match search_gpu(&plan, wordlist, &target, args.no_checksum) {
            Ok(found) => found,
            Err(e) => {
                eprintln!("GPU search unavailable ({e:#}); falling back to CPU.");
                run_cpu_search(&args, &plan, wordlist, language, &target)?
            }
        }
    };
    let elapsed = start.elapsed();

    if !found {
        println!(
            "Exhausted all permutations without a match (elapsed: {:?})",
            elapsed
        );
    }

    Ok(())
}

/// A fully resolved search: the slot template plus whichever enumerator the
/// CLI flags selected. Everything downstream (describe, GPU, CPU, hit report)
/// works off this, so both modes share the search machinery.
enum Plan {
    /// One pool over all holes, fill-set fallback for the surplus holes.
    Template {
        slots: [Slot; 12],
        pool: Vec<u16>,
        fill: Vec<u16>,
    },
    /// --post/--video: 6 words per batch, each batch's open slots drawn from
    /// its own pool, the slot-to-batch assignment enumerated.
    Batches {
        slots: [Slot; 12],
        post: Vec<u16>,
        video: Vec<u16>,
        /// How many open slots the post batch still has to fill (6 - its pins).
        post_open: usize,
    },
}

impl Plan {
    fn slots(&self) -> &[Slot; 12] {
        match self {
            Plan::Template { slots, .. } | Plan::Batches { slots, .. } => slots,
        }
    }

    /// The words a hole may legitimately take, for the hit report's
    /// "recovered from the full wordlist" distinction.
    fn report_pool(&self) -> Vec<u16> {
        match self {
            Plan::Template { pool, .. } => pool.clone(),
            Plan::Batches { post, video, .. } => {
                post.iter().chain(video.iter()).copied().collect()
            }
        }
    }

    fn count(&self) -> u128 {
        match self {
            Plan::Template { slots, pool, fill } => candidates::count(slots, pool.len(), fill.len()),
            Plan::Batches { slots, post, video, post_open } => {
                let holes = slots.iter().filter(|s| **s == Slot::Hole).count();
                candidates::count_two_pools(holes, *post_open, post.len(), video.len())
            }
        }
    }

    fn stream(&self) -> Box<dyn Iterator<Item = [u16; 12]> + Send> {
        match self {
            Plan::Template { slots, pool, fill } => {
                candidates::stream(*slots, pool.clone(), fill.clone())
            }
            Plan::Batches { slots, post, video, post_open } => {
                candidates::stream_two_pools(*slots, *post_open, post.clone(), video.clone())
            }
        }
    }

    /// Prints the shape of the space before committing to a long run — a
    /// miscounted pool or a stray `?` is much cheaper to notice here than an
    /// hour in.
    fn describe(&self, wordlist: &[&str]) {
        let slots = self.slots();
        let holes = slots.iter().filter(|s| **s == Slot::Hole).count();
        let fixed = 12 - holes;
        match self {
            Plan::Template { pool, fill, .. } => {
                println!(
                    "Template: {} pinned, {} open slot(s), {} pool word(s)",
                    fixed,
                    holes,
                    pool.len()
                );
                if pool.len() < holes {
                    let free = holes - pool.len();
                    let scope = if fill.len() == wordlist.len() {
                        format!("the full {}-word list", wordlist.len())
                    } else {
                        format!("a restricted set of {} word(s)", fill.len())
                    };
                    println!(
                        "  {free} slot(s) have no pool word and will be drawn from {scope} (x{} per slot)",
                        fill.len()
                    );
                } else if pool.len() > holes {
                    println!(
                        "  pool has {} more word(s) than slots, so subsets are enumerated too",
                        pool.len() - holes
                    );
                }
            }
            Plan::Batches { post, video, post_open, .. } => {
                println!(
                    "Two batches of 6: {} pinned, {} open slot(s) — post fills {} from {} candidate(s), video fills {} from {} candidate(s)",
                    fixed,
                    holes,
                    post_open,
                    post.len(),
                    holes - post_open,
                    video.len()
                );
                println!("  which open slots belong to which batch is enumerated too");
            }
        }
        println!("Searching {} candidates (streamed)...", format_u128(self.count()));
    }
}

/// Resolves the CLI into a [`Plan`].
///
/// `--pattern`/`--pool` build the single-pool template directly; loose
/// positional words are the same thing with every slot open. `--post`/`--video`
/// build the two-batch plan.
fn build_plan(args: &Args, wordlist: &[&str]) -> Result<Plan> {
    let index_of = |w: &str| -> Result<u16> {
        wordlist
            .iter()
            .position(|x| *x == w)
            .map(|p| p as u16)
            .with_context(|| format!("'{w}' is not in the {} BIP-39 wordlist", args.language))
    };

    // Two-batch mode: clap guarantees --post and --video come together and
    // exclude the other selectors.
    if let (Some(post_spec), Some(video_spec)) = (args.post.as_deref(), args.video.as_deref()) {
        let mut slots = [Slot::Hole; 12];
        let mut parse_batch = |spec: &str, name: &str| -> Result<(Vec<u16>, usize)> {
            let mut pool = Vec::new();
            let mut pins = 0usize;
            for tok in spec.split([' ', ',', '\t', '\n']).filter(|s| !s.is_empty()) {
                match tok.split_once('@') {
                    Some((word, pos)) => {
                        let pos: usize = pos
                            .parse()
                            .ok()
                            .filter(|p| (1..=12).contains(p))
                            .with_context(|| {
                                format!("--{name}: '{tok}' — position must be 1-12")
                            })?;
                        anyhow::ensure!(
                            slots[pos - 1] == Slot::Hole,
                            "--{name}: position {pos} is pinned twice"
                        );
                        slots[pos - 1] = Slot::Fixed(index_of(word)?);
                        pins += 1;
                    }
                    None => pool.push(index_of(tok)?),
                }
            }
            anyhow::ensure!(
                pins <= 6,
                "--{name} pins {pins} positions, but each batch holds only 6 of the 12 words"
            );
            let open = 6 - pins;
            anyhow::ensure!(
                pool.len() >= open,
                "--{name} pins {pins} word(s), so it must supply at least {open} candidate(s) \
                 for its remaining slots, got {}",
                pool.len()
            );
            Ok((pool, open))
        };
        let (post, post_open) = parse_batch(post_spec, "post")?;
        let (video, _) = parse_batch(video_spec, "video")?;
        return Ok(Plan::Batches { slots, post, video, post_open });
    }

    // Which words a `?` slot may take once the pool runs out.
    let fill: Vec<u16> = match args.fill.as_deref() {
        None => (0..wordlist.len() as u16).collect(),
        Some(spec) => {
            let mut set: Vec<u16> = Vec::new();
            for tok in spec.split([' ', ',', '\t', '\n']).filter(|s| !s.is_empty()) {
                match tok.strip_suffix('*') {
                    // `abc*` — every word with that prefix.
                    Some(prefix) => {
                        let before = set.len();
                        set.extend(
                            wordlist
                                .iter()
                                .enumerate()
                                .filter(|(_, w)| w.starts_with(prefix))
                                .map(|(i, _)| i as u16),
                        );
                        anyhow::ensure!(
                            set.len() > before,
                            "no {} BIP-39 word starts with '{prefix}'",
                            args.language
                        );
                    }
                    None => set.push(index_of(tok)?),
                }
            }
            set.sort_unstable();
            set.dedup();
            anyhow::ensure!(!set.is_empty(), "--fill matched no words");
            set
        }
    };

    let Some(pattern) = args.pattern.as_deref() else {
        anyhow::ensure!(
            args.pool.is_none(),
            "--pool only means something with --pattern; without it, pass the words positionally"
        );
        anyhow::ensure!(
            (10..=12).contains(&args.words.len()),
            "Expected 10, 11, or 12 words, got {}",
            args.words.len()
        );
        let pool = args
            .words
            .iter()
            .map(|w| index_of(w))
            .collect::<Result<Vec<u16>>>()?;
        return Ok(Plan::Template {
            slots: [Slot::Hole; 12],
            pool,
            fill,
        });
    };

    let fields: Vec<&str> = pattern.split_whitespace().collect();
    anyhow::ensure!(
        fields.len() == 12,
        "--pattern needs exactly 12 slots, got {} in {pattern:?}",
        fields.len()
    );
    let mut slots = [Slot::Hole; 12];
    for (i, f) in fields.iter().enumerate() {
        if *f != "?" {
            slots[i] = Slot::Fixed(index_of(f)?);
        }
    }

    let pool = args
        .pool
        .as_deref()
        .unwrap_or("")
        .split([' ', ',', '\t', '\n'])
        .filter(|s| !s.is_empty())
        .map(index_of)
        .collect::<Result<Vec<u16>>>()?;

    Ok(Plan::Template { slots, pool, fill })
}

/// Configures the rayon pool and runs the CPU search.
fn run_cpu_search(
    args: &Args,
    plan: &Plan,
    wordlist: &'static [&'static str],
    language: Language,
    target: &[u8; 20],
) -> Result<bool> {
    let num_threads = if args.threads == 0 {
        num_cpus::get()
    } else {
        args.threads
    };
    // build_global can only be called once; ignore an already-initialized pool.
    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build_global();
    println!("Using CPU with {} threads", num_threads);
    search_cpu(plan, wordlist, language, target, args.no_checksum)
}

/// Streams candidates to the GPU in batches and reports a hit.
fn search_gpu(
    plan: &Plan,
    wordlist: &'static [&'static str],
    target: &[u8; 20],
    no_checksum: bool,
) -> Result<bool> {
    let gpu = gpu::Gpu::new()?;
    println!("Using GPU (CUDA)");

    let gpu_wordlist = gpu::GpuWordlist::new(wordlist)?;
    let candidates = plan.stream();
    let batch_size = 1 << 20;
    let hit = gpu.search(candidates, &gpu_wordlist, target, batch_size, !no_checksum)?;

    match hit {
        Some(h) => {
            report_hit(
                &h.indices,
                Some(h.global_index),
                plan.slots(),
                &plan.report_pool(),
                wordlist,
                target,
            );
            Ok(true)
        }
        None => Ok(false),
    }
}

/// Prints a match, calling out any word that came from the full wordlist rather
/// than from the pattern or the pool — those are the genuinely recovered ones.
fn report_hit(
    indices: &[u16; 12],
    index: Option<usize>,
    slots: &[Slot; 12],
    pool: &[u16],
    wordlist: &[&str],
    target: &[u8; 20],
) {
    let phrase: Vec<&str> = indices.iter().map(|&i| wordlist[i as usize]).collect();
    println!("Found matching mnemonic: {}", phrase.join(" "));

    let mut unused: Vec<u16> = pool.to_vec();
    let mut recovered: Vec<String> = Vec::new();
    for (i, &w) in indices.iter().enumerate() {
        if slots[i] != Slot::Hole {
            continue; // pinned by --pattern
        }
        match unused.iter().position(|&p| p == w) {
            Some(p) => {
                unused.remove(p);
            }
            None => recovered.push(format!("position {}: {}", i + 1, wordlist[w as usize])),
        }
    }
    if !recovered.is_empty() {
        println!("Recovered from the full wordlist: {}", recovered.join(", "));
    }
    if let Some(i) = index {
        println!("Candidate index (0-based): {}", i);
    }
    println!("Derived address: {}", eth::to_eip55(target));
}

pub fn format_number(n: usize) -> String {
    if n >= 1_000_000_000 {
        format!("{:.1}G", n as f64 / 1_000_000_000.0)
    } else if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.1}K", n as f64 / 1_000.0)
    } else {
        n.to_string()
    }
}

fn parse_language(lang: &str) -> Result<Language> {
    match lang.to_lowercase().as_str() {
        "english" => Ok(Language::English),
        "portuguese" => Ok(Language::Portuguese),
        "spanish" => Ok(Language::Spanish),
        "french" => Ok(Language::French),
        "italian" => Ok(Language::Italian),
        "czech" => Ok(Language::Czech),
        "korean" => Ok(Language::Korean),
        "japanese" => Ok(Language::Japanese),
        "chinese-simplified" => Ok(Language::SimplifiedChinese),
        "chinese-traditional" => Ok(Language::TraditionalChinese),
        _ => anyhow::bail!("Unknown language: {}. Supported: english, portuguese, spanish, french, italian, czech, korean, japanese, chinese-simplified, chinese-traditional", lang),
    }
}

/// CPU search over the same candidate stream the GPU consumes.
///
/// `find_any` rather than `for_each` with a flag: the flag version keeps pulling
/// every remaining candidate off the iterator after a hit, which on a 12! space
/// meant minutes of work to report a match already in hand.
fn search_cpu(
    plan: &Plan,
    wordlist: &'static [&'static str],
    language: Language,
    target: &[u8; 20],
    no_checksum: bool,
) -> Result<bool> {
    let derivation_path: DerivationPath = eth::ETH_PATH.parse()?;
    // Shared Secp256k1 context (thread-safe).
    let secp = Arc::new(Secp256k1::new());
    let checked = Arc::new(AtomicUsize::new(0));
    let _ = io::stdout().flush();

    let hit = plan
        .stream()
        .par_bridge()
        .find_any(|indices| {
            let n = checked.fetch_add(1, Ordering::Relaxed);
            if n % 1_000_000 == 0 && n > 0 {
                println!("Checked {} candidates...", format_number(n));
                let _ = io::stdout().flush();
            }

            let phrase: Vec<&str> = indices.iter().map(|&i| wordlist[i as usize]).collect();
            let phrase = phrase.join(" ");

            // Rejects the ~15/16 of candidates with a bad BIP-39 checksum, the
            // same cheap filter the GPU applies before deriving. --no-checksum
            // keeps them all, matching the GPU's pass-through mode.
            let parsed = if no_checksum {
                Mnemonic::parse_in_normalized_without_checksum_check(language, &phrase)
            } else {
                Mnemonic::parse_in_normalized(language, &phrase)
            };
            let Ok(mnemonic) = parsed else {
                return false;
            };
            let seed = mnemonic.to_seed("");
            match eth::address_from_seed(&secp, &derivation_path, &seed) {
                Ok(addr) => addr == *target,
                Err(_) => false,
            }
        });

    match hit {
        Some(indices) => {
            // The CPU index would be a racy atomic counter rather than a real
            // position, so none is reported here; rerun on the GPU if you need it.
            report_hit(&indices, None, plan.slots(), &plan.report_pool(), wordlist, target);
            Ok(true)
        }
        None => Ok(false),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// End-to-end: a two-batch plan whose pins and pools describe a real
    /// phrase must find it via the CPU path, including discovering which open
    /// slots belong to which batch.
    #[test]
    fn two_batch_cpu_search_finds_planted_phrase() {
        let language = Language::English;
        let wordlist: &'static [&'static str] = language.words_by_prefix("");
        let mnemonic = Mnemonic::from_entropy_in(language, &[0x5A; 16]).unwrap();
        let idx: Vec<u16> = mnemonic
            .word_iter()
            .map(|w| wordlist.iter().position(|x| *x == w).unwrap() as u16)
            .collect();

        let secp = Secp256k1::new();
        let path: DerivationPath = eth::ETH_PATH.parse().unwrap();
        let target = eth::address_from_seed(&secp, &path, &mnemonic.to_seed("")).unwrap();

        // Words 1-6 are the "post" batch, 7-12 the "video" batch. Pin four of
        // each; the remaining two per batch float over the four open slots
        // (5-8), so the search must also recover the slot-to-batch split.
        let mut slots = [Slot::Hole; 12];
        for i in [0, 1, 2, 3, 8, 9, 10, 11] {
            slots[i] = Slot::Fixed(idx[i]);
        }
        let plan = Plan::Batches {
            slots,
            post: vec![idx[4], idx[5]],
            video: vec![idx[6], idx[7]],
            post_open: 2,
        };
        assert_eq!(plan.count(), 24);
        assert!(search_cpu(&plan, wordlist, language, &target, false).unwrap());
    }
}

/// Formats a `u128` candidate count with a magnitude suffix.
fn format_u128(n: u128) -> String {
    const UNITS: [(u128, &str); 4] = [
        (1_000_000_000_000, "T"),
        (1_000_000_000, "G"),
        (1_000_000, "M"),
        (1_000, "K"),
    ];
    for (scale, suffix) in UNITS {
        if n >= scale {
            return format!("{:.1}{}", n as f64 / scale as f64, suffix);
        }
    }
    n.to_string()
}
