# CLAUDE.md

GPU brute-forcer that recovers a 12-word BIP-39 mnemonic from a known Ethereum
address by enumerating word orderings. Built to attack the Guntis Vitolins
"10 ETH challenge" (published 2020-02-12, YouTube `w4mpiuBP_aY`).

## Build & run

```bash
cargo build --release
./target/release/words-breaker --selftest          # verify GPU primitives vs CPU
./target/release/words-breaker <ADDR> <12 words>   # loose-word mode
./target/release/words-breaker <ADDR> --pattern "dutch ? ? ? fog ? ? ? ? ? ? parrot" --pool "..."
./target/release/words-breaker <ADDR> --post "dutch@1 fiber@4 <cands>" --video "fog@5 parrot@12 <cands>"
```

Two-batch mode (`--post`/`--video`): exactly 6 words per batch, `word@N` pins a
word at position N *and* assigns it to that batch, bare words are that batch's
candidates (drawn without replacement, subsets enumerated when the pool is
oversize). Which open slots belong to which batch is enumerated. Space:
`C(h, a) * P(p_post, a) * P(p_video, h-a)` with `h` open slots of which `a` are
post-open. No `--fill` fallback in this mode.

`nvcc` + an NVIDIA GPU are required to build; `CUDA_LIBRARY_PATH` is set in
`.cargo/config.toml` to `/usr/lib/cuda` (distro CUDA layout). `--cpu` forces the
rayon path. See README.md for the full flag table.

## Layout

| File | Role |
|---|---|
| `src/main.rs` | CLI, pattern/pool parsing, address checksum validation |
| `src/candidates.rs` | the enumerators: `stream` (12 slots + one pool + `--fill` for leftovers) and `stream_two_pools` (6+6 batch split) |
| `src/gpu.rs` | CUDA host side: batching, producer thread, checksum pre-filter, `--selftest` |
| `src/cuda/kernels.cu` | SHA-256/512, Keccak-256, HMAC, PBKDF2, secp256k1, BIP32, `k_pipeline` |
| `src/eth.rs` | CPU reference implementation (also the selftest oracle) |

Derivation is fixed at `m/44'/60'/0'/0/0`, no BIP-39 passphrase.

## Traps

- **Do not remove `__noinline__`** from `pbkdf2_hmac_sha512_64`,
  `seed_to_eth_address`, or `keccak256`. Inlined into `k_pipeline`, `nvcc -O`
  miscompiles PBKDF2 and the seed comes out wrong — but the standalone kernels
  stay correct, so `--selftest` passes while the real search silently finds
  nothing.
- **PBKDF2-HMAC-SHA512 is ~92–94% of GPU time** (2048 iterations, fixed by
  BIP-39). secp256k1 + the address hash are single-digit percent, so optimizing
  them caps out around a 1.06x total win. Profile first; check
  `ptxas --verbose` after touching `k_pipeline` (baseline on the RTX 3050:
  2720-byte frame, 128 registers, 0 spills).
- **WSL copy artifacts**: this tree came from Windows. `target/` build scripts
  lose their exec bit (`find target -name "build-script-build*" -not -name "*Zone.Identifier" -exec chmod +x {} \;`,
  same for `*.so`) — chmod, do not `cargo clean`. ~894 `*:Zone.Identifier`
  files litter the tree; ask before deleting them.
- The pool is drawn **without replacement**, so a phrase that repeats a word is
  only reachable by listing that word twice in `--pool`.

## Cost model

Measured 2.42M candidates/s on an RTX 3050 (~151k full derivations/s after the
1-in-16 checksum filter). The full 12! space is ~197 s.

With `h` open slots, pool `p`, fill set `f`: `p!/(p-h)!` when `p >= h`, else
`h!/(h-p)! * f^(h-p)`. With the four pins held (**8 open slots**) and free slots
drawn from the 218 `d*`/`f*` words:

| unknown slots | space | time |
|---|---|---|
| 0 | 40,320 | 0.02 s |
| 1 | 8.8M | 3.6 s |
| 2 | 9.6e8 | 6.6 min |
| 3 | 7.0e10 | 8.0 h |
| 4 | 3.8e12 | 18 days |

Pinning `fiber`@4 bought roughly an order of magnitude at every row — three
unknowns went from 72 h to 8 h, so three is now inside the practical frontier
where before only two was.

If a candidate *list* of size C supplies the words, don't forget the `C(C, k)`
choose-factor — it dominates. With `fork` confirmed and U of the 7 remaining
slots unknown, the cost is `C(C, 7-U) * P(8, 1+(7-U)) * 218^U`; that first
factor is the one that is easy to drop and it is worth orders of magnitude.

## Challenge state

Target `0x9C2F44EFAd0c1E852a09dF9939e6DaF061140CaF`, confirmed on-chain to hold
8.612541554256944620 ETH.

### Known positions (Leo: certain)

| Position | Word |
|---|---|
| 1 | `dutch` |
| 4 | `fiber` |
| 5 | `fog` |
| 12 | `parrot` |

Positions 2–3 and 6–11 are open — **8 open slots**. The `fog`@5 pin is
corroborated: dropping it and replacing position 5 with any `d*`/`f*` word
(791M candidates) found no match.

### Known words (position unknown)

`fork` — Leo: certain it is in the phrase, position not yet identified. That
leaves 7 genuinely unknown slots.

### Extraction rule

Both confirmed words (`dutch`, `fiber`) appear as exact verbatim tokens in the
planted sentences, so the rule is: **take every token that is an exact BIP-39
word, grammatical connectors included.** BIP-39's 4-letter uniqueness means
stems resolve (`healthy`→`health`, `hunter`→`hunt`).

Blog text ("Round dutch cattle is living in the forest and eating wood. Only
because there is a lot of healthy fiber. Hunter like the rib roast dinner
fresh.") → 14 words:
`round dutch cattle forest wood only because there fiber like rib roast dinner fresh`

Video fragment ("Don't expect anything easy there will be dark fog on the
lake") → exactly 6, matching the 6 words the puzzle hides in the video:
`expect easy there will fog lake`

`parrot` appears in neither fragment, so the source text we have is incomplete.
The published research says part of the pool hides in the blog post's HTML
`article:tag` metadata rather than visible text.

### Already exhausted — do NOT re-run

All against the target above, all no match:

- pool `fork fiber forest dinner goat seed key lake` with `fog`@5 (743M), and
  the same with `cloud`@5, with `dinner` but no `goat`, and with `dutch` added
  to the pool
- position 5 unpinned, `fog` or `cloud` floating over 10 slots (3.6M each)
- 13-word pool (+`round wood cattle roast fresh`) and 14-word (+`dutch`), with
  `fog`@5 and `cloud`@5 (259M / 726M each)
- the all-`d`/`f` theory `dutch fog parrot fork fiber forest dinner fresh donor
  favorite find decide`: pinned (363K), **all 12 words free in every ordering
  (479M)**, and dropping each unpinned word for any `d*`/`f*` replacement
  (9 × 79M)
- pins held, pool of `fork fiber` + candidates incl. `deliver`, `detail`,
  `digital`, `day`: pools of 10/11/12/13 → 3.6M / 20M / 80M / 259M

All four pins held (`dutch`@1, `fiber`@4, `fog`@5, `parrot`@12), 2026-08-25:

- pool `dinner deliver fork forest winter shop update seed` — exact fit to the
  8 open slots (40.3K)
- pool `fork roast expect any live fresh because cattle` — exact fit (40.3K).
  Note `anything` and `living` are **not** BIP-39 words; `any`/`live` are the
  only stems they resolve to.
- pool `fork roast expect fresh because cattle` + 2 slots over any `d*`/`f*`
  (958M) — closes that whole two-word family, not just one ordering
- pool `easy goat sponsor fork seed` + 3 slots over the 21-word candidate list
  `round cattle forest wood only because there like rib roast dinner fresh
  expect will lake deliver winter shop update any live` (62.2M), and the same
  with `sponsor donor` added to the fill set (81.8M)
- pool `easy goat sponsor fork seed update video sponsor donor key find round
  cattle forest wood dinner` — all P(16,8) = 518.9M orderings. Strong negative:
  with the pins right, the other 8 words are **not** all drawn from that set.
- the same pool + `roast`, all P(17,8) = 980.2M — same conclusion, 17 words.

None of the earlier (pre-pin) runs need re-running under the `fiber`@4 pin: every one of them
either let `fiber` float freely (so `fiber`@4 was already covered) or excluded
`fiber` entirely. The pin narrows future searches, it does not reopen past ones.

The 479M full-freedom run is the decisive one: **with positions fully free the
word set itself is wrong.** The per-word drop runs then show it is not a single
wrong word among the nine unpinned ones either. So ≥2 words are wrong, or one
of the three pins is — and given the pins are corroborated, the pool is the
likelier error.

### In flight

`/home/leo/words-breaker-sweep/sweep.sh` (started 2026-08-25): the 17-word pool
into 7 open slots, with the 8th open slot pinned to each of the 2048 BIP-39
words in turn — 8 x 2048 sub-runs of P(17,7)=98.0M each, ~1.6e12 total, ~7.6
days. Resumable: `done.txt` holds every finished `slot:word`, re-running the
script picks up from there; a hit lands in `HIT.txt`. This tests exactly one
hypothesis — all four pins right and exactly *one* phrase word outside the 17.

### Untried levers, cheapest first

1. Unpin position 5 and let `fog`/`cloud` float over 10 slots: `P(14,10)`=3.6e9
   (~25 min), `P(15,10)`=1.1e10 (~75 min).
2. Get the *complete* video transcript and blog post source (HTML metadata
   included) and re-run the extraction rule. This is the highest-value move —
   the blocker is the word pool, not throughput.
