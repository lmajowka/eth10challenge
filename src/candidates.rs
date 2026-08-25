//! Index-based candidate streaming for the search.
//!
//! Every search mode is one template: 12 slots, each either a fixed BIP-39 index
//! or a hole, plus a pool of words that fill the holes **without replacement**.
//! If the pool is smaller than the number of holes, the leftover holes each draw
//! independently from the *fill set* — by default the whole wordlist, but
//! narrowing it is the single most effective lever there is, since each such
//! hole multiplies the space by the fill set's size.
//!
//! Supplying 12 loose words is just the special case of 12 holes and a 12-word
//! pool, so the two modes share one enumerator and one candidate count.
//!
//! Candidates are yielded as `[u16; 12]` arrays of word indices — the compact
//! form the GPU kernel consumes.

use itertools::Itertools;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Slot {
    /// A position whose word is known.
    Fixed(u16),
    /// A position to be filled from the pool, or from the full wordlist once the
    /// pool runs out.
    Hole,
}

/// Exactly how many candidates [`stream`] will yield.
///
/// With `h` holes, a pool of `p` and a fill set of `f`:
/// - `p >= h`: arrange `h` of the `p` pool words over the holes — `p!/(p-h)!`.
/// - `p < h`: place all `p` pool words into distinct holes (`h!/(h-p)!` ways),
///   then draw each of the `h-p` remaining holes from the fill set — `f^(h-p)`.
///
/// Returns `u128` because a few holes drawn from the full list overflow `u64`
/// quickly — `2048^6` alone is ~7.4e19.
pub fn count(slots: &[Slot; 12], pool_len: usize, fill_len: usize) -> u128 {
    let h = slots.iter().filter(|s| **s == Slot::Hole).count();
    if pool_len >= h {
        ((pool_len - h + 1)..=pool_len).map(|x| x as u128).product()
    } else {
        let w = h - pool_len;
        let placements: u128 = ((w + 1)..=h).map(|x| x as u128).product();
        placements * (fill_len as u128).pow(w as u32)
    }
}

/// Streams every candidate the template describes.
///
/// Nothing is collected up front: memory stays flat no matter how large the
/// space is.
/// `fill` is the set of word indices a hole may take once the pool is spent.
pub fn stream(
    slots: [Slot; 12],
    pool: Vec<u16>,
    fill: Vec<u16>,
) -> Box<dyn Iterator<Item = [u16; 12]> + Send> {
    let holes: Vec<usize> = (0..12).filter(|&i| slots[i] == Slot::Hole).collect();
    let mut base = [0u16; 12];
    for (i, s) in slots.iter().enumerate() {
        if let Slot::Fixed(w) = *s {
            base[i] = w;
        }
    }

    let h = holes.len();
    let p = pool.len();

    if h == 0 {
        return Box::new(std::iter::once(base));
    }

    if p >= h {
        // Every hole gets a distinct pool word; surplus pool words mean we also
        // choose *which* ones, which `permutations(h)` already enumerates.
        return Box::new(pool.into_iter().permutations(h).map(move |arr| {
            let mut out = base;
            for (&slot, w) in holes.iter().zip(arr) {
                out[slot] = w;
            }
            out
        }));
    }

    // Fewer pool words than holes. Pick which holes fall back to the fill set
    // first; the pool then permutes over the rest. Choosing the fallback slots
    // up front is what keeps this duplicate-free — inserting unknown words one
    // at a time generates each candidate w! times over.
    let w = h - p;
    let all_holes = holes.clone();
    Box::new(
        holes
            .into_iter()
            .combinations(w)
            .flat_map(move |wild| {
                let rest: Vec<usize> = all_holes
                    .iter()
                    .copied()
                    .filter(|i| !wild.contains(i))
                    .collect();
                let pool = pool.clone();
                let fill = fill.clone();
                pool.into_iter().permutations(p).flat_map(move |arr| {
                    let mut tmpl = base;
                    for (&slot, word) in rest.iter().zip(&arr) {
                        tmpl[slot] = *word;
                    }
                    fill_holes(tmpl, wild.clone(), fill.clone())
                })
            }),
    )
}

/// Expands the given slots over the fill set, one nesting level per slot.
fn fill_holes(
    tmpl: [u16; 12],
    slots: Vec<usize>,
    fill: Vec<u16>,
) -> Box<dyn Iterator<Item = [u16; 12]> + Send> {
    match slots.split_first() {
        None => Box::new(std::iter::once(tmpl)),
        Some((&slot, rest)) => {
            let rest = rest.to_vec();
            Box::new(fill.clone().into_iter().flat_map(move |word| {
                let mut t = tmpl;
                t[slot] = word;
                fill_holes(t, rest.clone(), fill.clone())
            }))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    fn holes(n: usize) -> [Slot; 12] {
        let mut s = [Slot::Fixed(0); 12];
        for slot in s.iter_mut().take(n) {
            *slot = Slot::Hole;
        }
        s
    }

    /// `count` must agree with what `stream` actually yields, for every shape.
    #[test]
    fn count_matches_stream() {
        for (slots, pool, fill_len) in [
            (holes(3), vec![10u16, 20, 30], 5usize),  // p == h
            (holes(3), vec![10, 20, 30, 40, 50], 5),  // p > h  (choose and arrange)
            (holes(3), vec![10, 20], 5),              // p < h  (one free slot)
            (holes(4), vec![10, 20], 4),              // p < h  (two free slots)
            (holes(2), vec![], 7),                    // pool empty: fill^2
            (holes(2), vec![], 3),                    // narrowed fill set
            (holes(0), vec![], 4),                    // no holes at all
        ] {
            let fill: Vec<u16> = (0..fill_len as u16).collect();
            let want = count(&slots, pool.len(), fill.len());
            let got: Vec<_> = stream(slots, pool.clone(), fill.clone()).collect();
            assert_eq!(got.len() as u128, want, "slots={slots:?} pool={pool:?}");
            // Enumeration must be duplicate-free, which is the whole point of
            // choosing the fill slots before permuting the pool.
            let uniq: HashSet<_> = got.iter().collect();
            assert_eq!(uniq.len(), got.len(), "duplicates for pool={pool:?}");
            // Free slots must only ever take words from the fill set.
            for c in &got {
                for (i, &word) in c.iter().enumerate() {
                    if slots[i] == Slot::Hole && !pool.contains(&word) {
                        assert!(fill.contains(&word), "{word} not in fill set");
                    }
                }
            }
        }
    }

    /// The d/f hypothesis: 7 known words, 2 slots restricted to a 218-word set.
    #[test]
    fn restricted_fill_shrinks_the_space() {
        let mut slots = [Slot::Hole; 12];
        slots[0] = Slot::Fixed(1);
        slots[4] = Slot::Fixed(2);
        slots[11] = Slot::Fixed(3);
        assert_eq!(count(&slots, 7, 2048), 761_014_517_760);
        assert_eq!(count(&slots, 7, 218), 8_622_754_560);
    }

    #[test]
    fn fixed_slots_are_never_touched() {
        let mut slots = [Slot::Hole; 12];
        slots[0] = Slot::Fixed(111);
        slots[4] = Slot::Fixed(222);
        slots[11] = Slot::Fixed(333);
        // 9 holes, 9 pool words -> 9! arrangements, all with the pins intact.
        let pool: Vec<u16> = (1..=9).collect();
        assert_eq!(count(&slots, pool.len(), 2048), 362_880);
        let fill: Vec<u16> = (0..2048).collect();
        for c in stream(slots, pool.clone(), fill).take(5000) {
            assert_eq!((c[0], c[4], c[11]), (111, 222, 333));
            let mut mid: Vec<u16> = (0..12).filter(|i| ![0, 4, 11].contains(i)).map(|i| c[i]).collect();
            mid.sort();
            assert_eq!(mid, pool);
        }
    }

    /// The challenge's actual shape: 3 pinned, 9 holes, 8 pool words.
    #[test]
    fn challenge_shape_counts() {
        let mut slots = [Slot::Hole; 12];
        slots[0] = Slot::Fixed(1);
        slots[4] = Slot::Fixed(2);
        slots[11] = Slot::Fixed(3);
        assert_eq!(count(&slots, 8, 2048), 743_178_240);
    }

    /// 12 loose words must still enumerate exactly 12!, unchanged.
    #[test]
    fn twelve_loose_words_is_twelve_factorial() {
        assert_eq!(count(&[Slot::Hole; 12], 12, 2048), 479_001_600);
    }
}
