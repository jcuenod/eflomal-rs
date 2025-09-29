//! Symmetrization utilities.

use std::collections::HashSet;

use crate::types::{Link, NULL_LINK};
use crate::text::Text;

/// Symmetrize forward and reverse alignments using the Moses "grow-diag-final-and" heuristic.
///
/// References:
/// - Koehn, Philipp, and others. "Moses: Open Source Toolkit for Statistical Machine Translation."
///
/// Algorithm (as implemented here):
/// - For each sentence:
///   - If either sentence is missing in source/target, or either alignment is None: output None.
///   - Build S_fw (pairs from forward) and S_rev (pairs from reverse, reoriented to (source, target)).
///   - A = S_fw ∩ S_rev (intersection), U = S_fw ∪ S_rev (union).
///   - Grow-diag:
///     - Repeat until convergence:
///       - For each (i, j) in A, consider its 8-neighbours. If neighbour is in U and not in A, and
///         either i or j is currently unaligned in A, add neighbour to A.
///   - Final:
///     - For each (i, j) in U \ A, if either i or j is unaligned in A, add (i, j) to A.
///   - Final-and:
///     - For remaining (i, j) in S_fw \ A, if both i and j are unaligned in A, add it.
///     - Repeat for (i, j) in S_rev \ A.
///   - Construct result vector with length target_len:
///     - For each target position j, if there are one or more pairs (i, j) ∈ A, choose the smallest i
///       deterministically and set result[j] = i as Link; otherwise NULL_LINK.
///
/// Errors:
/// - Returns Err(...) if the counts of sentences mismatch across inputs.
/// - Returns Err(...) if for a sentence both alignments and both sentences exist, but lengths mismatch,
///   or if any index in the alignments is out of bounds.
///
/// Note:
/// - The union/intersection are sets over (source_index, target_index).
/// - Deterministic: no randomness, stable tie-breaking by choosing the smallest source index per target.
pub fn grow_diag_final_and(
    forward: &[Option<Vec<Link>>],
    reverse: &[Option<Vec<Link>>],
    source: &Text,
    target: &Text,
) -> Result<Vec<Option<Vec<Link>>>, String> {
    // Validate sentence counts
    let n_src = source.sentences.len();
    let n_tgt = target.sentences.len();

    if n_src != n_tgt {
        return Err(format!(
            "Mismatched sentence counts between source ({}) and target ({})",
            n_src, n_tgt
        ));
    }
    if forward.len() != n_tgt {
        return Err(format!(
            "Mismatched sentence counts: forward ({}) vs target ({})",
            forward.len(),
            n_tgt
        ));
    }
    if reverse.len() != n_tgt {
        return Err(format!(
            "Mismatched sentence counts: reverse ({}) vs target ({})",
            reverse.len(),
            n_tgt
        ));
    }

    let n_sentences = n_tgt;
    let mut merged: Vec<Option<Vec<Link>>> = Vec::with_capacity(n_sentences);

    for s in 0..n_sentences {
        let src_sent_opt = source.sentences[s].as_ref();
        let tgt_sent_opt = target.sentences[s].as_ref();
        let fwd_opt = &forward[s];
        let rev_opt = &reverse[s];

        // If sentence is missing or alignment missing, pass through None
        if src_sent_opt.is_none()
            || tgt_sent_opt.is_none()
            || fwd_opt.is_none()
            || rev_opt.is_none()
        {
            merged.push(None);
            continue;
        }

        let src_sent = src_sent_opt.unwrap();
        let tgt_sent = tgt_sent_opt.unwrap();
        let src_len = src_sent.tokens.len();
        let tgt_len = tgt_sent.tokens.len();

        let fwd = fwd_opt.as_ref().unwrap();
        let rev = rev_opt.as_ref().unwrap();

        // Validate vector lengths against sentence lengths
        if fwd.len() != tgt_len {
            return Err(format!(
                "Sentence {}: forward alignment length ({}) != target length ({})",
                s,
                fwd.len(),
                tgt_len
            ));
        }
        if rev.len() != src_len {
            return Err(format!(
                "Sentence {}: reverse alignment length ({}) != source length ({})",
                s,
                rev.len(),
                src_len
            ));
        }

        // Build S_fw and S_rev
        let mut s_fw: HashSet<(usize, usize)> = HashSet::new();
        for (j, &src_idx_link) in fwd.iter().enumerate() {
            if src_idx_link != NULL_LINK {
                let i = src_idx_link as usize;
                if i >= src_len {
                    return Err(format!(
                        "Sentence {}: forward index out of bounds: j={} -> i={} (src_len={})",
                        s, j, i, src_len
                    ));
                }
                s_fw.insert((i, j));
            }
        }
        let mut s_rev: HashSet<(usize, usize)> = HashSet::new();
        for (i, &tgt_idx_link) in rev.iter().enumerate() {
            if tgt_idx_link != NULL_LINK {
                let j = tgt_idx_link as usize;
                if j >= tgt_len {
                    return Err(format!(
                        "Sentence {}: reverse index out of bounds: i={} -> j={} (tgt_len={})",
                        s, i, j, tgt_len
                    ));
                }
                s_rev.insert((i, j));
            }
        }

        // Intersection A and union U
        let mut a: HashSet<(usize, usize)> =
            s_fw.intersection(&s_rev).copied().collect::<HashSet<_>>();
        let u: HashSet<(usize, usize)> =
            s_fw.union(&s_rev).copied().collect::<HashSet<_>>();

        // Track aligned status for fast checks
        let mut src_aligned = vec![false; src_len];
        let mut tgt_aligned = vec![false; tgt_len];
        for &(i, j) in &a {
            src_aligned[i] = true;
            tgt_aligned[j] = true;
        }

        // Grow-diag
        let mut changed = true;
        while changed {
            changed = false;
            // Snapshot A to avoid simultaneous iteration/modification issues
            let current = a.iter().copied().collect::<Vec<_>>();
            for &(i, j) in &current {
                // Enumerate 8-neighbours within bounds
                let mut candidates: [(isize, isize); 8] = [
                    (i as isize - 1, j as isize),     // up
                    (i as isize + 1, j as isize),     // down
                    (i as isize, j as isize - 1),     // left
                    (i as isize, j as isize + 1),     // right
                    (i as isize - 1, j as isize - 1), // up-left
                    (i as isize - 1, j as isize + 1), // up-right
                    (i as isize + 1, j as isize - 1), // down-left
                    (i as isize + 1, j as isize + 1), // down-right
                ];
                for &(ci, cj) in &candidates {
                    if ci < 0 || cj < 0 {
                        continue;
                    }
                    let i2 = ci as usize;
                    let j2 = cj as usize;
                    if i2 >= src_len || j2 >= tgt_len {
                        continue;
                    }
                    let pair = (i2, j2);
                    if u.contains(&pair) && !a.contains(&pair) && (!src_aligned[i2] || !tgt_aligned[j2]) {
                        a.insert(pair);
                        src_aligned[i2] = true;
                        tgt_aligned[j2] = true;
                        changed = true;
                    }
                }
            }
        }

        // Final (OR): add pairs from union if either side is unaligned
        for &(i, j) in &u {
            if !a.contains(&(i, j)) && (!src_aligned[i] || !tgt_aligned[j]) {
                a.insert((i, j));
                src_aligned[i] = true;
                tgt_aligned[j] = true;
            }
        }

        // Final-and: add remaining forward-only or reverse-only links that connect two unaligned words
        for &(i, j) in &s_fw {
            if !a.contains(&(i, j)) && !src_aligned[i] && !tgt_aligned[j] {
                a.insert((i, j));
                src_aligned[i] = true;
                tgt_aligned[j] = true;
            }
        }
        for &(i, j) in &s_rev {
            if !a.contains(&(i, j)) && !src_aligned[i] && !tgt_aligned[j] {
                a.insert((i, j));
                src_aligned[i] = true;
                tgt_aligned[j] = true;
            }
        }

        // Build per-target links; choose the smallest source index deterministically if multiple
        let mut sent_links = vec![NULL_LINK; tgt_len];
        for &(i, j) in &a {
            match sent_links[j] {
                v if v == NULL_LINK => sent_links[j] = i as Link,
                v if (i as usize) < v as usize => sent_links[j] = i as Link,
                _ => {}
            }
        }

        merged.push(Some(sent_links));
    }

    Ok(merged)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::text::{Sentence, Text};
    use crate::symmetrize::Link;
    use crate::types::{Token, NULL_LINK};

    fn mk_sentence(tokens: &[Token]) -> Sentence {
        Sentence { tokens: tokens.to_vec() }
    }

    #[test]
    fn symmetric_intersection_equals_union() {
        // Source and target with 3 tokens
        let source = Text {
            sentences: vec![Some(mk_sentence(&[1, 2, 3]))],
            n_sentences: 1,
            vocabulary_size: 3,
        };
        let target = Text {
            sentences: vec![Some(mk_sentence(&[10, 20, 30]))],
            n_sentences: 1,
            vocabulary_size: 3,
        };

        // Forward: for each target j, link to source i=j
        let forward: Vec<Option<Vec<Link>>> = vec![Some(vec![0 as Link, 1 as Link, 2 as Link])];
        // Reverse: for each source i, link to target j=i
        let reverse: Vec<Option<Vec<Link>>> = vec![Some(vec![0 as Link, 1 as Link, 2 as Link])];

        let merged = grow_diag_final_and(&forward, &reverse, &source, &target).unwrap();
        assert_eq!(merged.len(), 1);
        let sent = merged[0].as_ref().unwrap();
        assert_eq!(sent, &vec![0 as Link, 1 as Link, 2 as Link]);
    }

    #[test]
    fn asymmetric_grow_diag_then_final_forward() {
        // 3x3 example
        // forward: j=0->i=0, j=1->NULL, j=2->i=2
        // reverse: i=0->j=0, i=1->j=1, i=2->NULL
        // Steps:
        // - A = {(0,0)}
        // - U = {(0,0), (1,1), (2,2)}
        // - grow-diag adds (1,1) (diagonal neighbor)
        // - final adds (2,2) because i=2 or j=2 is unaligned
        // - final-and adds nothing further
        let source = Text {
            sentences: vec![Some(mk_sentence(&[1, 2, 3]))],
            n_sentences: 1,
            vocabulary_size: 3,
        };
        let target = Text {
            sentences: vec![Some(mk_sentence(&[10, 20, 30]))],
            n_sentences: 1,
            vocabulary_size: 3,
        };

        let forward: Vec<Option<Vec<Link>>> = vec![Some(vec![
            0 as Link,
            NULL_LINK,
            2 as Link,
        ])];
        let reverse: Vec<Option<Vec<Link>>> = vec![Some(vec![
            0 as Link,
            1 as Link,
            NULL_LINK,
        ])];

        let merged = grow_diag_final_and(&forward, &reverse, &source, &target).unwrap();
        assert_eq!(merged.len(), 1);
        let sent = merged[0].as_ref().unwrap();
        // Expect links for every target position: 0->0, 1->1 (from grow), 2->2 (from final)
        assert_eq!(sent, &vec![0 as Link, 1 as Link, 2 as Link]);
    }
}