//! Symmetrization utilities.

use std::collections::{HashSet, VecDeque};

use crate::types::{NULL_LINK};
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
///     - Use a work queue initialized with points from A.
///     - While the queue is not empty, pop a point and check its 8-neighbours. If a neighbour is in U
///       and not in A, and either its source or target index is currently unaligned in A, add the
///       neighbour to A and push it to the queue for further expansion.
///   - Final-and:
///     - For remaining (i, j) in S_fw \ A, if both i and j are unaligned in A, add it.
///     - Repeat for (i, j) in S_rev \ A.
///   - Collect all pairs from A into a sorted Vec of (source, target) pairs.
///
/// Errors:
/// - Returns Err(...) if the counts of sentences mismatch across inputs.
/// - Returns Err(...) if for a sentence both alignments and both sentences exist, but lengths mismatch,
///   or if any index in the alignments is out of bounds.
///
/// Note:
/// - The union/intersection are sets over (source_index, target_index).
/// - Returns sorted (source, target) pairs per sentence, supporting many-to-many alignments.
pub fn grow_diag_final_and(
    forward: &[Option<Vec<u16>>],
    reverse: &[Option<Vec<u16>>],
    source: &Text,
    target: &Text,
) -> Result<Vec<Option<Vec<(u16, u16)>>>, String> {
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
    let mut merged: Vec<Option<Vec<(u16, u16)>>> = Vec::with_capacity(n_sentences);

    for s in 0..n_sentences {
        let src_sent_opt = source.sentences[s].as_ref();
        let tgt_sent_opt = target.sentences[s].as_ref();
        let fwd_opt = &forward[s];
        let rev_opt = &reverse[s];

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

        // Initialize a work queue with the initial intersection points.
        // This ensures newly added points are processed for further expansion.
        let mut queue: VecDeque<(usize, usize)> = a.iter().copied().collect();
        
        while let Some((i, j)) = queue.pop_front() {
            // Enumerate 8-neighbours within bounds
            let candidates: [(isize, isize); 8] = [
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
                if ci >= 0 && cj >= 0 {
                    let i2 = ci as usize;
                    let j2 = cj as usize;
                    
                    if i2 < src_len && j2 < tgt_len {
                        let pair = (i2, j2);
                        // The condition: in Union, not yet in A, and at least one word is unaligned
                        if !a.contains(&pair) && u.contains(&pair) && (!src_aligned[i2] || !tgt_aligned[j2]) {
                            // Add to A, mark words as aligned, and add to queue for expansion
                            a.insert(pair);
                            src_aligned[i2] = true;
                            tgt_aligned[j2] = true;
                            queue.push_back(pair);
                        }
                    }
                }
            }
        }

        // Final-and: add remaining links connecting two unaligned words
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

        // Collect all alignment pairs, sorted for determinism
        let mut pairs: Vec<(u16, u16)> = a.into_iter()
            .map(|(i, j)| (i as u16, j as u16))
            .collect();
        pairs.sort();
        merged.push(Some(pairs));
    }

    Ok(merged)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::text::{Sentence, Text};
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
        let forward: Vec<Option<Vec<u16>>> = vec![Some(vec![0, 1, 2])];
        // Reverse: for each source i, link to target j=i
        let reverse: Vec<Option<Vec<u16>>> = vec![Some(vec![0, 1, 2])];

        let merged = grow_diag_final_and(&forward, &reverse, &source, &target).unwrap();
        assert_eq!(merged.len(), 1);
        let sent = merged[0].as_ref().unwrap();
        assert_eq!(sent, &vec![(0, 0), (1, 1), (2, 2)]);
    }

    #[test]
    fn asymmetric_grow_diag_then_final_and() {
        // 3x3 example
        // forward: j=0->i=0, j=1->NULL, j=2->i=2
        // reverse: i=0->j=0, i=1->j=1, i=2->NULL
        // Steps:
        // - A = {(0,0)}
        // - U = {(0,0), (1,1), (2,2)}
        // - grow-diag: (1,1) is diagonal neighbor of (0,0) and in U, src[1] unaligned -> add
        //              (2,2) is diagonal neighbor of (1,1) and in U, src[2] unaligned -> add
        // - final-and adds nothing further (all words aligned)
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

        let forward: Vec<Option<Vec<u16>>> = vec![Some(vec![
            0,
            NULL_LINK,
            2,
        ])];
        let reverse: Vec<Option<Vec<u16>>> = vec![Some(vec![
            0,
            1,
            NULL_LINK,
        ])];

        let merged = grow_diag_final_and(&forward, &reverse, &source, &target).unwrap();
        assert_eq!(merged.len(), 1);
        let sent = merged[0].as_ref().unwrap();
        assert_eq!(sent, &vec![(0, 0), (1, 1), (2, 2)]);
    }

    #[test]
    fn final_and_prefers_forward_over_reverse() {
        // 2x2 example where empty intersection means grow-diag does nothing,
        // and final-and with fwd runs first, blocking reverse pairs.
        // forward: j=0->i=1, j=1->i=0  => s_fw = {(1,0), (0,1)}
        // reverse: i=0->j=0, i=1->j=1  => s_rev = {(0,0), (1,1)}
        // Intersection = {}, Union = {(0,0), (0,1), (1,0), (1,1)}
        // Grow-diag: empty queue, nothing added.
        // Final-and with fwd: (1,0) - both unaligned -> add. (0,1) - both unaligned -> add.
        //   Now src[0,1] and tgt[0,1] all aligned.
        // Final-and with rev: (0,0) - src[0] aligned -> skip. (1,1) - src[1] aligned -> skip.
        // Result: {(0,1), (1,0)}
        let source = Text {
            sentences: vec![Some(mk_sentence(&[1, 2]))],
            n_sentences: 1,
            vocabulary_size: 2,
        };
        let target = Text {
            sentences: vec![Some(mk_sentence(&[10, 20]))],
            n_sentences: 1,
            vocabulary_size: 2,
        };

        let forward: Vec<Option<Vec<u16>>> = vec![Some(vec![1, 0])];
        let reverse: Vec<Option<Vec<u16>>> = vec![Some(vec![0, 1])];

        let merged = grow_diag_final_and(&forward, &reverse, &source, &target).unwrap();
        assert_eq!(merged.len(), 1);
        let sent = merged[0].as_ref().unwrap();
        // Final-and processes fwd first: adds (0,1) and (1,0); rev pairs blocked
        assert_eq!(sent, &vec![(0, 1), (1, 0)]);
    }
}