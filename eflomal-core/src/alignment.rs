use crate::types::*;
use crate::prng::{Pcg32, dirichlet_unnormalized};
use crate::text::{Text};
use hashbrown::HashMap;

#[derive(Clone, Debug)]
pub struct AlignOptions {
    pub model: u8,          // 1..3
    pub score_model: u8,    // 1..3
    pub n_iters: [usize; 3],// per model
    pub n_samplers: usize,  // >=1
    pub null_prior: Count,  // e.g., 0.2
    pub n_clean: Option<usize>, // None => all
    pub priors: Option<String>, // same text format as C
    pub reverse: bool,
    pub seed: u64,
}

impl Default for AlignOptions {
    fn default() -> Self {
        AlignOptions {
            model: 3,
            score_model: 3,
            n_iters: [1, 1, 1],
            n_samplers: 1,
            null_prior: 0.2,
            n_clean: None,
            priors: None,
            reverse: false,
            seed: 1,
        }
    }
}

#[derive(Clone)]
struct TA<'a> {
    model: u8,
    source: &'a Text,
    target: &'a Text,
    sentence_links: Vec<Option<Vec<Link>>>,
    source_prior: Option<Vec<HashMap<Token, f32>>>,
    source_prior_sum: Option<Vec<Count>>,
    has_jump_prior: bool,
    jump_prior: [Count; JUMP_ARRAY_LEN],
    fert_prior: Option<Vec<Count>>,
    source_count: Vec<HashMap<Token, u32>>,
    inv_source_count_sum: Vec<Count>,
    jump_counts: [Count; JUMP_ARRAY_LEN],
    fert_counts: Vec<Count>,
    n_clean: usize,
    null_prior: Count,
}

impl<'a> TA<'a> {
    fn create(source: &'a Text, target: &'a Text, null_prior: Count) -> Self {
        assert_eq!(source.n_sentences, target.n_sentences);
        let mut sentence_links = Vec::with_capacity(target.n_sentences);
        for i in 0..target.n_sentences {
            let tl = target.sentences[i].as_ref().map(|s| s.len()).unwrap_or(0);
            if target.sentences[i].is_some() && source.sentences[i].is_some() {
                sentence_links.push(Some(vec![NULL_LINK; tl]));
            } else {
                sentence_links.push(None);
            }
        }
        let vocab = source.vocabulary_size as usize;
        TA {
            model: 1,
            source,
            target,
            sentence_links,
            source_prior: None,
            source_prior_sum: None,
            has_jump_prior: false,
            jump_prior: [0.0; JUMP_ARRAY_LEN],
            fert_prior: None,
            source_count: (0..vocab).map(|_| HashMap::new()).collect(),
            inv_source_count_sum: vec![0.0; vocab],
            jump_counts: [0.0; JUMP_ARRAY_LEN],
            fert_counts: vec![0.0; vocab * FERT_ARRAY_LEN],
            n_clean: 0,
            null_prior,
        }
    }

    fn load_priors(&mut self, s: &str, reverse: bool) -> Result<(), String> {
        let mut it = s.lines();
        let header = it.next().ok_or("priors: missing header")?;
        let mut hp = header.split_whitespace();
        let mut src_vs: usize = hp.next().ok_or("priors: header")?.parse().map_err(|_|"bad")?;
        let mut trg_vs: usize = hp.next().ok_or("priors: header")?.parse().map_err(|_|"bad")?;
        let n_lex: usize = hp.next().ok_or("priors: header")?.parse().map_err(|_|"bad")?;
        let n_fwd_jump: usize = hp.next().ok_or("priors: header")?.parse().map_err(|_|"bad")?;
        let n_rev_jump: usize = hp.next().ok_or("priors: header")?.parse().map_err(|_|"bad")?;
        let n_fwd_fert: usize = hp.next().ok_or("priors: header")?.parse().map_err(|_|"bad")?;
        let n_rev_fert: usize = hp.next().ok_or("priors: header")?.parse().map_err(|_|"bad")?;

        let (mut n_jump, mut n_fert) = if reverse {(n_rev_jump, n_rev_fert)} else {(n_fwd_jump, n_fwd_fert)};
        if reverse { core::mem::swap(&mut src_vs, &mut trg_vs); }

        if src_vs != self.source.vocabulary_size as usize || trg_vs != self.target.vocabulary_size as usize {
            return Err("priors: vocabulary mismatch".into());
        }

        if n_lex > 0 {
            self.source_prior = Some((0..self.source.vocabulary_size as usize).map(|_| HashMap::new()).collect());
            self.source_prior_sum = Some(vec![0.0; self.source.vocabulary_size as usize]);
        }
        if n_fert > 0 {
            let sz = self.source.vocabulary_size as usize * FERT_ARRAY_LEN;
            self.fert_prior = Some(vec![0.0; sz]);
        }
        if n_fwd_jump > 0 || n_rev_jump > 0 {
            self.has_jump_prior = true;
            self.jump_prior.fill(0.0);
        }

        // Lexical priors
        for _ in 0..n_lex {
            let line = it.next().ok_or("priors: missing lex line")?;
            let mut sp = line.split_whitespace();
            let mut e: Token = sp.next().ok_or("priors: lex")?.parse().map_err(|_|"bad")?;
            let mut f: Token = sp.next().ok_or("priors: lex")?.parse().map_err(|_|"bad")?;
            let alpha: f32 = sp.next().ok_or("priors: lex")?.parse().map_err(|_|"bad")?;
            if reverse { core::mem::swap(&mut e, &mut f); }
            if let Some(vecmap) = &mut self.source_prior {
                vecmap[e as usize].insert(f, alpha);
            }
            if let Some(sum) = &mut self.source_prior_sum {
                sum[e as usize] += alpha as Count;
            }
        }
        if let Some(sum) = &mut self.source_prior_sum {
            for e in 0..sum.len() {
                sum[e] += LEX_ALPHA * (self.target.vocabulary_size as Count);
            }
        }

        // Jump priors
        for _ in 0..n_fwd_jump {
            let line = it.next().ok_or("priors: missing jump fwd")?;
            let mut sp = line.split_whitespace();
            let jump: i32 = sp.next().ok_or("priors:")?.parse().map_err(|_|"bad")?;
            let alpha: f32 = sp.next().ok_or("priors:")?.parse().map_err(|_|"bad")?;
            if !reverse {
                let idx = ((jump + (JUMP_ARRAY_LEN as i32)/2).clamp(0, (JUMP_ARRAY_LEN as i32)-1)) as usize;
                self.jump_prior[idx] += alpha as Count;
            }
        }
        for _ in 0..n_rev_jump {
            let line = it.next().ok_or("priors: missing jump rev")?;
            let mut sp = line.split_whitespace();
            let jump: i32 = sp.next().ok_or("priors:")?.parse().map_err(|_|"bad")?;
            let alpha: f32 = sp.next().ok_or("priors:")?.parse().map_err(|_|"bad")?;
            if reverse {
                let idx = ((jump + (JUMP_ARRAY_LEN as i32)/2).clamp(0, (JUMP_ARRAY_LEN as i32)-1)) as usize;
                self.jump_prior[idx] += alpha as Count;
            }
        }

        // Fert priors
        for _ in 0..n_fwd_fert {
            let line = it.next().ok_or("priors: missing fert fwd")?;
            let mut sp = line.split_whitespace();
            let e: Token = sp.next().ok_or("priors:")?.parse().map_err(|_|"bad")?;
            let k: usize = sp.next().ok_or("priors:")?.parse().map_err(|_|"bad")?;
            let alpha: f32 = sp.next().ok_or("priors:")?.parse().map_err(|_|"bad")?;
            if !reverse {
                if let Some(fert) = &mut self.fert_prior {
                    let idx = get_fert_index(e, k);
                    if idx >= fert.len() { return Err("priors: fert idx oob".into()); }
                    fert[idx] += alpha as Count;
                }
            }
        }
        for _ in 0..n_rev_fert {
            let line = it.next().ok_or("priors: missing fert rev")?;
            let mut sp = line.split_whitespace();
            let e: Token = sp.next().ok_or("priors:")?.parse().map_err(|_|"bad")?;
            let k: usize = sp.next().ok_or("priors:")?.parse().map_err(|_|"bad")?;
            let alpha: f32 = sp.next().ok_or("priors:")?.parse().map_err(|_|"bad")?;
            if reverse {
                if let Some(fert) = &mut self.fert_prior {
                    let idx = get_fert_index(e, k);
                    if idx >= fert.len() { return Err("priors: fert idx oob".into()); }
                    fert[idx] += alpha as Count;
                }
            }
        }

        Ok(())
    }

    fn randomize(&mut self, rng: &mut Pcg32) {
        for s in 0..self.target.n_sentences {
            if self.sentence_links[s].is_none() { continue; }
            let tgt = self.target.sentences[s].as_ref().unwrap();
            let src = self.source.sentences[s].as_ref().unwrap();
            let nsrc = src.len();
            let links = self.sentence_links[s].as_mut().unwrap();
            for j in 0..tgt.len() {
                if rng.next_f32() < self.null_prior as f32 {
                    links[j] = NULL_LINK;
                } else {
                    links[j] = rng.next_usize(nsrc) as Link;
                }
            }
        }
    }

    fn make_counts(&mut self) {
        let model = self.model;
        let vocab = self.source.vocabulary_size as usize;

        for e in 0..vocab {
            self.source_count[e].clear();
            self.inv_source_count_sum[e] = if let Some(sum) = &self.source_prior_sum {
                sum[e]
            } else {
                LEX_ALPHA * (self.target.vocabulary_size as Count)
            };
        }

        if model >= 2 {
            if self.has_jump_prior {
                self.jump_counts[JUMP_SUM] = JUMP_MAX_EST * JUMP_ALPHA;
                for i in 0..JUMP_ARRAY_LEN-1 {
                    self.jump_counts[i] = self.jump_prior[i] + JUMP_ALPHA;
                    self.jump_counts[JUMP_SUM] += self.jump_prior[i];
                }
            } else {
                for i in 0..JUMP_ARRAY_LEN-1 { self.jump_counts[i] = JUMP_ALPHA; }
                self.jump_counts[JUMP_SUM] = JUMP_MAX_EST * JUMP_ALPHA;
            }
        }

        let n_sentences = self.n_clean.max(1);
        let n_use = if self.n_clean == 0 { self.target.n_sentences } else { self.n_clean };

        for s in 0..n_use {
            if self.sentence_links[s].is_none() { continue; }
            let src = self.source.sentences[s].as_ref().unwrap();
            let tgt = self.target.sentences[s].as_ref().unwrap();
            let src_len = src.len();
            let links = self.sentence_links[s].as_ref().unwrap();
            let mut aa_jm1: isize = -1;
            for (j, &li) in links.iter().enumerate() {
                let e = if li == NULL_LINK { 0 } else { src.tokens[li as usize] };
                let f = tgt.tokens[j];
                self.inv_source_count_sum[e as usize] += 1.0;
                *self.source_count[e as usize].entry(f).or_insert(0) += 1;

                if model >= 2 && e != 0 {
                    let jump = get_jump_index(aa_jm1, li as isize, src_len);
                    aa_jm1 = li as isize;
                    self.jump_counts[jump] += 1.0;
                    self.jump_counts[JUMP_SUM] += 1.0;
                }
            }
            if model >= 2 && aa_jm1 >= 0 {
                let jump = get_jump_index(aa_jm1, src_len as isize, src_len);
                self.jump_counts[jump] += 1.0;
                self.jump_counts[JUMP_SUM] += 1.0;
            }
        }

        for e in 0..vocab {
            self.inv_source_count_sum[e] = 1.0 / self.inv_source_count_sum[e];
        }
    }

    fn sample(
        &mut self,
        rng: &mut Pcg32,
        mut sentence_scores: Option<&mut [Count]>,
        tas: Option<&[&TA<'a>]>,
        _n_samplers: usize,
    ) {
        // ARGMAX consensus pass across samplers: accumulate distributions and pick best alignment.
        if let Some(others) = tas {
            let model = self.model;
            for s in 0..self.target.n_sentences {
                if self.sentence_links[s].is_none() { continue; }
                let src = self.source.sentences[s].as_ref().unwrap();
                let tgt = self.target.sentences[s].as_ref().unwrap();
                let src_len = src.len();
                let tgt_len = tgt.len();

                // Precompute aa_jp1/aa_jm1 and fert for self
                let self_links = self.sentence_links[s].as_ref().unwrap();
                let (self_aa_jp1, self_aa_jm1) = {
                    let mut aa_jp1 = vec![src_len as isize; tgt_len];
                    let mut cur = src_len as isize;
                    for j in (0..tgt_len).rev() {
                        aa_jp1[j] = cur;
                        if self_links[j] != NULL_LINK { cur = self_links[j] as isize; }
                    }
                    let mut aa_jm1 = vec![-1isize; tgt_len];
                    let mut lcur = -1isize;
                    for j in 0..tgt_len {
                        aa_jm1[j] = lcur;
                        if self_links[j] != NULL_LINK { lcur = self_links[j] as isize; }
                    }
                    (aa_jp1, aa_jm1)
                };
                let self_fert = if model >= 3 {
                    let mut f = vec![0usize; src_len];
                    for &li in self_links.iter() { if li != NULL_LINK { f[li as usize] += 1; } }
                    Some(f)
                } else { None };

                // Precompute for other samplers
                struct Pre<'b, 'a> {
                    ta: &'b TA<'a>,
                    links: &'b [Link],
                    aa_jp1: Vec<isize>,
                    aa_jm1: Vec<isize>,
                    fert: Option<Vec<usize>>,
                }
                let mut pres: Vec<Pre> = Vec::new();
                for ta in others.iter() {
                    if ta.sentence_links[s].is_none() { continue; }
                    let links = ta.sentence_links[s].as_ref().unwrap();
                    let mut aa_jp1 = vec![src_len as isize; tgt_len];
                    let mut cur = src_len as isize;
                    for j in (0..tgt_len).rev() {
                        aa_jp1[j] = cur;
                        if links[j] != NULL_LINK { cur = links[j] as isize; }
                    }
                    let mut aa_jm1 = vec![-1isize; tgt_len];
                    let mut lcur = -1isize;
                    for j in 0..tgt_len {
                        aa_jm1[j] = lcur;
                        if links[j] != NULL_LINK { lcur = links[j] as isize; }
                    }
                    let fert = if model >= 3 {
                        let mut f = vec![0usize; src_len];
                        for &li in links.iter() { if li != NULL_LINK { f[li as usize] += 1; } }
                        Some(f)
                    } else { None };
                    pres.push(Pre { ta, links, aa_jp1, aa_jm1, fert });
                }

                // For each target position, accumulate probabilities across samplers
                for j in 0..tgt_len {
                    let f_tok = tgt.tokens[j];
                    let mut acc = vec![0.0 as Count; src_len + 1];

                    let mut add_from_ta = |ta: &TA, aa_jm1: isize, aa_jp1: isize, fert: Option<&[usize]>| {
                        let mut tmp = vec![0.0 as Count; src_len + 1];
                        let mut sum: Count = 0.0;

                        // non-NULL positions
                        for i in 0..src_len {
                            let e = src.tokens[i];
                            let n = ta.source_count[e as usize].get(&f_tok).copied().unwrap_or(0) as Count;
                            let alpha = if let Some(pr) = &ta.source_prior {
                                pr[e as usize].get(&f_tok).copied().unwrap_or(0.0) as Count + LEX_ALPHA
                            } else { LEX_ALPHA };
                            let mut term = ta.inv_source_count_sum[e as usize] * (alpha + n);
                            if model >= 2 {
                                let j1 = get_jump_index(aa_jm1, i as isize, src_len);
                                let j2 = get_jump_index(i as isize, aa_jp1, src_len);
                                term *= ta.jump_counts[j1] * ta.jump_counts[j2];
                            }
                            if model >= 3 {
                                let fert_idx = get_fert_index(e, fert.unwrap()[i] + 1);
                                term *= ta.fert_counts[fert_idx];
                            }
                            tmp[i] = term;
                            sum += term;
                        }

                        // NULL position
                        let null_n = ta.source_count[0].get(&f_tok).copied().unwrap_or(0) as Count;
                        let mut null_term = ta.null_prior * ta.inv_source_count_sum[0] * (NULL_ALPHA + null_n);
                        if model >= 2 {
                            let skip = get_jump_index(aa_jm1, aa_jp1, src_len);
                            null_term *= ta.jump_counts[JUMP_SUM] * ta.jump_counts[skip];
                        }
                        tmp[src_len] = null_term;
                        sum += null_term;

                        if sum > 0.0 {
                            let inv = 1.0 / sum;
                            for k in 0..tmp.len() {
                                acc[k] += tmp[k] * inv;
                            }
                        }
                    };

                    // others
                    for pre in pres.iter() {
                        add_from_ta(pre.ta, pre.aa_jm1[j], pre.aa_jp1[j], pre.fert.as_deref());
                    }
                    // self
                    add_from_ta(self, self_aa_jm1[j], self_aa_jp1[j], self_fert.as_deref());

                    // pick argmax and write to self
                    let mut best_k = 0usize;
                    let mut best_v = acc[0];
                    for k in 1..acc.len() {
                        let v = acc[k];
                        if v > best_v { best_v = v; best_k = k; }
                    }
                    let out_links = self.sentence_links[s].as_mut().unwrap();
                    if best_k == src_len { out_links[j] = NULL_LINK; }
                    else { out_links[j] = best_k as Link; }
                }
            }
            return;
        }

        let model = self.model;
        let src_sents = &self.source.sentences;
        let tgt_sents = &self.target.sentences;

        // Fertility sampling prep (model >= 3)
        if model >= 3 {
            let vocab = self.source.vocabulary_size as usize;
            let n_use = if self.n_clean == 0 { self.target.n_sentences } else { self.n_clean };

            // Initialize fert counts
            if let Some(prior) = &self.fert_prior {
                for i in 0..self.fert_counts.len() {
                    self.fert_counts[i] = prior[i] + FERT_ALPHA;
                }
            } else {
                for i in 0..self.fert_counts.len() {
                    self.fert_counts[i] = FERT_ALPHA;
                }
            }

            let mut e_count = vec![0usize; vocab];
            let mut fert = vec![0usize; MAX_SENT_LEN];

            for s in 0..n_use {
                if self.sentence_links[s].is_none() { continue; }
                let src = src_sents[s].as_ref().unwrap();
                let tgt = tgt_sents[s].as_ref().unwrap();
                let links = self.sentence_links[s].as_ref().unwrap();
                let src_len = src.len();
                for i in 0..src_len { fert[i] = 0; }
                for j in 0..tgt.len() {
                    if links[j] != NULL_LINK { fert[links[j] as usize] += 1; }
                }
                for i in 0..src_len {
                    let e = src.tokens[i] as usize;
                    e_count[e] += 1;
                    let idx = get_fert_index(src.tokens[i], fert[i]);
                    self.fert_counts[idx] += 1.0;
                }
            }

            // Sample categorical fertility distributions
            for e in 1..(self.source.vocabulary_size as usize) {
                if e_count[e] == 0 { continue; }
                // alpha vector
                let mut alpha = [0f64; FERT_ARRAY_LEN];
                let base = e * FERT_ARRAY_LEN;
                for k in 0..FERT_ARRAY_LEN {
                    alpha[k] = self.fert_counts[base + k] as f64;
                }
                let mut out = [0f64; FERT_ARRAY_LEN];
                dirichlet_unnormalized(&alpha, &mut out, rng);
                // store ratio P(phi(i))/P(phi(i)-1)
                // index 0 unused by caller, leave as is; last set small
                out[FERT_ARRAY_LEN-1] = 1e-10;
                for k in (1..FERT_ARRAY_LEN-1).rev() {
                    out[k] = out[k] / out[k-1].max(1e-20);
                }
                for k in 0..FERT_ARRAY_LEN {
                    self.fert_counts[base+k] = out[k] as Count;
                }
            }
        }

        let n_sentences = self.target.n_sentences;
        let n_use = if self.n_clean == 0 { n_sentences } else { self.n_clean };

        for s in 0..n_sentences {
            if self.sentence_links[s].is_none() { continue; }
            let src = src_sents[s].as_ref().unwrap();
            let tgt = tgt_sents[s].as_ref().unwrap();
            let src_len = src.len();
            let tgt_len = tgt.len();
            let links = self.sentence_links[s].as_mut().unwrap();

            let mut fert = vec![0usize; src_len];
            if model >= 3 {
                for i in 0..src_len { fert[i] = 0; }
                for &li in links.iter() {
                    if li != NULL_LINK { fert[li as usize] += 1; }
                }
            }

            // nearest non-NULL to the right for HMM
            let mut aa_jp1_table = vec![src_len as isize; tgt_len];
            if model >= 2 {
                let mut aa_jp1: isize = src_len as isize;
                for j in (0..tgt_len).rev() {
                    aa_jp1_table[j] = aa_jp1;
                    if links[j] != NULL_LINK { aa_jp1 = links[j] as isize; }
                }
            }

            // left neighbor
            let mut aa_jm1: isize = -1;

            // sampling
            for j in 0..tgt_len {
                let f = tgt.tokens[j];
                let old_i = links[j];
                let old_e = if old_i == NULL_LINK { 0 } else { src.tokens[old_i as usize] };

                if model >= 3 && old_i != NULL_LINK {
                    fert[old_i as usize] = fert[old_i as usize].saturating_sub(1);
                }
                let aa_jp1 = aa_jp1_table[j];

                // remove old counts if within n_use
                if s < n_use {
                    // update inv sum
                    let inv = self.inv_source_count_sum[old_e as usize];
                    self.inv_source_count_sum[old_e as usize] = 1.0 / (1.0/inv - 1.0);

                    // decrement lexical count
                    if old_e as usize >= self.source_count.len() { continue; }
                    if let Some(c) = self.source_count[old_e as usize].get_mut(&f) {
                        if *c > 1 { *c -= 1; } else { self.source_count[old_e as usize].remove(&f); }
                    }

                    if model >= 2 {
                        let skip_jump = get_jump_index(aa_jm1, aa_jp1, src_len);
                        if old_i == NULL_LINK {
                            self.jump_counts[JUMP_SUM] -= 1.0;
                            self.jump_counts[skip_jump] -= 1.0;
                        } else {
                            let j1 = get_jump_index(aa_jm1, old_i as isize, src_len);
                            let j2 = get_jump_index(old_i as isize, aa_jp1, src_len);
                            self.jump_counts[JUMP_SUM] -= 2.0;
                            self.jump_counts[j1] -= 1.0;
                            self.jump_counts[j2] -= 1.0;
                        }
                    }
                }

                // build distribution
                let mut ps = vec![0.0 as Count; src_len + 1];
                let mut ps_sum: Count = 0.0;

                let mut null_n = 0u32;
                if let Some(c) = self.source_count[0].get(&f) { null_n = *c; }

                if model >= 3 {
                    for i in 0..src_len {
                        let e = src.tokens[i];
                        let n = self.source_count[e as usize].get(&f).copied().unwrap_or(0) as Count;
                        let alpha = if let Some(pr) = &self.source_prior {
                            let a = pr[e as usize].get(&f).copied().unwrap_or(0.0) as Count + LEX_ALPHA;
                            a
                        } else {
                            LEX_ALPHA
                        };
                        let jump1 = get_jump_index(aa_jm1, i as isize, src_len);
                        let jump2 = get_jump_index(i as isize, aa_jp1, src_len);
                        let fert_idx = get_fert_index(e, fert[i] + 1);
                        let fert_w = self.fert_counts[fert_idx];
                        ps_sum += self.inv_source_count_sum[e as usize] * (alpha + n)
                                * self.jump_counts[jump1] * self.jump_counts[jump2] * fert_w;
                        ps[i] = ps_sum;
                    }
                    // scoring max p (non-null) if requested
                    if let Some(scores) = sentence_scores.as_deref_mut() {
                        let mut max_p = 0.0;
                        for i in 0..src_len {
                            let p = if i == 0 { ps[0] } else { ps[i] - ps[i-1] };
                            if p > max_p { max_p = p; }
                        }
                        ps_sum += self.null_prior * self.inv_source_count_sum[0] *
                                  (NULL_ALPHA + null_n as Count) *
                                  self.jump_counts[JUMP_SUM] *
                                  self.jump_counts[get_jump_index(aa_jm1, aa_jp1, src_len)];
                        let denom = (self.jump_counts[JUMP_SUM] * self.jump_counts[JUMP_SUM]).max(1e-30);
                        scores[s] += (max_p / denom).ln() as Count;
                    } else {
                        ps_sum += self.null_prior * self.inv_source_count_sum[0] *
                                  (NULL_ALPHA + null_n as Count) *
                                  self.jump_counts[JUMP_SUM] *
                                  self.jump_counts[get_jump_index(aa_jm1, aa_jp1, src_len)];
                    }
                } else if model >= 2 {
                    for i in 0..src_len {
                        let e = src.tokens[i];
                        let n = self.source_count[e as usize].get(&f).copied().unwrap_or(0) as Count;
                        let alpha = if let Some(pr) = &self.source_prior {
                            pr[e as usize].get(&f).copied().unwrap_or(0.0) as Count + LEX_ALPHA
                        } else { LEX_ALPHA };
                        let j1 = get_jump_index(aa_jm1, i as isize, src_len);
                        let j2 = get_jump_index(i as isize, aa_jp1, src_len);
                        ps_sum += self.inv_source_count_sum[e as usize] * (alpha + n)
                                * self.jump_counts[j1] * self.jump_counts[j2];
                        ps[i] = ps_sum;
                    }
                    if let Some(scores) = sentence_scores.as_deref_mut() {
                        let mut max_p = 0.0;
                        for i in 0..src_len {
                            let p = if i == 0 { ps[0] } else { ps[i] - ps[i-1] };
                            if p > max_p { max_p = p; }
                        }
                        ps_sum += self.null_prior * self.inv_source_count_sum[0] *
                                  (NULL_ALPHA + null_n as Count) *
                                  self.jump_counts[JUMP_SUM] *
                                  self.jump_counts[get_jump_index(aa_jm1, aa_jp1, src_len)];
                        let denom = (self.jump_counts[JUMP_SUM] * self.jump_counts[JUMP_SUM]).max(1e-30);
                        scores[s] += (max_p / denom).ln() as Count;
                    } else {
                        ps_sum += self.null_prior * self.inv_source_count_sum[0] *
                                  (NULL_ALPHA + null_n as Count) *
                                  self.jump_counts[JUMP_SUM] *
                                  self.jump_counts[get_jump_index(aa_jm1, aa_jp1, src_len)];
                    }
                } else {
                    for i in 0..src_len {
                        let e = src.tokens[i];
                        let n = self.source_count[e as usize].get(&f).copied().unwrap_or(0) as Count;
                        let alpha = if let Some(pr) = &self.source_prior {
                            pr[e as usize].get(&f).copied().unwrap_or(0.0) as Count + LEX_ALPHA
                        } else { LEX_ALPHA };
                        ps_sum += self.inv_source_count_sum[e as usize] * (alpha + n);
                        ps[i] = ps_sum;
                    }
                    if let Some(scores) = sentence_scores.as_deref_mut() {
                        let mut max_p = 0.0;
                        for i in 0..src_len {
                            let p = if i == 0 { ps[0] } else { ps[i] - ps[i-1] };
                            if p > max_p { max_p = p; }
                        }
                        scores[s] += max_p.ln() as Count;
                    }
                    ps_sum += self.null_prior * self.inv_source_count_sum[0] *
                              (NULL_ALPHA + null_n as Count);
                }
                ps[src_len] = ps_sum;

                let new_i = if sentence_scores.is_some() {
                    if old_i == NULL_LINK { src_len as Link } else { old_i }
                } else {
                    let u = (rng.next_f32() as Count) * ps_sum.max(1e-30);
                    let mut k = 0usize;
                    while k < src_len && ps[k] < u { k += 1; }
                    k as Link
                };

                let new_e = if new_i as usize == src_len {
                    links[j] = NULL_LINK; 0
                } else {
                    links[j] = new_i;
                    if model >= 3 {
                        fert[new_i as usize] += 1;
                    }
                    src.tokens[new_i as usize]
                };

                // add new counts if within n_use
                if s < n_use {
                    // remove zero entries already handled above
                    self.inv_source_count_sum[new_e as usize] = 1.0 / (1.0/self.inv_source_count_sum[new_e as usize] + 1.0);
                    *self.source_count[new_e as usize].entry(f).or_insert(0) += 1;

                    if model >= 2 {
                        let skip_jump = get_jump_index(aa_jm1, aa_jp1, src_len);
                        if new_e == 0 {
                            self.jump_counts[JUMP_SUM] += 1.0;
                            self.jump_counts[skip_jump] += 1.0;
                        } else {
                            let j1 = get_jump_index(aa_jm1, new_i as isize, src_len);
                            let j2 = get_jump_index(new_i as isize, aa_jp1, src_len);
                            self.jump_counts[JUMP_SUM] += 2.0;
                            self.jump_counts[j1] += 1.0;
                            self.jump_counts[j2] += 1.0;
                        }
                    }
                }

                if model >= 2 && new_e != 0 { aa_jm1 = new_i as isize; }
            }

            if let Some(scores) = sentence_scores.as_deref_mut() {
                if tgt_len > 0 {
                    scores[s] /= tgt_len as Count;
                }
            }
        }
    }
}


// calculate_iterations calculates the number of iterations for each model (following the Python script in the eflomal repo)
pub fn calculate_iterations(n_sentences: usize, model: u8) -> (usize, usize, usize) {
    // For some reason I have this set to 1 in previous work on eflomal, so I'm keeping it here.
    let rel_iterations = 1.0;
    let iters = (rel_iterations * 5000.0 / (n_sentences as f64).sqrt()).round().max(2.0) as usize;
    let iters4 = ((iters as f64) / 4.0).max(1.0) as usize;

    match model {
        1 => (iters, 0, 0),
        2 => (iters4.max(2), iters, 0),
        _ => (iters4.max(2), iters4, iters),
    }
}

#[derive(Debug)]
pub struct AlignResult {
    pub links_moses: String,
    pub stats: Option<String>,
    pub forward_scores: Option<String>,
    pub reverse_scores: Option<String>,
    pub links_vec: Option<Vec<Option<Vec<Link>>>>,
}

pub fn align(
    reverse: bool,
    source: &Text,
    target: &Text,
    opts: &AlignOptions,
    want_links: bool,
    want_stats: bool,
    want_scores: bool,
) -> Result<AlignResult, String> {
    if source.n_sentences != target.n_sentences {
        return Err("source/target sentence count mismatch".into());
    }
    // Setup samplers
    let mut samplers: Vec<TA> = (0..opts.n_samplers).map(|_| {
        let mut ta = TA::create(if reverse { target } else { source }, if reverse { source } else { target }, opts.null_prior);
        ta
    }).collect();

    if let Some(p) = &opts.priors {
        for ta in samplers.iter_mut() {
            ta.load_priors(p, reverse)?;
        }
    }

    // Randomize
    let mut rng_master = Pcg32::new(opts.seed, 54);
    let mut rngs: Vec<Pcg32> = (0..opts.n_samplers).map(|_| rng_master.split()).collect();

    for (i, ta) in samplers.iter_mut().enumerate() {
        let r = &mut rngs[i];
        ta.randomize(r);
    }

    // Train per model
    for m in 1..=opts.model {
        let iters = opts.n_iters[(m as usize)-1];
        if iters == 0 { continue; }
        for ta in samplers.iter_mut() {
            ta.model = m;
            if let Some(nc) = opts.n_clean { ta.n_clean = nc; } else { ta.n_clean = 0; }
            ta.make_counts();
        }
        for _ in 0..iters {
            for i in 0..samplers.len() {
                let r = &mut rngs[i];
                samplers[i].sample(r, None, None, 1);
            }
        }
    }

    // Final argmax iteration: just use first sampler for simplicity
    // Consensus argmax across samplers; write result into samplers[0]
    let (first, rest) = samplers.split_at_mut(1);
    let ta0 = &mut first[0];
    let others: Vec<&TA> = rest.iter().map(|t| t).collect();
    ta0.sample(&mut rngs[0], None, Some(&others), opts.n_samplers);

    // Outputs
    let links_moses = if want_links {
        crate::text::write_moses(&ta0.sentence_links, ta0.target, reverse)
    } else { String::new() };

    let stats = if want_stats {
        Some(crate::text::write_stats(&ta0.jump_counts))
    } else { None };

    let (forward_scores, reverse_scores) = if want_scores {
        // Score forward
        let mut scores_fwd = vec![0.0 as Count; ta0.source.n_sentences];
        ta0.model = opts.score_model;
        ta0.sample(&mut rngs[0], Some(&mut scores_fwd), None, 1);
        let fwd = crate::text::write_scores(&scores_fwd);

        // Score reverse if needed: run again with reversed
        (Some(fwd), None)
    } else { (None, None) };
    
    let links_vec = if want_links { Some(ta0.sentence_links.clone()) } else { None };

    Ok(AlignResult { links_moses, stats, forward_scores, reverse_scores, links_vec })
}