use crate::types::*;
use crate::prng::{Pcg32, dirichlet_unnormalized, logapprox};
use crate::text::Text;
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

    fn resample_fertility(&mut self, rng: &mut Pcg32) {
        if self.model < 3 { return; }

        let vocab = self.source.vocabulary_size as usize;
        let n_use = if self.n_clean == 0 { self.target.n_sentences } else { self.n_clean };

        // Initialize fertility counts with priors + FERT_ALPHA
        if let Some(prior) = &self.fert_prior {
            for i in 0..self.fert_counts.len() {
                self.fert_counts[i] = prior[i] + FERT_ALPHA;
            }
        } else {
            for x in self.fert_counts.iter_mut() {
                *x = FERT_ALPHA;
            }
        }

        // Collect fert stats from the corpus
        let mut e_count = vec![0usize; vocab];
        let mut fert = vec![0usize; MAX_SENT_LEN];
        for s in 0..n_use {
            if self.sentence_links[s].is_none() { continue; }
            let src = self.source.sentences[s].as_ref().unwrap();
            let links = self.sentence_links[s].as_ref().unwrap();
            let src_len = src.len();

            for i in 0..src_len { fert[i] = 0; }
            for &li in links.iter() {
                if li != NULL_LINK { fert[li as usize] += 1; }
            }
            for i in 0..src_len {
                let e = src.tokens[i] as usize;
                e_count[e] += 1;
                let idx = get_fert_index(src.tokens[i], fert[i]);
                self.fert_counts[idx] += 1.0;
            }
        }

        // Sample from Dirichlet posterior for each source type and store ratios
        for e in 1..vocab {
            if e_count[e] == 0 { continue; }
            
            let base_idx = e * FERT_ARRAY_LEN;
            let fert_slice = &mut self.fert_counts[base_idx..base_idx + FERT_ARRAY_LEN];
            
            // Prepare alpha parameters for Dirichlet from current counts
            let mut alpha = [0.0; FERT_ARRAY_LEN];
            for k in 0..FERT_ARRAY_LEN {
                alpha[k] = fert_slice[k] as f64;
            }

            // Sample unnormalized probabilities from Dirichlet
            let mut prob_sample = [0.0; FERT_ARRAY_LEN];
            dirichlet_unnormalized(&alpha, &mut prob_sample, rng);

            // This part is critical to match the C implementation precisely.
            // C code:
            // buf[FERT_ARRAY_LEN-1] = (count) 1e-10;
            // for (size_t i=FERT_ARRAY_LEN-2; i; i--)
            //     buf[i] /= buf[i-1];
            //
            // This calculates P(k)/P(k-1) for k > 0, leaving P(0) as is.
            // The loop must run backwards to use the original probability values.
            
            // Store the ratios back into the main fert_counts array.
            fert_slice[FERT_ARRAY_LEN - 1] = 1e-10;
            for k in (1..FERT_ARRAY_LEN - 1).rev() {
                prob_sample[k] /= prob_sample[k - 1].max(1e-30);
            }
            // Handle k=0 separately as it's not a ratio
            fert_slice[0] = prob_sample[0] as Count;
            // Store the calculated ratios
            for k in 1..FERT_ARRAY_LEN-1 {
                fert_slice[k] = prob_sample[k] as Count;
            }
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

        let n_fert = if reverse {n_rev_fert} else {n_fwd_fert};
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
                *vecmap[e as usize].entry(f).or_insert(0.0) += alpha;
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
                    let jump = get_jump_index(aa_jm1, li as isize);
                    aa_jm1 = li as isize;
                    self.jump_counts[jump] += 1.0;
                    self.jump_counts[JUMP_SUM] += 1.0;
                }
            }
            if model >= 2 && aa_jm1 >= 0 {
                let jump = get_jump_index(aa_jm1, src_len as isize);
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
) {
    // Training/scoring path (single sampler)
    if self.model >= 3 {
        self.resample_fertility(rng);
    }

    let src_sents = &self.source.sentences;
    let tgt_sents = &self.target.sentences;

    let n_sentences = self.target.n_sentences;
    let n_use = if self.n_clean == 0 { n_sentences } else { self.n_clean };

    for s in 0..n_sentences {
        if self.sentence_links[s].is_none() {
            continue;
        }
        let src = src_sents[s].as_ref().unwrap();
        let tgt = tgt_sents[s].as_ref().unwrap();
        let src_len = src.len();
        let tgt_len = tgt.len();
        let links = self.sentence_links[s].as_mut().unwrap();

        let mut fert = if self.model >= 3 {
            let mut f = vec![0usize; src_len];
            for &li in links.iter() {
                if li != NULL_LINK {
                    f[li as usize] += 1;
                }
            }
            Some(f)
        } else {
            None
        };

        let mut aa_jp1_table = vec![src_len as isize; tgt_len];
        if self.model >= 2 {
            let mut aa_jp1: isize = src_len as isize;
            for j in (0..tgt_len).rev() {
                aa_jp1_table[j] = aa_jp1;
                if links[j] != NULL_LINK {
                    aa_jp1 = links[j] as isize;
                }
            }
        }

        let mut aa_jm1: isize = -1;

        for j in 0..tgt_len {
            let f_tok = tgt.tokens[j];
            let old_i = links[j];
            let old_e = if old_i == NULL_LINK { 0 } else { src.tokens[old_i as usize] };

            if self.model >= 3 {
                if old_i != NULL_LINK {
                    if let Some(ref mut fv) = fert {
                        fv[old_i as usize] = fv[old_i as usize].saturating_sub(1);
                    }
                }
            }

            let aa_jp1 = aa_jp1_table[j];

            if s < n_use {
                // inv sum decrement for old_e
                let inv = self.inv_source_count_sum[old_e as usize];
                self.inv_source_count_sum[old_e as usize] =
                    1.0 / ((1.0 / inv) - 1.0).max(1e-30);

                // decrement lexical(e,f)
                if let Some(c) = self.source_count[old_e as usize].get_mut(&f_tok) {
                    if *c > 1 {
                        *c -= 1;
                    } else {
                        self.source_count[old_e as usize].remove(&f_tok);
                    }
                }

                if self.model >= 2 {
                    let skip_jump = get_jump_index(aa_jm1, aa_jp1);
                    if old_i == NULL_LINK {
                        self.jump_counts[JUMP_SUM] -= 1.0;
                        self.jump_counts[skip_jump] -= 1.0;
                    } else {
                        let j1 = get_jump_index(aa_jm1, old_i as isize);
                        let j2 = get_jump_index(old_i as isize, aa_jp1);
                        self.jump_counts[JUMP_SUM] -= 2.0;
                        self.jump_counts[j1] -= 1.0;
                        self.jump_counts[j2] -= 1.0;
                    }
                }
            }

            let mut ps = vec![0.0 as Count; src_len + 1];
            let mut ps_sum: Count = 0.0;

            let null_n = self.source_count[0].get(&f_tok).copied().unwrap_or(0) as Count;

            if self.model >= 3 {
                for i in 0..src_len {
                    let e = src.tokens[i];
                    let n = self.source_count[e as usize].get(&f_tok).copied().unwrap_or(0) as Count;
                    let alpha = if let Some(pr) = &self.source_prior {
                        pr[e as usize].get(&f_tok).copied().unwrap_or(0.0) as Count + LEX_ALPHA
                    } else { LEX_ALPHA };
                    let j1 = get_jump_index(aa_jm1, i as isize);
                    let j2 = get_jump_index(i as isize, aa_jp1);
                    let fi = if let Some(ref fv) = fert { fv[i] + 1 } else { 1 };
                    let fert_idx = get_fert_index(e, fi);
                    let fert_w = self.fert_counts[fert_idx];

                    ps_sum += self.inv_source_count_sum[e as usize]
                        * (alpha + n)
                        * self.jump_counts[j1]
                        * self.jump_counts[j2]
                        * fert_w;
                    ps[i] = ps_sum;
                }

                if let Some(scores) = sentence_scores.as_deref_mut() {
                    let mut max_p = 0.0;
                    for i in 0..src_len {
                        let p = if i == 0 { ps[0] } else { ps[i] - ps[i - 1] };
                        if p > max_p { max_p = p; }
                    }
                    ps_sum += self.null_prior
                        * self.inv_source_count_sum[0]
                        * (NULL_ALPHA + null_n)
                        * self.jump_counts[JUMP_SUM]
                        * self.jump_counts[get_jump_index(aa_jm1, aa_jp1)];
                    let denom = self.jump_counts[JUMP_SUM] * self.jump_counts[JUMP_SUM];
                    scores[s] += logapprox((max_p / denom) as f32) as Count;
                } else {
                    ps_sum += self.null_prior
                        * self.inv_source_count_sum[0]
                        * (NULL_ALPHA + null_n)
                        * self.jump_counts[JUMP_SUM]
                        * self.jump_counts[get_jump_index(aa_jm1, aa_jp1)];
                }
            } else if self.model >= 2 {
                for i in 0..src_len {
                    let e = src.tokens[i];
                    let n = self.source_count[e as usize].get(&f_tok).copied().unwrap_or(0) as Count;
                    let alpha = if let Some(pr) = &self.source_prior {
                        pr[e as usize].get(&f_tok).copied().unwrap_or(0.0) as Count + LEX_ALPHA
                    } else { LEX_ALPHA };
                    let j1 = get_jump_index(aa_jm1, i as isize);
                    let j2 = get_jump_index(i as isize, aa_jp1);
                    ps_sum += self.inv_source_count_sum[e as usize]
                        * (alpha + n)
                        * self.jump_counts[j1]
                        * self.jump_counts[j2];
                    ps[i] = ps_sum;
                }

                if let Some(scores) = sentence_scores.as_deref_mut() {
                    let mut max_p = 0.0;
                    for i in 0..src_len {
                        let p = if i == 0 { ps[0] } else { ps[i] - ps[i - 1] };
                        if p > max_p { max_p = p; }
                    }
                    ps_sum += self.null_prior
                        * self.inv_source_count_sum[0]
                        * (NULL_ALPHA + null_n)
                        * self.jump_counts[JUMP_SUM]
                        * self.jump_counts[get_jump_index(aa_jm1, aa_jp1)];
                    let denom = self.jump_counts[JUMP_SUM] * self.jump_counts[JUMP_SUM];
                    scores[s] += logapprox((max_p / denom) as f32) as Count;
                } else {
                    ps_sum += self.null_prior
                        * self.inv_source_count_sum[0]
                        * (NULL_ALPHA + null_n)
                        * self.jump_counts[JUMP_SUM]
                        * self.jump_counts[get_jump_index(aa_jm1, aa_jp1)];
                }
            } else {
                for i in 0..src_len {
                    let e = src.tokens[i];
                    let n = self.source_count[e as usize].get(&f_tok).copied().unwrap_or(0) as Count;
                    let alpha = if let Some(pr) = &self.source_prior {
                        pr[e as usize].get(&f_tok).copied().unwrap_or(0.0) as Count + LEX_ALPHA
                    } else { LEX_ALPHA };
                    ps_sum += self.inv_source_count_sum[e as usize] * (alpha + n);
                    ps[i] = ps_sum;
                }

                if let Some(scores) = sentence_scores.as_deref_mut() {
                    let mut max_p = 0.0;
                    for i in 0..src_len {
                        let p = if i == 0 { ps[0] } else { ps[i] - ps[i - 1] };
                        if p > max_p { max_p = p; }
                    }
                    scores[s] += logapprox(max_p as f32) as Count;
                }

                ps_sum += self.null_prior * self.inv_source_count_sum[0] * (NULL_ALPHA + null_n);
            }

            ps[src_len] = ps_sum;

            let new_i: Link = if sentence_scores.is_some() {
                if old_i == NULL_LINK { src_len as Link } else { old_i }
            } else {
                let u = (rng.next_f32() as Count) * ps_sum.max(1e-30);
                let mut k = 0usize;
                while k < src_len && ps[k] < u { k += 1; }
                k as Link
            };

            let new_e = if new_i as usize == src_len {
                links[j] = NULL_LINK;
                0
            } else {
                links[j] = new_i;
                if self.model >= 3 {
                    if let Some(ref mut fv) = fert {
                        fv[new_i as usize] += 1;
                    }
                }
                src.tokens[new_i as usize]
            };

            if s < n_use {
                self.inv_source_count_sum[new_e as usize] =
                    1.0 / (1.0 / self.inv_source_count_sum[new_e as usize] + 1.0);
                *self.source_count[new_e as usize].entry(f_tok).or_insert(0) += 1;

                if self.model >= 2 {
                    let skip_jump = get_jump_index(aa_jm1, aa_jp1);
                    if new_e == 0 {
                        self.jump_counts[JUMP_SUM] += 1.0;
                        self.jump_counts[skip_jump] += 1.0;
                    } else {
                        let j1 = get_jump_index(aa_jm1, new_i as isize);
                        let j2 = get_jump_index(new_i as isize, aa_jp1);
                        self.jump_counts[JUMP_SUM] += 2.0;
                        self.jump_counts[j1] += 1.0;
                        self.jump_counts[j2] += 1.0;
                    }
                }
            }

            if self.model >= 2 && new_e != 0 {
                aa_jm1 = new_i as isize;
            }
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

fn final_argmax_iteration<'a>(samplers: &mut [TA<'a>], rng: &mut Pcg32) {
    if samplers.is_empty() {
        return;
    }
    let n_samplers = samplers.len();
    let model = samplers[0].model;
    let n_sentences = samplers[0].target.n_sentences;
    let n_use = if samplers[0].n_clean == 0 {
        n_sentences
    } else {
        samplers[0].n_clean
    };

    // Resample fertility exactly once, and only for sampler 0
    if model >= 3 {
        samplers[0].resample_fertility(rng);
    }

    // We must use sampler[0]'s jump and fertility arrays for all distributions
    // during this pass (matching how C captures the pointers before the loop).
    // We'll mutate sampler[0].jump_counts as we sweep each sampler.
    // Fertility ratios (samplers[0].fert_counts) are read-only here.
    let source = samplers[0].source;
    let target = samplers[0].target;

    // Extract shared data from samplers[0] to avoid borrowing conflicts
    let mut shared_jump_counts = samplers[0].jump_counts.clone();
    let shared_fert_counts = samplers[0].fert_counts.clone();

    for s in 0..n_sentences {
        // Skip sentences that aren’t aligned in sampler 0
        if samplers[0].sentence_links[s].is_none() {
            continue;
        }
        let src = source.sentences[s].as_ref().unwrap();
        let tgt = target.sentences[s].as_ref().unwrap();
        let src_len = src.len();
        let tgt_len = tgt.len();

        // Accumulator across samplers: normalized probabilities per position
        // laid out as j-major blocks of (src_len + 1)
        let mut acc_ps = vec![0.0 as Count; tgt_len * (src_len + 1)];
        let mut acc_base: usize;

        // Sweep all samplers in the same order as C (from last down to 0)
        for idx_rev in (0..n_samplers).rev() {
            // Reset acc_base for each sampler so it indexes from the start of acc_ps
            acc_base = 0;

            // If this sampler has no links for this sentence, skip it exactly as C does.
            if samplers[idx_rev].sentence_links[s].is_none() {
                continue;
            }

            // Borrow this sampler
            let ta = &mut samplers[idx_rev];

            // Precompute aa_jp1 from current links; C does this once per sampler per sentence.
            let links = ta.sentence_links[s].as_mut().unwrap();
            let mut aa_jp1_table = vec![src_len as isize; tgt_len];
            if model >= 2 {
                let mut cur = src_len as isize;
                for j in (0..tgt_len).rev() {
                    aa_jp1_table[j] = cur;
                    if links[j] != NULL_LINK {
                        cur = links[j] as isize;
                    }
                }
            }

            // Fertility of tokens in this sentence (for model >= 3)
            let mut fert = if model >= 3 {
                let mut f = vec![0usize; src_len];
                for &li in links.iter() {
                    if li != NULL_LINK {
                        f[li as usize] += 1;
                    }
                }
                Some(f)
            } else {
                None
            };

            // Left neighbor of nearest non-NULL aligned token to the left
            let mut aa_jm1: isize = -1;

            // Sweep target positions
            for j in 0..tgt_len {
                let f_tok = tgt.tokens[j];
                let old_i = links[j];
                let old_e = if old_i == NULL_LINK {
                    0
                } else {
                    src.tokens[old_i as usize]
                };
                let aa_jp1 = aa_jp1_table[j];

                if model >= 3 {
                    if let (Some(ref mut fv), true) = (&mut fert, old_i != NULL_LINK) {
                        fv[old_i as usize] = fv[old_i as usize].saturating_sub(1);
                    }
                }

                // If this sentence is part of the “clean” set, decrement lexical/inv-sum for this sampler,
                // and decrement sampler[0]'s jump counts by this sampler’s old jump(s)
                if s < n_use {
                    // inv_source_count_sum decrement for old_e
                    let inv = ta.inv_source_count_sum[old_e as usize];
                    ta.inv_source_count_sum[old_e as usize] = 1.0 / (1.0 / inv - 1.0);

                    // Decrement lexical count n(old_e, f)
                    if let Some(c) = ta.source_count[old_e as usize].get_mut(&f_tok) {
                        if *c > 1 {
                            *c -= 1;
                        } else {
                            ta.source_count[old_e as usize].remove(&f_tok);
                        }
                    }

                    // Decrement shared jump_counts (from sampler 0) for the old alignment
                    if model >= 2 {
                        let skip_jump = get_jump_index(aa_jm1, aa_jp1);
                        if old_i == NULL_LINK {
                            shared_jump_counts[JUMP_SUM] -= 1.0;
                            shared_jump_counts[skip_jump] -= 1.0;
                        } else {
                            let j1 = get_jump_index(aa_jm1, old_i as isize);
                            let j2 = get_jump_index(old_i as isize, aa_jp1);
                            shared_jump_counts[JUMP_SUM] -= 2.0;
                            shared_jump_counts[j1] -= 1.0;
                            shared_jump_counts[j2] -= 1.0;
                        }
                    }
                }

                // Build this sampler’s (unnormalized) non-NULL probabilities
                // using this sampler’s lexical counts and inv sums,
                // but sampler[0]'s jump and fert ratios (exactly like the C pointer-capture).
                let mut numer = vec![0.0 as Count; src_len + 1]; // last slot will be NULL
                let mut sum: Count = 0.0;

                if model >= 3 {
                    // HMM + fertility
                    // Using aa_jm1 and aa_jp1, but jump weights from sampler[0]
                    let mut j1 = get_jump_index(aa_jm1, 0);
                    let mut j2 = get_jump_index(0, aa_jp1);
                    for i in 0..src_len {
                        let e = src.tokens[i];
                        let n = ta.source_count[e as usize]
                            .get(&f_tok)
                            .copied()
                            .unwrap_or(0) as Count;
                        let alpha = if let Some(pr) = &ta.source_prior {
                            pr[e as usize]
                                .get(&f_tok)
                                .copied()
                                .unwrap_or(0.0) as Count
                                + LEX_ALPHA
                        } else {
                            LEX_ALPHA
                        };
                        let fi = if let Some(ref fv) = fert { fv[i] + 1 } else { 1 };
                        let fert_w = shared_fert_counts[get_fert_index(e, fi)];

                        let term = ta.inv_source_count_sum[e as usize]
                            * (alpha + n)
                            * shared_jump_counts[j1]
                            * shared_jump_counts[j2]
                            * fert_w;
                        numer[i] = term;
                        sum += term;

                        // Bounded increments (same micro-optimization as C)
                        j1 = (j1 + 1).min(JUMP_ARRAY_LEN - 1);
                        j2 = j2.saturating_sub(1);
                    }
                    // NULL term
                    let null_n =
                        ta.source_count[0].get(&f_tok).copied().unwrap_or(0) as Count;
                    let skip = get_jump_index(aa_jm1, aa_jp1);
                    let null_term = ta.null_prior
                        * ta.inv_source_count_sum[0]
                        * (NULL_ALPHA + null_n)
                        * shared_jump_counts[JUMP_SUM]
                        * shared_jump_counts[skip];
                    numer[src_len] = null_term;
                    sum += null_term;
                } else if model >= 2 {
                    // HMM (no fertility)
                    let mut j1 = get_jump_index(aa_jm1, 0);
                    let mut j2 = get_jump_index(0, aa_jp1);
                    for i in 0..src_len {
                        let e = src.tokens[i];
                        let n = ta.source_count[e as usize]
                            .get(&f_tok)
                            .copied()
                            .unwrap_or(0) as Count;
                        let alpha = if let Some(pr) = &ta.source_prior {
                            pr[e as usize]
                                .get(&f_tok)
                                .copied()
                                .unwrap_or(0.0) as Count
                                + LEX_ALPHA
                        } else {
                            LEX_ALPHA
                        };
                        let term = ta.inv_source_count_sum[e as usize]
                            * (alpha + n)
                            * shared_jump_counts[j1]
                            * shared_jump_counts[j2];
                        numer[i] = term;
                        sum += term;

                        j1 = (j1 + 1).min(JUMP_ARRAY_LEN - 1);
                        j2 = j2.saturating_sub(1);
                    }
                    let null_n =
                        ta.source_count[0].get(&f_tok).copied().unwrap_or(0) as Count;
                    let skip = get_jump_index(aa_jm1, aa_jp1);
                    let null_term = ta.null_prior
                        * ta.inv_source_count_sum[0]
                        * (NULL_ALPHA + null_n)
                        * shared_jump_counts[JUMP_SUM]
                        * shared_jump_counts[skip];
                    numer[src_len] = null_term;
                    sum += null_term;
                } else {
                    // Model 1
                    for i in 0..src_len {
                        let e = src.tokens[i];
                        let n = ta.source_count[e as usize]
                            .get(&f_tok)
                            .copied()
                            .unwrap_or(0) as Count;
                        let alpha = if let Some(pr) = &ta.source_prior {
                            pr[e as usize]
                                .get(&f_tok)
                                .copied()
                                .unwrap_or(0.0) as Count
                                + LEX_ALPHA
                        } else {
                            LEX_ALPHA
                        };
                        let term = ta.inv_source_count_sum[e as usize] * (alpha + n);
                        numer[i] = term;
                        sum += term;
                    }
                    let null_n =
                        ta.source_count[0].get(&f_tok).copied().unwrap_or(0) as Count;
                    let null_term =
                        ta.null_prior * ta.inv_source_count_sum[0] * (NULL_ALPHA + null_n);
                    numer[src_len] = null_term;
                    sum += null_term;
                }

                // Accumulate normalized distribution into acc_ps for this j
                if sum > 0.0 {
                    let invsum = 1.0 / sum;
                    for k in 0..(src_len + 1) {
                        acc_ps[acc_base + k] += numer[k] * invsum;
                    }
                }

                // Argmax over accumulated probabilities for ALL samplers
                // (matching C behavior: all samplers use argmax in the final pass)
                let mut max_k = 0usize;
                let mut max_v = acc_ps[acc_base + 0];

                for k in 1..=src_len {
                    let v = acc_ps[acc_base + k];
                    if v > max_v {
                        max_v = v;
                        max_k = k;
                    }
                }
                let best_k = max_k;

                // Apply chosen link to the current sampler
                let new_i = if best_k == src_len {
                    links[j] = NULL_LINK;
                    NULL_LINK
                } else {
                    links[j] = best_k as Link;
                    links[j]
                };
                let new_e = if new_i == NULL_LINK {
                    0
                } else {
                    src.tokens[new_i as usize]
                };

                if model >= 3 {
                    if let Some(ref mut fv) = fert {
                        if new_i != NULL_LINK {
                            fv[new_i as usize] += 1;
                        }
                    }
                }

                if s < n_use {
                    // inv sum increment and lexical increment for new_e
                    ta.inv_source_count_sum[new_e as usize] =
                        1.0 / (1.0 / ta.inv_source_count_sum[new_e as usize] + 1.0);
                    *ta.source_count[new_e as usize].entry(f_tok).or_insert(0) += 1;

                    // Increment shared jump counts (sampler 0) for the new link
                    if model >= 2 {
                        let skip = get_jump_index(aa_jm1, aa_jp1);
                        if new_e == 0 {
                            shared_jump_counts[JUMP_SUM] += 1.0;
                            shared_jump_counts[skip] += 1.0;
                        } else {
                            let j1 = get_jump_index(aa_jm1, new_i as isize);
                            let j2 = get_jump_index(new_i as isize, aa_jp1);
                            shared_jump_counts[JUMP_SUM] += 2.0;
                            shared_jump_counts[j1] += 1.0;
                            shared_jump_counts[j2] += 1.0;
                        }
                    }
                }

                // Update left neighbor if new link is non-NULL
                if model >= 2 && new_e != 0 {
                    aa_jm1 = new_i as isize;
                }

                // Advance acc_base for this position
                acc_base += src_len + 1;
            }
        }

        // After sweeping all samplers, the links for sampler 0 are already set
        // to the consensus argmax choices (because idx_rev==0 was processed last).
    }
    
    // Copy the shared jump counts back to samplers[0]
    samplers[0].jump_counts = shared_jump_counts;
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
        let ta = TA::create(if reverse { target } else { source }, if reverse { source } else { target }, opts.null_prior);
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
                samplers[i].sample(r, None);
            }
        }
    }

    // Final argmax iteration: consensus argmax across samplers, writing into samplers[0]
    final_argmax_iteration(&mut samplers, &mut rngs[0]);

    // Outputs
    let links_moses = if want_links {
        let pairs = crate::text::links_to_pairs(&samplers[0].sentence_links, reverse);
        crate::text::write_moses(&pairs)
    } else { String::new() };

    let stats = if want_stats {
        Some(crate::text::write_stats(&samplers[0].jump_counts))
    } else { None };

    let (forward_scores, reverse_scores) = if want_scores {
        // Score forward - we need mutable access to samplers[0]
        let mut scores_fwd = vec![0.0 as Count; samplers[0].source.n_sentences];
        samplers[0].model = opts.score_model;
        samplers[0].sample(&mut rngs[0], Some(&mut scores_fwd));
        let fwd = crate::text::write_scores(&scores_fwd);

        // Score reverse if needed: run again with reversed
        (Some(fwd), None)
    } else { (None, None) };
    
    let links_vec = if want_links { Some(samplers[0].sentence_links.clone()) } else { None };

    Ok(AlignResult { links_moses, stats, forward_scores, reverse_scores, links_vec })
}