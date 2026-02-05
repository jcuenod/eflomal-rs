use eflomal_core::{align, alignment::calculate_iterations, parse_plaintext, parse_text, AlignOptions};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct AlignConfig {
    model: u8,
    score_model: Option<u8>,
    it1: Option<usize>,
    it2: Option<usize>,
    it3: Option<usize>,
    n_samplers: usize,
    null_prior: f32,
    seed: u64,
    priors: Option<String>,
    
    // Control what outputs to generate
    forward: bool,
    reverse: bool,
    symmetrize: bool,
    want_stats: bool,
    want_scores: bool,
}

#[wasm_bindgen]
impl AlignConfig {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            model: 3,  // default model
            score_model: None,
            it1: None,
            it2: None,
            it3: None,
            n_samplers: 1,
            null_prior: 0.2,
            seed: 1,
            priors: None,
            forward: true,
            reverse: true,
            symmetrize: true,
            want_stats: false,
            want_scores: false,
        }
    }

    // Setter methods - only one parameter each
    #[wasm_bindgen(setter)]
    pub fn set_model(&mut self, model: u8) {
        self.model = model;
    }

    #[wasm_bindgen(setter)]
    pub fn set_score_model(&mut self, score_model: u8) {
        self.score_model = Some(score_model);
    }

    // Individual setters for iterations
    #[wasm_bindgen(setter)]
    pub fn set_it1(&mut self, it1: usize) {
        self.it1 = Some(it1);
    }

    #[wasm_bindgen(setter)]
    pub fn set_it2(&mut self, it2: usize) {
        self.it2 = Some(it2);
    }

    #[wasm_bindgen(setter)]
    pub fn set_it3(&mut self, it3: usize) {
        self.it3 = Some(it3);
    }

    // Regular method (not a setter) for setting all iterations at once
    pub fn set_iterations(&mut self, it1: usize, it2: usize, it3: usize) {
        self.it1 = Some(it1);
        self.it2 = Some(it2);
        self.it3 = Some(it3);
    }

    #[wasm_bindgen(setter)]
    pub fn set_n_samplers(&mut self, n: usize) {
        self.n_samplers = n;
    }

    #[wasm_bindgen(setter)]
    pub fn set_null_prior(&mut self, p: f32) {
        self.null_prior = p;
    }

    #[wasm_bindgen(setter)]
    pub fn set_seed(&mut self, seed: u64) {
        self.seed = seed;
    }

    #[wasm_bindgen(setter)]
    pub fn set_priors(&mut self, priors: String) {
        self.priors = Some(priors);
    }

    // Regular methods (not setters) for multi-parameter configurations
    pub fn set_direction(&mut self, forward: bool, reverse: bool, symmetrize: bool) {
        self.forward = forward;
        self.reverse = reverse;
        self.symmetrize = symmetrize;
    }

    pub fn set_outputs(&mut self, want_stats: bool, want_scores: bool) {
        self.want_stats = want_stats;
        self.want_scores = want_scores;
    }

    // Convenience setters for direction flags
    #[wasm_bindgen(setter)]
    pub fn set_forward(&mut self, forward: bool) {
        self.forward = forward;
    }

    #[wasm_bindgen(setter)]
    pub fn set_reverse(&mut self, reverse: bool) {
        self.reverse = reverse;
    }

    #[wasm_bindgen(setter)]
    pub fn set_symmetrize(&mut self, symmetrize: bool) {
        self.symmetrize = symmetrize;
    }

    #[wasm_bindgen(setter)]
    pub fn set_want_stats(&mut self, want: bool) {
        self.want_stats = want;
    }

    #[wasm_bindgen(setter)]
    pub fn set_want_scores(&mut self, want: bool) {
        self.want_scores = want;
    }
}

#[wasm_bindgen]
pub struct AlignOutput {
    links_forward: Option<String>,
    links_reverse: Option<String>,
    links_symmetrized: Option<String>,
    stats: Option<String>,
    scores_forward: Option<String>,
    scores_reverse: Option<String>,
}

#[wasm_bindgen]
impl AlignOutput {
    #[wasm_bindgen(getter)]
    pub fn links_forward(&self) -> Option<String> {
        self.links_forward.clone()
    }
    
    #[wasm_bindgen(getter)]
    pub fn links_reverse(&self) -> Option<String> {
        self.links_reverse.clone()
    }
    
    #[wasm_bindgen(getter)]
    pub fn links_symmetrized(&self) -> Option<String> {
        self.links_symmetrized.clone()
    }
    
    #[wasm_bindgen(getter)]
    pub fn stats(&self) -> Option<String> {
        self.stats.clone()
    }
    
    #[wasm_bindgen(getter)]
    pub fn scores_forward(&self) -> Option<String> {
        self.scores_forward.clone()
    }
    
    #[wasm_bindgen(getter)]
    pub fn scores_reverse(&self) -> Option<String> {
        self.scores_reverse.clone()
    }
    
    // Convenience getter that returns the primary alignment result
    #[wasm_bindgen(getter)]
    pub fn links(&self) -> Option<String> {
        self.links_symmetrized.clone()
            .or_else(|| self.links_forward.clone())
            .or_else(|| self.links_reverse.clone())
    }
}

fn parse_auto(s: &str, label: &str) -> Result<eflomal_core::Text, String> {
    match parse_text(s) {
        Ok(t) => Ok(t),
        Err(_) => {
            // Fallback to plaintext for convenience
            parse_plaintext(s).map_err(|e| format!("{label}: {e} (tried numeric first)"))
        }
    }
}

#[wasm_bindgen]
pub fn align_text(
    source_text: &str,
    target_text: &str,
    config: &AlignConfig,
) -> Result<AlignOutput, JsValue> {
    // Parse inputs with auto-detection
    let source = parse_auto(source_text, "source").map_err(|e| JsValue::from_str(&e))?;
    let target = parse_auto(target_text, "target").map_err(|e| JsValue::from_str(&e))?;
    
    // Calculate iterations automatically if not specified
    let (approx_it1, approx_it2, approx_it3) = calculate_iterations(source.n_sentences, config.model);
    let it1 = config.it1.unwrap_or(if approx_it1 > 0 { approx_it1 } else { 1 });
    let it2 = config.it2.unwrap_or(if approx_it2 > 0 { approx_it2 } else { 1 });
    let it3 = config.it3.unwrap_or(if approx_it3 > 0 { approx_it3 } else { 1 });
    
    let opts = AlignOptions {
        model: config.model,
        score_model: config.score_model.unwrap_or(config.model),
        n_iters: [it1, it2, it3],
        n_samplers: config.n_samplers,
        null_prior: config.null_prior,
        n_clean: None,
        priors: config.priors.clone(),
        reverse: false,
        seed: config.seed,
    };
    
    let mut output = AlignOutput {
        links_forward: None,
        links_reverse: None,
        links_symmetrized: None,
        stats: None,
        scores_forward: None,
        scores_reverse: None,
    };
    
    // Handle different alignment modes
    if config.symmetrize || (!config.forward && !config.reverse) {
        // Do both directions and symmetrize
        let res_fwd = align(false, &source, &target, &opts, true, config.want_stats, config.want_scores)
            .map_err(|e| JsValue::from_str(&e))?;
        
        let mut opts_rev = opts.clone();
        opts_rev.reverse = true;
        let res_rev = align(true, &source, &target, &opts_rev, true, false, config.want_scores)
            .map_err(|e| JsValue::from_str(&e))?;
        
        if let (Some(fwd_links), Some(rev_links)) = (res_fwd.links_vec.as_ref(), res_rev.links_vec.as_ref()) {
            let merged = eflomal_core::symmetrize::grow_diag_final_and(fwd_links, rev_links, &source, &target)
                .map_err(|e| JsValue::from_str(&e))?;
            let moses = eflomal_core::text::write_moses_pairs(&merged);
            output.links_symmetrized = Some(moses);
        }
        
        if config.forward {
            output.links_forward = Some(res_fwd.links_moses.clone());
        }
        if config.reverse {
            output.links_reverse = Some(res_rev.links_moses.clone());
        }
        
        output.stats = res_fwd.stats;
        output.scores_forward = res_fwd.forward_scores;
        output.scores_reverse = res_rev.forward_scores;
        
    } else {
        // Single direction alignment
        if config.forward {
            let res = align(false, &source, &target, &opts, true, config.want_stats, config.want_scores)
                .map_err(|e| JsValue::from_str(&e))?;
            output.links_forward = Some(res.links_moses);
            output.stats = res.stats;
            output.scores_forward = res.forward_scores;
        }
        
        if config.reverse {
            let mut opts_rev = opts.clone();
            opts_rev.reverse = true;
            let res = align(true, &source, &target, &opts_rev, true, config.want_stats && !config.forward, config.want_scores)
                .map_err(|e| JsValue::from_str(&e))?;
            output.links_reverse = Some(res.links_moses);
            if !config.forward {
                output.stats = res.stats;
            }
            output.scores_reverse = res.forward_scores;
        }
    }
    
    Ok(output)
}

// Convenience function with defaults for simple use cases
#[wasm_bindgen]
pub fn align_simple(
    source_text: &str,
    target_text: &str,
) -> Result<String, JsValue> {
    let config = AlignConfig::new();
    let output = align_text(source_text, target_text, &config)?;
    output.links()
        .ok_or_else(|| JsValue::from_str("No alignment produced"))
}