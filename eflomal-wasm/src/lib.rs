use eflomal_core::{align, parse_plaintext, parse_text, AlignOptions};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct AlignOutput {
    links_moses: String,
    stats: Option<String>,
    scores: Option<String>,
}

#[wasm_bindgen]
impl AlignOutput {
    #[wasm_bindgen(getter)]
    pub fn links_moses(&self) -> String {
        self.links_moses.clone()
    }
    #[wasm_bindgen(getter)]
    pub fn stats(&self) -> Option<String> {
        self.stats.clone()
    }
    #[wasm_bindgen(getter)]
    pub fn scores(&self) -> Option<String> {
        self.scores.clone()
    }
}

#[wasm_bindgen]
pub fn align_moses(
    source_text: &str,
    target_text: &str,
    priors: Option<String>,
    model: u8,
    score_model: u8,
    it1: usize,
    it2: usize,
    it3: usize,
    n_samplers: usize,
    null_prior: f32,
    reverse: bool,
    seed: u64,
    want_stats: bool,
    want_scores: bool,
) -> Result<AlignOutput, JsValue> {
    let source = parse_text(source_text).map_err(|e| JsValue::from_str(&e))?;
    let target = parse_text(target_text).map_err(|e| JsValue::from_str(&e))?;
    let opts = AlignOptions {
        model,
        score_model,
        n_iters: [it1, it2, it3],
        n_samplers,
        null_prior,
        n_clean: None,
        priors,
        reverse,
        seed,
    };
    let res = align(
        reverse,
        &source,
        &target,
        &opts,
        true,
        want_stats,
        want_scores,
    )
    .map_err(|e| JsValue::from_str(&e))?;
    Ok(AlignOutput {
        links_moses: res.links_moses,
        stats: res.stats,
        scores: res.forward_scores, // reverse scores omitted for simplicity
    })
}

#[wasm_bindgen]
pub fn align_plaintext(
    source_text: &str,
    target_text: &str,
    model: u8,
    score_model: u8,
    it1: usize,
    it2: usize,
    it3: usize,
    n_samplers: usize,
    null_prior: f32,
    reverse: bool,
    seed: u64,
    want_stats: bool,
    want_scores: bool,
) -> Result<AlignOutput, JsValue> {
    let source = parse_plaintext(source_text).map_err(|e| JsValue::from_str(&e))?;
    let target = parse_plaintext(target_text).map_err(|e| JsValue::from_str(&e))?;
    let opts = AlignOptions {
        model,
        score_model,
        n_iters: [it1, it2, it3],
        n_samplers,
        null_prior,
        n_clean: None,
        priors: None,
        reverse,
        seed,
    };
    let res = align(
        reverse,
        &source,
        &target,
        &opts,
        true,
        want_stats,
        want_scores,
    )
    .map_err(|e| JsValue::from_str(&e))?;
    Ok(AlignOutput {
        links_moses: res.links_moses,
        stats: res.stats,
        scores: res.forward_scores,
    })
}
