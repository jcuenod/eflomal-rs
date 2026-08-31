//! WebAssembly bindings for eflomal.
//!
//! The JavaScript surface is three functions — [`align`], [`align_detailed`] and
//! [`format_moses`] — plus the hand-written TypeScript declarations in
//! [`TYPESCRIPT_DEFINITIONS`], which are what consumers actually program against.

use std::collections::HashMap;

use eflomal_core::{
    align as core_align,
    alignment::calculate_iterations,
    symmetrize::grow_diag_final_and,
    text::{write_moses, Sentence, Text},
    types::{Token, MAX_SENT_LEN},
    AlignOptions, AlignResult,
};
use serde::Deserialize;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;

/// Hand-written TypeScript for the values that cross the boundary as plain JS
/// objects. wasm-bindgen copies this verbatim into `eflomal_wasm.d.ts`.
#[wasm_bindgen(typescript_custom_section)]
const TYPESCRIPT_DEFINITIONS: &'static str = r#"
/**
 * One sentence of a corpus, either as a string (split on whitespace) or as
 * tokens you have already split yourself.
 *
 * Link indices always refer to positions in the tokenization eflomal saw, so
 * pre-tokenized input is the only way to guarantee indices line up with your
 * own token array.
 */
export type Sentence = string | string[];

/** A parallel corpus: one entry per sentence, aligned by index with its counterpart. */
export type Corpus = Sentence[];

/** A single word link: `[sourceIndex, targetIndex]`, both 0-based. */
export type Link = [source: number, target: number];

/**
 * Links for a whole corpus: one entry per sentence pair, in input order.
 * Sentence pairs that could not be aligned (either side empty) yield `[]`.
 */
export type Alignment = Link[][];

/** IBM model to run. Higher models subsume the lower ones and cost more time. */
export type Model =
  /** Lexical translation probabilities only. */
  | 1
  /** Adds HMM-style jump probabilities. */
  | 2
  /** Adds fertility. */
  | 3;

/** Which alignment to return. */
export type Direction =
  /** Align source to target. */
  | "forward"
  /** Align target to source. */
  | "reverse"
  /** Run both directions and merge them with grow-diag-final-and. */
  | "symmetric";

/** Gibbs sampling iterations per model. Omit a field to derive it from the corpus size. */
export interface Iterations {
  model1?: number;
  model2?: number;
  model3?: number;
}

/** Settings shared by `align` and `alignDetailed`. */
export interface BaseAlignOptions {
  /** Model to train up to. Default `3`. */
  model?: Model;
  /** Iterations per model. Default: derived from the number of sentences. */
  iterations?: Iterations;
  /** Independent samplers whose final alignments are combined by consensus. Default `1`. */
  samplers?: number;
  /** Prior probability that a word aligns to NULL. Default `0.2`. */
  nullPrior?: number;
  /** Seed for the random number generator. Alignment is deterministic given a seed. Default `1`. */
  seed?: number;
  /** Alignment priors, in eflomal's priors text format. */
  priors?: string;
}

/** Options for `align`. */
export interface AlignOptions extends BaseAlignOptions {
  /** Which alignment to return. Default `"symmetric"`. */
  direction?: Direction;
}

/** Options for `alignDetailed`. */
export interface DetailedAlignOptions extends BaseAlignOptions {
  /** Also return per-sentence alignment scores. Default `false`. */
  scores?: boolean;
  /** Also return jump-length statistics. Default `false`. */
  statistics?: boolean;
  /** Model used to compute scores, when `scores` is set. Default: the value of `model`. */
  scoreModel?: Model;
}

/** Everything `alignDetailed` can produce. */
export interface DetailedAlignment {
  /** Source-to-target links. */
  forward: Alignment;
  /** Target-to-source links, still expressed as `[sourceIndex, targetIndex]`. */
  reverse: Alignment;
  /** `forward` and `reverse` merged with grow-diag-final-and. */
  symmetric: Alignment;
  /**
   * Per-sentence score of the forward alignment: the negated mean
   * log-probability per target token, so a larger value means a less probable
   * alignment. Present only when `scores` was set.
   */
  forwardScores?: number[];
  /** The same measure for the reverse alignment. Present only when `scores` was set. */
  reverseScores?: number[];
  /**
   * Jump-length counts from the forward direction, as written by the CLI's
   * `--stats`: entry `i` counts jumps of length `i - 1024`, and the last entry
   * is the sampler's running total rather than a jump count.
   * Present only when `statistics` was set.
   */
  jumpCounts?: number[];
}
"#;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(typescript_type = "Corpus")]
    pub type CorpusJs;

    #[wasm_bindgen(typescript_type = "AlignOptions")]
    pub type AlignOptionsJs;

    #[wasm_bindgen(typescript_type = "DetailedAlignOptions")]
    pub type DetailedAlignOptionsJs;

    #[wasm_bindgen(typescript_type = "Alignment")]
    pub type AlignmentJs;

    #[wasm_bindgen(typescript_type = "DetailedAlignment")]
    pub type DetailedAlignmentJs;
}

// ── Option decoding ──────────────────────────────────────────────────────────

#[derive(Deserialize)]
#[serde(untagged)]
enum SentenceInput {
    Raw(String),
    Tokens(Vec<String>),
}

#[derive(Deserialize, Default)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Iterations {
    model1: Option<usize>,
    model2: Option<usize>,
    model3: Option<usize>,
}

#[derive(Deserialize, Clone, Copy)]
#[serde(rename_all = "lowercase")]
enum Direction {
    Forward,
    Reverse,
    Symmetric,
}

/// The settings `AlignOptions` and `DetailedAlignOptions` have in common.
#[derive(Deserialize, Default)]
#[serde(rename_all = "camelCase")]
struct BaseOptionsInput {
    model: Option<u8>,
    iterations: Option<Iterations>,
    samplers: Option<usize>,
    null_prior: Option<f32>,
    seed: Option<u64>,
    priors: Option<String>,
}

#[derive(Deserialize, Default)]
#[serde(rename_all = "camelCase")]
struct AlignOptionsInput {
    #[serde(flatten)]
    base: BaseOptionsInput,
    direction: Option<Direction>,
}

#[derive(Deserialize, Default)]
#[serde(rename_all = "camelCase")]
struct DetailedAlignOptionsInput {
    #[serde(flatten)]
    base: BaseOptionsInput,
    scores: Option<bool>,
    statistics: Option<bool>,
    score_model: Option<u8>,
}

const BASE_OPTION_KEYS: &[&str] = &[
    "model",
    "iterations",
    "samplers",
    "nullPrior",
    "seed",
    "priors",
];
const ALIGN_OPTION_KEYS: &[&str] = &["direction"];
const DETAILED_OPTION_KEYS: &[&str] = &["scores", "statistics", "scoreModel"];
const ITERATION_KEYS: &[&str] = &["model1", "model2", "model3"];

/// Reject keys the options type does not declare. serde's `deny_unknown_fields`
/// cannot do this for us: serde-wasm-bindgen looks fields up by name instead of
/// iterating the object, so a misspelled option would otherwise be ignored.
fn reject_unknown_keys(value: &JsValue, label: &str, allowed: &[&[&str]]) -> Result<(), JsError> {
    if !value.is_object() {
        return Err(JsError::new(&format!("{label} must be an object")));
    }
    for key in js_sys::Object::keys(value.unchecked_ref()).iter() {
        let key = key.as_string().unwrap_or_default();
        if !allowed.iter().any(|group| group.contains(&key.as_str())) {
            let mut known: Vec<&str> = allowed
                .iter()
                .flat_map(|group| group.iter().copied())
                .collect();
            known.sort_unstable();
            return Err(JsError::new(&format!(
                "unknown key \"{key}\" in {label} (expected one of: {})",
                known.join(", ")
            )));
        }
    }
    Ok(())
}

/// Deserialize an optional options bag, treating `undefined`/`null`/absent as "all defaults".
fn decode_options<T: for<'de> Deserialize<'de> + Default>(
    value: Option<JsValue>,
    allowed: &[&[&str]],
) -> Result<T, JsError> {
    let value = match value {
        None => return Ok(T::default()),
        Some(v) if v.is_undefined() || v.is_null() => return Ok(T::default()),
        Some(v) => v,
    };

    reject_unknown_keys(&value, "options", allowed)?;
    let iterations = js_sys::Reflect::get(&value, &JsValue::from_str("iterations"))
        .map_err(|_| JsError::new("could not read options"))?;
    if !iterations.is_undefined() && !iterations.is_null() {
        reject_unknown_keys(&iterations, "iterations", &[ITERATION_KEYS])?;
    }

    serde_wasm_bindgen::from_value(value)
        .map_err(|e| JsError::new(&format!("invalid options: {e}")))
}

fn check_model(model: u8, field: &str) -> Result<u8, JsError> {
    match model {
        1..=3 => Ok(model),
        other => Err(JsError::new(&format!(
            "{field} must be 1, 2 or 3 (got {other})"
        ))),
    }
}

/// Build the core options, filling unspecified iteration counts from the corpus size.
fn build_core_options(
    n_sentences: usize,
    options: BaseOptionsInput,
    score_model: Option<u8>,
) -> Result<AlignOptions, JsError> {
    let BaseOptionsInput {
        model,
        iterations,
        samplers,
        null_prior,
        seed,
        priors,
    } = options;
    let defaults = AlignOptions::default();

    let model = check_model(model.unwrap_or(defaults.model), "model")?;
    let score_model = match score_model {
        Some(m) => check_model(m, "scoreModel")?,
        None => model,
    };

    let n_samplers = samplers.unwrap_or(defaults.n_samplers);
    if n_samplers == 0 {
        return Err(JsError::new("samplers must be at least 1"));
    }

    let null_prior = null_prior.unwrap_or(defaults.null_prior);
    if !(0.0..1.0).contains(&null_prior) {
        return Err(JsError::new(&format!(
            "nullPrior must be in [0, 1) (got {null_prior})"
        )));
    }

    let (auto1, auto2, auto3) = calculate_iterations(n_sentences, model);
    let iterations = iterations.unwrap_or_default();

    Ok(AlignOptions {
        model,
        score_model,
        n_iters: [
            iterations.model1.unwrap_or(auto1),
            iterations.model2.unwrap_or(auto2),
            iterations.model3.unwrap_or(auto3),
        ],
        n_samplers,
        null_prior,
        n_clean: None,
        priors,
        reverse: false,
        seed: seed.unwrap_or(defaults.seed),
    })
}

// ── Corpus decoding ──────────────────────────────────────────────────────────

fn decode_corpus(value: &JsValue, label: &str) -> Result<Vec<SentenceInput>, JsError> {
    serde_wasm_bindgen::from_value(value.clone()).map_err(|e| {
        JsError::new(&format!(
            "{label} must be an array of strings or token arrays: {e}"
        ))
    })
}

/// Index every distinct token, reserving id 0 for NULL (as the core expects).
fn build_text(sentences: &[SentenceInput], label: &str) -> Result<Text, JsError> {
    let mut vocabulary: HashMap<&str, Token> = HashMap::new();
    let mut next_id: Token = 1;
    let mut out: Vec<Option<Sentence>> = Vec::with_capacity(sentences.len());

    for (index, sentence) in sentences.iter().enumerate() {
        let words: Vec<&str> = match sentence {
            SentenceInput::Raw(text) => text.split_whitespace().collect(),
            SentenceInput::Tokens(tokens) => tokens.iter().map(String::as_str).collect(),
        };

        if words.is_empty() {
            out.push(None);
            continue;
        }
        if words.len() > MAX_SENT_LEN {
            return Err(JsError::new(&format!(
                "{label} sentence {index} has {} tokens, but the maximum is {MAX_SENT_LEN}",
                words.len()
            )));
        }

        let tokens = words
            .into_iter()
            .map(|word| {
                *vocabulary.entry(word).or_insert_with(|| {
                    let id = next_id;
                    next_id += 1;
                    id
                })
            })
            .collect();
        out.push(Some(Sentence { tokens }));
    }

    Ok(Text {
        n_sentences: out.len(),
        vocabulary_size: next_id,
        sentences: out,
    })
}

fn decode_pair(source: &JsValue, target: &JsValue) -> Result<(Text, Text), JsError> {
    let source_sentences = decode_corpus(source, "source")?;
    let target_sentences = decode_corpus(target, "target")?;

    if source_sentences.len() != target_sentences.len() {
        return Err(JsError::new(&format!(
            "source and target must have the same number of sentences (got {} and {})",
            source_sentences.len(),
            target_sentences.len()
        )));
    }

    Ok((
        build_text(&source_sentences, "source")?,
        build_text(&target_sentences, "target")?,
    ))
}

// ── Alignment ────────────────────────────────────────────────────────────────

type Pairs = Vec<Option<Vec<(u16, u16)>>>;

/// Links as the JS side sees them: one array per sentence pair, `[]` where unaligned.
fn to_link_lists(pairs: Pairs) -> Vec<Vec<(u16, u16)>> {
    pairs.into_iter().map(Option::unwrap_or_default).collect()
}

fn run(
    reverse: bool,
    source: &Text,
    target: &Text,
    options: &AlignOptions,
    want_stats: bool,
    want_scores: bool,
) -> Result<AlignResult, JsError> {
    let mut options = options.clone();
    options.reverse = reverse;
    core_align(
        reverse,
        source,
        target,
        &options,
        true,
        want_stats,
        want_scores,
    )
    .map_err(|e| JsError::new(&e))
}

/// Pull the per-target-position links out of a result and orient them as (source, target).
fn pairs_of(result: &AlignResult, reverse: bool) -> Result<Pairs, JsError> {
    let links = result
        .links_vec
        .as_ref()
        .ok_or_else(|| JsError::new("aligner returned no links"))?;
    Ok(eflomal_core::links_to_pairs(links, reverse))
}

fn symmetrized(
    forward: &AlignResult,
    reverse: &AlignResult,
    source: &Text,
    target: &Text,
) -> Result<Pairs, JsError> {
    let forward_links = forward
        .links_vec
        .as_ref()
        .ok_or_else(|| JsError::new("aligner returned no forward links"))?;
    let reverse_links = reverse
        .links_vec
        .as_ref()
        .ok_or_else(|| JsError::new("aligner returned no reverse links"))?;
    grow_diag_final_and(forward_links, reverse_links, source, target).map_err(|e| JsError::new(&e))
}

fn serialize<T: serde::Serialize, U: JsCast>(value: &T) -> Result<U, JsError> {
    let js = serde_wasm_bindgen::to_value(value)
        .map_err(|e| JsError::new(&format!("could not serialize result: {e}")))?;
    Ok(js.unchecked_into())
}

/// Align a parallel corpus and return the links for one direction.
///
/// `source` and `target` must have the same number of sentences.
#[wasm_bindgen]
pub fn align(
    source: &CorpusJs,
    target: &CorpusJs,
    options: Option<AlignOptionsJs>,
) -> Result<AlignmentJs, JsError> {
    let options: AlignOptionsInput = decode_options(
        options.map(JsValue::from),
        &[BASE_OPTION_KEYS, ALIGN_OPTION_KEYS],
    )?;
    let (source, target) = decode_pair(source.as_ref(), target.as_ref())?;

    let core_options = build_core_options(source.n_sentences, options.base, None)?;

    let pairs = match options.direction.unwrap_or(Direction::Symmetric) {
        Direction::Forward => {
            let result = run(false, &source, &target, &core_options, false, false)?;
            pairs_of(&result, false)?
        }
        Direction::Reverse => {
            let result = run(true, &source, &target, &core_options, false, false)?;
            pairs_of(&result, true)?
        }
        Direction::Symmetric => {
            let forward = run(false, &source, &target, &core_options, false, false)?;
            let reverse = run(true, &source, &target, &core_options, false, false)?;
            symmetrized(&forward, &reverse, &source, &target)?
        }
    };

    serialize(&to_link_lists(pairs))
}

/// Align a parallel corpus and return every direction at once, plus optional
/// scores and jump statistics.
#[wasm_bindgen(js_name = alignDetailed)]
pub fn align_detailed(
    source: &CorpusJs,
    target: &CorpusJs,
    options: Option<DetailedAlignOptionsJs>,
) -> Result<DetailedAlignmentJs, JsError> {
    let options: DetailedAlignOptionsInput = decode_options(
        options.map(JsValue::from),
        &[BASE_OPTION_KEYS, DETAILED_OPTION_KEYS],
    )?;
    let (source, target) = decode_pair(source.as_ref(), target.as_ref())?;

    let want_scores = options.scores.unwrap_or(false);
    let want_stats = options.statistics.unwrap_or(false);

    let core_options = build_core_options(source.n_sentences, options.base, options.score_model)?;

    let forward = run(
        false,
        &source,
        &target,
        &core_options,
        want_stats,
        want_scores,
    )?;
    let reverse = run(true, &source, &target, &core_options, false, want_scores)?;

    #[derive(serde::Serialize)]
    #[serde(rename_all = "camelCase")]
    struct DetailedAlignment {
        forward: Vec<Vec<(u16, u16)>>,
        reverse: Vec<Vec<(u16, u16)>>,
        symmetric: Vec<Vec<(u16, u16)>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        forward_scores: Option<Vec<f64>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        reverse_scores: Option<Vec<f64>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        jump_counts: Option<Vec<i32>>,
    }

    serialize(&DetailedAlignment {
        forward: to_link_lists(pairs_of(&forward, false)?),
        reverse: to_link_lists(pairs_of(&reverse, true)?),
        symmetric: to_link_lists(symmetrized(&forward, &reverse, &source, &target)?),
        forward_scores: forward.forward_scores_vec.clone(),
        reverse_scores: reverse.forward_scores_vec.clone(),
        jump_counts: forward.jump_counts.clone(),
    })
}

/// Render an alignment as Moses text: one line per sentence pair, each link
/// written as `sourceIndex-targetIndex`.
#[wasm_bindgen(js_name = formatMoses)]
pub fn format_moses(alignment: &AlignmentJs) -> Result<String, JsError> {
    let links: Vec<Vec<(u16, u16)>> = serde_wasm_bindgen::from_value(
        AsRef::<JsValue>::as_ref(alignment).clone(),
    )
    .map_err(|e| {
        JsError::new(&format!(
            "alignment must be an array of [source, target] pairs: {e}"
        ))
    })?;
    let pairs: Pairs = links.into_iter().map(Some).collect();
    Ok(write_moses(&pairs))
}
