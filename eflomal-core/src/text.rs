extern crate alloc;

use crate::types::*;

#[derive(Clone, Debug)]
pub struct Sentence {
    pub tokens: Vec<Token>, // token 0 reserved for NULL
}
impl Sentence {
    #[inline] pub fn len(&self) -> usize { self.tokens.len() }
}

#[derive(Clone, Debug)]
pub struct Text {
    pub n_sentences: usize,
    pub vocabulary_size: Token, // includes +1 for NULL=0
    pub sentences: Vec<Option<Sentence>>,
}

pub fn parse_plaintext(s: &str) -> Result<Text, String> {
    use hashbrown::HashMap;
    let mut vocab: HashMap<String, u32> = HashMap::new();
    let mut next_id: u32 = 1; // 0 is reserved for NULL
    let mut sentences: Vec<Option<Sentence>> = Vec::new();

    for line in s.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            // Treat empty line as “no sentence” (None), like length 0 input
            sentences.push(None);
            continue;
        }
        let words: Vec<&str> = trimmed.split_whitespace().collect();
        if words.len() > MAX_SENT_LEN {
            return Err(format!("sentence too long: {} > {}", words.len(), MAX_SENT_LEN));
        }
        let mut tokens: Vec<Token> = Vec::with_capacity(words.len());
        for w in words {
            let id = match vocab.get(w) {
                Some(&id) => id,
                None => {
                    let id = next_id;
                    next_id += 1;
                    vocab.insert(w.to_string(), id);
                    id
                }
            };
            tokens.push(id);
        }
        sentences.push(Some(Sentence { tokens }));
    }

    let n_sentences = sentences.len();
    // vocabulary_size must include +1 for NULL=0, next_id is already (1 + num_types)
    let vocabulary_size = next_id as Token;

    Ok(Text { n_sentences, vocabulary_size, sentences })
}

pub fn parse_text(s: &str) -> Result<Text, String> {
    let mut lines = s.lines();
    let header = lines.next().ok_or("missing header")?;
    let mut it = header.split_whitespace();
    let n_sentences: usize = it.next().ok_or("bad header")?.parse().map_err(|_|"bad n_sentences")?;
    let mut vocab: Token = it.next().ok_or("bad header")?.parse().map_err(|_|"bad vocab")?;
    // Reserve 0 for NULL
    vocab = vocab + 1;

    let mut sentences = Vec::with_capacity(n_sentences);
    for _ in 0..n_sentences {
        let line = lines.next().ok_or("missing sentence line")?;
        let mut it = line.split_whitespace();
        let len: usize = it.next().ok_or("missing length")?.parse().map_err(|_|"bad length")?;
        if len == 0 {
            sentences.push(None);
            continue;
        }
        if len > MAX_SENT_LEN { return Err("sentence too long".into()); }
        let mut tokens = Vec::with_capacity(len);
        for _ in 0..len {
            let t: u32 = it.next().ok_or("missing token")?.parse().map_err(|_|"bad token")?;
            let token = t + 1; // shift by 1
            if token >= vocab { return Err("token >= vocab".into()); }
            tokens.push(token);
        }
        sentences.push(Some(Sentence { tokens }));
    }

    Ok(Text { n_sentences, vocabulary_size: vocab, sentences })
}

// Moses alignment writer (per sentence line)
pub fn write_moses(
    links: &[Option<Vec<Link>>],
    target: &Text,
    reverse: bool
) -> String {
    let mut out = String::new();
    for (sent, links_opt) in links.iter().enumerate() {
        let ls = match links_opt { None => { out.push('\n'); continue; }, Some(v)=>v };
        let tgt = match &target.sentences[sent] { None => { out.push('\n'); continue; }, Some(x)=>x };
        let mut first = true;
        for (j, &li) in ls.iter().enumerate() {
            if li != NULL_LINK {
                if reverse {
                    if first { out.push_str(&format!("{}-{}", j, li)); first=false; }
                    else { out.push_str(&format!(" {}-{}", j, li)); }
                } else {
                    if first { out.push_str(&format!("{}-{}", li, j)); first=false; }
                    else { out.push_str(&format!(" {}-{}", li, j)); }
                }
            }
        }
        out.push('\n');
    }
    out
}

// Stats output (only jump stats like original)
pub fn write_stats(jump_counts: &[Count; JUMP_ARRAY_LEN]) -> String {
    use alloc::format;
    let mut s = String::new();
    s.push_str(&format!("{}\n", JUMP_ARRAY_LEN));
    for i in 0..JUMP_ARRAY_LEN {
        let v = (jump_counts[i] - JUMP_ALPHA).round() as i32;
        s.push_str(&format!("{}\n", v));
    }
    s
}

// Scores: print per sentence (negative log-score)
pub fn write_scores(scores: &[Count]) -> String {
    let mut s = String::new();
    for &sc in scores {
        s.push_str(&format!("{}\n", -(sc as f64)));
    }
    s
}