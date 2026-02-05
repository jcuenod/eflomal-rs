use clap::Parser;
use std::fs;
use eflomal_core::{align, alignment::calculate_iterations, parse_plaintext, parse_text, AlignOptions};

#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Args {
    #[arg(short='s', long="source", default_value="-")]
    source: String,
    #[arg(short='t', long="target", default_value="-")]
    target: String,
    #[arg(long, default_value_t=false)]
    raw: bool,
    #[arg(short='p', long="priors")]
    priors: Option<String>,
    #[arg(short='f', long="forward")]
    links_fwd: Option<String>,
    #[arg(short='r', long="reverse")]
    links_rev: Option<String>,
    #[arg(short='S', long="stats")]
    stats_out: Option<String>,
    #[arg(short='F', long="forward-scores")]
    scores_fwd: Option<String>,
    #[arg(short='R', long="reverse-scores")]
    scores_rev: Option<String>,
    #[arg(short='1', default_value_t=0)]
    it1: usize,
    #[arg(short='2', default_value_t=0)]
    it2: usize,
    #[arg(short='3', default_value_t=0)]
    it3: usize,
    #[arg(short='n', default_value_t=1)]
    n_samplers: usize,
    #[arg(short='N', default_value_t=0.2)]
    null_prior: f32,
    #[arg(short='m', default_value_t=3)]
    model: u8,
    #[arg(short='M')]
    score_model: Option<u8>,
    #[arg(long, default_value_t=1)]
    seed: u64,
}

fn read_all(path: &str) -> std::io::Result<String> {
    if path == "-" {
        use std::io::Read;
        let mut s = String::new();
        std::io::stdin().read_to_string(&mut s)?;
        Ok(s)
    } else {
        fs::read_to_string(path)
    }
}

fn write_all(path: Option<String>, data: &str) -> std::io::Result<()> {
    if let Some(p) = path {
        if p == "-" { print!("{data}"); }
        else { fs::write(p, data)?; }
    }
    Ok(())
}

fn parse_auto(s: &str, raw: bool, label: &str) -> Result<eflomal_core::Text, String> {
    if raw {
        return parse_plaintext(s).map_err(|e| format!("{label}: {e}"));
    }
    match parse_text(s) {
        Ok(t) => Ok(t),
        Err(_) => {
            // Fallback to raw for convenience
            parse_plaintext(s).map_err(|e| format!("{label}: {e} (tried numeric first)"))
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let src_s = read_all(&args.source)?;
    let tgt_s = read_all(&args.target)?;
    let source = parse_auto(&src_s, args.raw, "source")?;
    let target = parse_auto(&tgt_s, args.raw, "target")?;

    let priors = if let Some(p) = args.priors.as_ref() {
        Some(read_all(p)?)
    } else { None };

    let (approx_it1, approx_it2, approx_it3) = calculate_iterations(source.n_sentences, args.model);

    let it1 = if args.it1 > 0 { args.it1 } else { approx_it1 };
    let it2 = if args.it2 > 0 { args.it2 } else { approx_it2 };
    let it3 = if args.it3 > 0 { args.it3 } else { approx_it3 };
    // println!("iters: {}, {}, {}", it1, it2, it3);

    let opts = AlignOptions {
        model: args.model,
        score_model: args.score_model.unwrap_or(args.model),
        n_iters: [it1, it2, it3],
        n_samplers: args.n_samplers,
        null_prior: args.null_prior as f32,
        n_clean: None,
        priors,
        reverse: false,
        seed: args.seed,
    };

    // Forward
    if args.links_fwd.is_some() || args.scores_fwd.is_some() {
        let res = align(false, &source, &target, &opts, true, args.stats_out.is_some(), args.scores_fwd.is_some())?;
        write_all(args.links_fwd.clone().or(Some("-".to_string())), &res.links_moses)?;
        if let Some(stats) = res.stats { write_all(args.stats_out.clone(), &stats)?; }
        if let Some(scores) = res.forward_scores { write_all(args.scores_fwd.clone(), &scores)?; }
    }

    // Reverse
    if args.links_rev.is_some() || args.scores_rev.is_some() {
        let mut opts_rev = opts.clone();
        opts_rev.reverse = true;
        let res = align(true, &source, &target, &opts_rev, true, false, args.scores_rev.is_some())?;
        write_all(args.links_rev.clone(), &res.links_moses)?;
        if let Some(scores) = res.forward_scores { write_all(args.scores_rev.clone(), &scores)?; }
    }

    // If neither forward nor reverse specified, do both and symmetrize
    if args.links_fwd.is_none() && args.links_rev.is_none() {
        let res_fwd = align(false, &source, &target, &opts, true, args.stats_out.is_some(), args.scores_fwd.is_some())?;
        let mut opts_rev = opts.clone();
        opts_rev.reverse = true;
        let res_rev = align(true, &source, &target, &opts_rev, true, false, args.scores_rev.is_some())?;

        if let (Some(fwd_links), Some(rev_links)) = (res_fwd.links_vec.as_ref(), res_rev.links_vec.as_ref()) {
            // symmetrize (forward is oriented (src->tgt) and reverse was created with reverse=true
            let merged = eflomal_core::symmetrize::grow_diag_final_and(fwd_links, rev_links, &source, &target)?;
            let moses = eflomal_core::text::write_moses(&merged);
            write_all(Some("-".to_string()), &moses)?;
        }
    }

    Ok(())
}