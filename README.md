# eflomal-wasm

A Rust reimplementation of [eflomal](https://github.com/robertostling/eflomal) (Efficient Low-Memory Aligner), a tool for statistical word alignment of parallel texts. This project provides both a **command-line tool** and a **WebAssembly module** for use in web projects.

Word alignment identifies which words in a source sentence correspond to which words in a target sentence across a parallel corpus. For example, given the English sentence "the cat sat" and its French translation "le chat s'assit", the aligner produces links like `0-0 1-1 2-2`.

## Features

- EM-based alignment using Models 1, 2, and 3 with Gibbs sampling
- Bidirectional alignment with "grow-diag-final-and" symmetrization
- Multiple independent samplers with consensus decoding
- Support for alignment priors (transfer learning)
- Plaintext and numeric input formats (auto-detected)
- Output in Moses alignment format (`source_index-target_index`)

## Project Structure

```
eflomal-core/   # Core alignment algorithm library
eflomal-cli/    # Command-line interface
eflomal-wasm/   # WebAssembly bindings via wasm-bindgen
```

## CLI Usage

### Build

```bash
cargo build --release -p eflomal-cli
```

### Input Format

Prepare two text files with one sentence per line (source and target must have the same number of lines):

**source.txt**
```
the cat sat
the world is big
```

**target.txt**
```
le chat s'assit
le monde est grand
```

### Run

By default, the aligner runs both forward and reverse alignment and outputs a symmetrized result to stdout:

```bash
eflomal-cli -s source.txt -t target.txt
```

Output (one line per sentence pair, Moses format):
```
0-0 1-1 2-2
0-0 1-1 2-2 3-3
```

### Options

```
-s, --source <FILE>           Source text file (default: stdin)
-t, --target <FILE>           Target text file (default: stdin)
-f, --forward <FILE>          Write forward alignment to file
-r, --reverse <FILE>          Write reverse alignment to file
-m <MODEL>                    Alignment model: 1, 2, or 3 (default: 3)
-n <N>                        Number of independent samplers (default: 1)
-N <P>                        NULL alignment prior probability (default: 0.2)
-1 <N>                        Model 1 iterations (0 = auto)
-2 <N>                        Model 2 iterations (0 = auto)
-3 <N>                        Model 3 iterations (0 = auto)
-p, --priors <FILE>           Alignment priors file
-S, --stats <FILE>            Write jump statistics to file
-F, --forward-scores <FILE>   Write forward alignment scores to file
-R, --reverse-scores <FILE>   Write reverse alignment scores to file
--seed <N>                    Random seed (default: 1)
--raw                         Force plaintext parsing (skip numeric format detection)
```

If neither `-f` nor `-r` is specified, the tool runs both directions and prints the symmetrized alignment to stdout. If only one direction is specified, it outputs that direction only.

## WASM / Web Usage

### Build

```bash
wasm-pack build eflomal-wasm --release --target web
```

This produces a package in `eflomal-wasm/pkg/` containing the `.wasm` binary, a JS wrapper, and TypeScript definitions.

### Import

```javascript
import init, { align_simple, align_text, AlignConfig } from './eflomal-wasm/pkg/eflomal_wasm.js';

await init(); // initialize the WASM module
```

### Simple Usage

`align_simple` runs bidirectional alignment with symmetrization using default settings:

```javascript
const source = "the cat sat\nthe world is big";
const target = "le chat s'assit\nle monde est grand";

const alignment = align_simple(source, target);
console.log(alignment);
// "0-0 1-1 2-2\n0-0 1-1 2-2 3-3\n"
```

### Advanced Usage

Use `AlignConfig` and `align_text` for full control over the alignment:

```javascript
const config = new AlignConfig();

// Model settings
config.model = 3;             // alignment model (1, 2, or 3)
config.n_samplers = 2;        // number of independent samplers
config.null_prior = 0.2;      // NULL alignment probability
config.seed = 42n;            // random seed (BigInt for u64)

// Iteration counts (omit to auto-calculate)
config.it1 = 5;               // model 1 iterations
config.it2 = 5;               // model 2 iterations
config.it3 = 10;              // model 3 iterations
// or: config.set_iterations(5, 5, 10)

// Direction control
config.forward = true;
config.reverse = true;
config.symmetrize = true;
// or: config.set_direction(true, true, true)

// Optional outputs
config.want_stats = true;
config.want_scores = true;
// or: config.set_outputs(true, true)

const output = align_text(source, target, config);

output.links;                // auto-selected best result (symmetrized > forward > reverse)
output.links_symmetrized;    // grow-diag-final-and merged alignment
output.links_forward;        // forward direction only
output.links_reverse;        // reverse direction only
output.stats;                // jump probability statistics
output.scores_forward;       // per-sentence forward scores
output.scores_reverse;       // per-sentence reverse scores
```

### Using in Node.js

```bash
node --experimental-wasm-modules test.mjs
```

## Output Format

Alignments are output in Moses format: each line contains space-separated `source_index-target_index` pairs (0-indexed) for one sentence pair.

```
0-0 1-1 2-2
0-0 1-2 2-1 3-3
```

## License

See [eflomal](https://github.com/robertostling/eflomal) for the original implementation by Robert Östling.
