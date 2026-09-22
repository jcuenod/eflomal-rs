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
eflomal-wasm/   # WebAssembly bindings via wasm-bindgen (published to npm as `eflomal`)
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

## WASM / JavaScript Usage

The WebAssembly bindings are published to npm as
[`eflomal`](https://www.npmjs.com/package/eflomal). See
[eflomal-wasm/README.md](eflomal-wasm/README.md) for the full JavaScript API —
that file is what ships as the package README.

```bash
npm install eflomal
```

```js
import { align } from "eflomal";

align(["the cat sat"], ["le chat s'assit"]);
// [ [[0, 0], [1, 1], [2, 2]] ]
```

Each link is `[sourceIndex, targetIndex]`, and each entry of the result
corresponds to the sentence pair at the same index. `alignDetailed` additionally
returns both directions, per-sentence scores and jump statistics; `formatMoses`
renders an alignment as Moses text.

### Build

```bash
wasm-pack build eflomal-wasm --release --target web
```

The `web` target is the only one built: browsers and bundlers load it directly,
and Node reaches it through [eflomal-wasm/node.js](eflomal-wasm/node.js), which
reads the `.wasm` off disk so callers need not await an initializer.

### Test

`eflomal-wasm/type-check-runtime.ts` checks the generated TypeScript against how
the API is actually used, and asserts the runtime shapes match. It runs against
the `pkg` build above:

```bash
cd eflomal-wasm
npx tsc --noEmit --strict --skipLibCheck --moduleResolution bundler --module esnext --target es2022 type-check-runtime.ts
npx tsx type-check-runtime.ts
```

### Publishing

Pushing a `v*` tag runs [.github/workflows/publish-npm.yml](.github/workflows/publish-npm.yml),
which verifies the crate, builds the web target, merges
[eflomal-wasm/package-overrides.json](eflomal-wasm/package-overrides.json) into
the wasm-pack manifest, stamps the version from the tag, and publishes to npm.

## Output Format

Alignments are output in Moses format: each line contains space-separated `source_index-target_index` pairs (0-indexed) for one sentence pair.

```
0-0 1-1 2-2
0-0 1-2 2-1 3-3
```

## License

MIT. See [eflomal](https://github.com/robertostling/eflomal) for the original implementation by Robert Östling.
