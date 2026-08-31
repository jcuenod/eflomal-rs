# eflomal

Statistical word alignment for parallel text, as a WebAssembly module. This is a
Rust port of [eflomal](https://github.com/robertostling/eflomal) (Efficient
Low-Memory Aligner) by Robert Östling, with no native dependencies — it runs in
Node and in the browser.

Word alignment tells you which words in a sentence correspond to which words in
its translation. Given `the cat sat` and `le chat s'assit`, the aligner links
word 0 to word 0, word 1 to word 1, and word 2 to word 2.

```bash
npm install eflomal
```

## Usage

```js
import { align } from "eflomal";

const source = ["the cat sat", "the world is big"];
const target = ["le chat s'assit", "le monde est grand"];

align(source, target);
// [ [[0,0], [1,1], [2,2]],
//   [[0,0], [1,1], [2,2], [3,3]] ]
```

Each entry of the result corresponds to the sentence pair at the same index.
Each link is `[sourceIndex, targetIndex]`, both 0-based word positions.

Alignment quality comes from the corpus as a whole, not from any one sentence —
the model learns which words translate to which by seeing them recur. Two
sentences will not align well; a few thousand will.

### In the browser

The web build fetches the `.wasm` file, so it needs to be initialized first:

```js
import init, { align } from "eflomal";

await init();
align(source, target);
```

`init()` also exists in Node, where it resolves immediately — calling it is
harmless, so isomorphic code can call it unconditionally.

### Tokenization

Sentences are strings split on whitespace, or arrays of tokens you split
yourself:

```js
align(
  [["the", "cat", "sat"]],
  [["le", "chat", "s'", "assit"]],
);
```

Link indices always refer to positions in the tokenization eflomal saw, so
passing your own tokens is the way to guarantee that indices line up with your
own arrays.

### Options

```js
align(source, target, {
  direction: "symmetric", // "forward" | "reverse" | "symmetric"  (default "symmetric")
  model: 3,               // 1 lexical, 2 + HMM jumps, 3 + fertility  (default 3)
  iterations: { model1: 5, model2: 5, model3: 10 }, // default: derived from corpus size
  samplers: 1,            // independent samplers combined by consensus  (default 1)
  nullPrior: 0.2,         // probability a word aligns to nothing  (default 0.2)
  seed: 1,                // alignment is deterministic given a seed  (default 1)
  priors: undefined,      // alignment priors, in eflomal's priors text format
});
```

`"forward"` aligns source to target, `"reverse"` aligns target to source, and
`"symmetric"` runs both and merges them with the Moses grow-diag-final-and
heuristic. Every direction returns links as `[sourceIndex, targetIndex]`.

### Everything at once

`alignDetailed` returns both directions and the symmetrization from a single
run, plus per-sentence scores and jump statistics on request:

```js
import { alignDetailed } from "eflomal";

const result = alignDetailed(source, target, { scores: true });

result.forward;        // Link[][]
result.reverse;        // Link[][]
result.symmetric;      // Link[][]
result.forwardScores;  // number[] — one per sentence pair
result.reverseScores;  // number[]
result.jumpCounts;     // number[] — only with { statistics: true }
```

A score is the negated mean log-probability per target token, so a larger score
means a less probable alignment. Sorting by score is a good way to find the
sentence pairs that aligned worst — usually the ones that are misaligned or
mistranslated in the corpus to begin with.

### Moses format

`formatMoses` renders an alignment as the usual interchange format — one line
per sentence pair, each link written as `sourceIndex-targetIndex`:

```js
import { align, formatMoses } from "eflomal";

formatMoses(align(source, target));
// "0-0 1-1 2-2\n0-0 1-1 2-2 3-3\n"
```

## Types

The package ships TypeScript declarations for everything above: `Corpus`,
`Sentence`, `Link`, `Alignment`, `Model`, `Direction`, `AlignOptions`,
`DetailedAlignOptions` and `DetailedAlignment`. Unknown or misspelled option
keys are rejected at runtime as well as by the type checker.

## Performance

Alignment is CPU-bound and synchronous — a large corpus will block the thread it
runs on. In a browser, run it in a Web Worker. Cost scales with corpus size,
`model`, `samplers`, and the iteration counts; the defaults derive iterations
from the number of sentences so that larger corpora do fewer passes.

## License

MIT. See [eflomal](https://github.com/robertostling/eflomal) for the original
implementation by Robert Östling.
