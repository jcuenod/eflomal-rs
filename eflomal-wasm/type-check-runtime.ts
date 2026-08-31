/**
 * Type and runtime validation for the eflomal npm package.
 *
 * Two guarantees in one file:
 *
 *   1. Compile-time (tsc --noEmit): the explicit annotations below fail to
 *      compile whenever a declared type drifts from how the API is actually
 *      used, and the `Record<Union, true>` tables fail whenever a union gains
 *      a member that isn't handled here.
 *
 *   2. Runtime (tsx): actually runs the WASM and asserts that the values match
 *      what the types promise — shapes, index ranges, which optional fields
 *      appear, and which inputs are rejected.
 *
 * Run after `wasm-pack build eflomal-wasm --target nodejs --out-dir pkg-node`:
 *   npx tsc --noEmit --strict --skipLibCheck --moduleResolution bundler --module esnext type-check-runtime.ts
 *   npx tsx type-check-runtime.ts
 *
 * Exit code 0 = all checks passed, 1 = at least one failure.
 */

import { align, alignDetailed, formatMoses } from "./pkg-node/eflomal_wasm.js";
import type {
  Alignment,
  AlignOptions,
  BaseAlignOptions,
  Corpus,
  DetailedAlignment,
  DetailedAlignOptions,
  Direction,
  Iterations,
  Link,
  Model,
  Sentence,
} from "./pkg-node/eflomal_wasm.js";

// ── Assertion framework ───────────────────────────────────────────────────────

let failures = 0;

function check(label: string, condition: boolean): void {
  if (!condition) {
    console.error(`FAIL  ${label}`);
    failures++;
  }
}

function checkThrows(label: string, run: () => unknown): void {
  try {
    run();
    console.error(`FAIL  ${label} (expected a throw)`);
    failures++;
  } catch {
    /* expected */
  }
}

// ── Exhaustive closed-union coverage ─────────────────────────────────────────
// Record<T, true> causes a compile error if T gains a member not listed here.

const _directions: Record<Direction, true> = {
  forward: true,
  reverse: true,
  symmetric: true,
};

const _models: Record<Model, true> = { 1: true, 2: true, 3: true };

const _iterations: Record<keyof Iterations, true> = {
  model1: true,
  model2: true,
  model3: true,
};

const _baseOptions: Record<keyof BaseAlignOptions, true> = {
  model: true,
  iterations: true,
  samplers: true,
  nullPrior: true,
  seed: true,
  priors: true,
};

const _detailed: Record<keyof DetailedAlignment, true> = {
  forward: true,
  reverse: true,
  symmetric: true,
  forwardScores: true,
  reverseScores: true,
  jumpCounts: true,
};

void _directions, _models, _iterations, _baseOptions, _detailed;

// ── Corpus ────────────────────────────────────────────────────────────────────

const source: Corpus = [
  "the cat sat",
  "the world is big",
  "the universe is vast",
  "the cat is big",
  "hello world",
  "hello universe",
  "goodbye world",
  "the big world",
];

const target: Corpus = [
  "le chat s'assit",
  "le monde est grand",
  "l'univers est vaste",
  "le chat est grand",
  "bonjour monde",
  "bonjour univers",
  "au revoir monde",
  "le grand monde",
];

/** Sentence accepts both members of its union, including within one corpus. */
const mixed: Corpus = ["the cat sat", ["the", "world", "is", "big"]];
const _sentence: Sentence = mixed[0];
void _sentence;

const sourceTokens: Corpus = source.map((s) => (s as string).split(" "));
const targetTokens: Corpus = target.map((s) => (s as string).split(" "));

/** Assert every link is an in-range [source, target] index pair. */
function checkAlignment(label: string, alignment: Alignment, src: Corpus, tgt: Corpus): void {
  check(`${label}: is an array`, Array.isArray(alignment));
  check(`${label}: one entry per sentence pair`, alignment.length === src.length);

  for (const [index, links] of alignment.entries()) {
    check(`${label}[${index}]: is an array`, Array.isArray(links));
    const sourceLength = (src[index] as string).split(" ").length;
    const targetLength = (tgt[index] as string).split(" ").length;

    for (const link of links) {
      const pair: Link = link;
      check(`${label}[${index}]: link is a 2-tuple`, Array.isArray(pair) && pair.length === 2);
      const [i, j] = pair;
      check(`${label}[${index}]: source index is an integer`, Number.isInteger(i));
      check(`${label}[${index}]: target index is an integer`, Number.isInteger(j));
      check(`${label}[${index}]: source index ${i} within 0..${sourceLength - 1}`, i >= 0 && i < sourceLength);
      check(`${label}[${index}]: target index ${j} within 0..${targetLength - 1}`, j >= 0 && j < targetLength);
    }
  }
}

// ── align() ───────────────────────────────────────────────────────────────────

const defaults: Alignment = align(source, target);
checkAlignment("align (defaults)", defaults, source, target);
check("align (defaults): finds some links", defaults.some((links) => links.length > 0));

for (const direction of ["forward", "reverse", "symmetric"] as const) {
  const options: AlignOptions = { direction, seed: 7 };
  const result: Alignment = align(source, target, options);
  checkAlignment(`align (${direction})`, result, source, target);
}

/** A forward-only alignment links each target word at most once. */
const forwardOnly: Alignment = align(source, target, { direction: "forward", seed: 7 });
check(
  "align (forward): each target index used at most once per sentence",
  forwardOnly.every((links) => new Set(links.map(([, j]) => j)).size === links.length),
);

/** Symmetrization is a union of both directions, so it cannot lose links. */
const symmetric: Alignment = align(source, target, { direction: "symmetric", seed: 7 });
const reverseOnly: Alignment = align(source, target, { direction: "reverse", seed: 7 });
check(
  "align: symmetric is at least as large as either direction",
  symmetric.every((links, i) => links.length >= Math.max(forwardOnly[i].length, reverseOnly[i].length)),
);

/** Options are honoured rather than ignored. */
const seeded = JSON.stringify(align(source, target, { seed: 42 }));
check("align: the same seed gives the same alignment", seeded === JSON.stringify(align(source, target, { seed: 42 })));

const everyOption: AlignOptions = {
  direction: "symmetric",
  model: 2,
  iterations: { model1: 3, model2: 3 },
  samplers: 2,
  nullPrior: 0.1,
  seed: 99,
};
checkAlignment("align (every option)", align(source, target, everyOption), source, target);

/** Pre-tokenized input indexes the caller's own tokens. */
const tokenized: Alignment = align(sourceTokens, targetTokens, { seed: 7 });
checkAlignment("align (pre-tokenized)", tokenized, source, target);
check(
  "align: pre-tokenized input matches whitespace-split input",
  JSON.stringify(tokenized) === JSON.stringify(align(source, target, { seed: 7 })),
);

/** A corpus with no sentences used to hang: 5000/sqrt(0) saturated the iteration count. */
const empty: Alignment = align([], []);
check("align: empty corpus returns an empty alignment", empty.length === 0);

/** Empty sentences are kept as empty link lists so indices stay aligned. */
const withGap: Alignment = align([...source, ""], [...target, "rien"], { seed: 7 });
check("align: keeps a slot for an empty sentence", withGap.length === source.length + 1);
check("align: empty sentence has no links", withGap[withGap.length - 1].length === 0);

// ── alignDetailed() ───────────────────────────────────────────────────────────

const bare: DetailedAlignment = alignDetailed(source, target, { seed: 7 });
checkAlignment("alignDetailed.forward", bare.forward, source, target);
checkAlignment("alignDetailed.reverse", bare.reverse, source, target);
checkAlignment("alignDetailed.symmetric", bare.symmetric, source, target);
check("alignDetailed: no scores unless asked", bare.forwardScores === undefined && bare.reverseScores === undefined);
check("alignDetailed: no statistics unless asked", bare.jumpCounts === undefined);
check(
  "alignDetailed.symmetric matches align(direction: symmetric)",
  JSON.stringify(bare.symmetric) === JSON.stringify(symmetric),
);
check(
  "alignDetailed.forward matches align(direction: forward)",
  JSON.stringify(bare.forward) === JSON.stringify(forwardOnly),
);

const full: DetailedAlignment = alignDetailed(source, target, {
  scores: true,
  statistics: true,
  scoreModel: 3,
  seed: 7,
} satisfies DetailedAlignOptions);

const forwardScores: number[] | undefined = full.forwardScores;
const reverseScores: number[] | undefined = full.reverseScores;
const jumpCounts: number[] | undefined = full.jumpCounts;

check("alignDetailed: forwardScores present", Array.isArray(forwardScores));
check("alignDetailed: one forward score per sentence", forwardScores?.length === source.length);
check("alignDetailed: forward scores are finite", forwardScores?.every(Number.isFinite) === true);
check("alignDetailed: reverseScores present", Array.isArray(reverseScores));
check("alignDetailed: one reverse score per sentence", reverseScores?.length === source.length);
check("alignDetailed: jumpCounts present", Array.isArray(jumpCounts));
check("alignDetailed: jumpCounts is one entry per jump length", jumpCounts?.length === 2048);
check("alignDetailed: jumpCounts are integers", jumpCounts?.every(Number.isInteger) === true);

// ── formatMoses() ─────────────────────────────────────────────────────────────

const moses: string = formatMoses(symmetric);
const lines = moses.split("\n");

check("formatMoses: one line per sentence pair", lines.length === symmetric.length + 1);
check("formatMoses: trailing newline", lines[lines.length - 1] === "");
check(
  "formatMoses: renders every link as i-j",
  lines.slice(0, -1).every((line, i) => line === symmetric[i].map(([a, b]) => `${a}-${b}`).join(" ")),
);
check("formatMoses: accepts an empty alignment", formatMoses([]) === "");

// ── Rejected input ────────────────────────────────────────────────────────────

checkThrows("rejects mismatched corpus lengths", () => align(source, target.slice(1)));
checkThrows("rejects a model outside 1..3", () => align(source, target, { model: 4 as Model }));
checkThrows("rejects zero samplers", () => align(source, target, { samplers: 0 }));
checkThrows("rejects a nullPrior of 1", () => align(source, target, { nullPrior: 1 }));
checkThrows("rejects an unknown direction", () => align(source, target, { direction: "both" as Direction }));
checkThrows("rejects a misspelled option", () =>
  align(source, target, { null_prior: 0.2 } as unknown as AlignOptions),
);
checkThrows("rejects a non-corpus argument", () => align("not a corpus" as unknown as Corpus, target));
checkThrows("rejects a malformed alignment", () => formatMoses([[[0]]] as unknown as Alignment));

// ── Result ────────────────────────────────────────────────────────────────────

if (failures > 0) {
  throw new Error(`${failures} check(s) failed.`);
} else {
  console.log("All checks passed.");
}
