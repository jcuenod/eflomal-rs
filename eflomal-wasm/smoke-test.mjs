// Verifies an installed `eflomal` package: that it resolves through the package
// name and its "exports" map, and that it aligns end to end. Correctness of the
// alignment itself is covered by type-check-runtime.ts; this checks the
// published artifact loads and runs.
import init, { align, alignDetailed, formatMoses } from "eflomal";

await init();

const source = [
  "the cat sat",
  "the world is big",
  "the universe is vast",
  "the cat is big",
  "hello world",
  "hello universe",
  "goodbye world",
  "the big world",
];
const target = [
  "le chat sassit",
  "le monde est grand",
  "lunivers est vaste",
  "le chat est grand",
  "bonjour monde",
  "bonjour univers",
  "au revoir monde",
  "le grand monde",
];

const links = align(source, target, { seed: 1 });
console.log(formatMoses(links).trimEnd());

if (links.length !== source.length) {
  throw new Error(`expected ${source.length} entries, got ${links.length}`);
}
if (!links.some((sentence) => sentence.length > 0)) {
  throw new Error("no links were produced");
}
for (const [index, sentence] of links.entries()) {
  const sourceLength = source[index].split(" ").length;
  const targetLength = target[index].split(" ").length;
  for (const [i, j] of sentence) {
    if (i < 0 || i >= sourceLength || j < 0 || j >= targetLength) {
      throw new Error(`link ${i}-${j} out of range in sentence ${index}`);
    }
  }
}

if (JSON.stringify(align(source, target, { seed: 1 })) !== JSON.stringify(links)) {
  throw new Error("alignment is not deterministic for a fixed seed");
}

const detailed = alignDetailed(source, target, { scores: true, seed: 1 });
if (detailed.forwardScores?.length !== source.length) {
  throw new Error("alignDetailed did not return one forward score per sentence");
}

console.log("Smoke test passed.");
