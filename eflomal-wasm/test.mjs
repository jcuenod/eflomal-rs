// node --experimental-wasm-modules test.mjs

const src = "hello\nworld\nhello world\nhello universe\nthe world is big\nthe universe is vast\nhello world\ngoodbye world";
const tgt = "bonjour\nmonde\nbonjour monde\nbonjour univers\nle monde est grand\nl'univers est vaste\nbonjour le monde\nau revoir le monde";
import * as eflomal from './pkg/eflomal_wasm.js';

console.log('=== Test 1: Simple alignment with defaults ===');
try {
  const simple_result = eflomal.align_simple(
    src,
    tgt
  );
  console.log('Simple alignment result:', simple_result);
} catch (e) {
  console.error('Simple alignment error:', e);
}

console.log('\n=== Test 2: Default config (symmetrized) ===');
// Using default config - this will do symmetrization by default
const default_config = new eflomal.AlignConfig();
const default_out = eflomal.align_text(
    src,
    tgt,
  default_config
);

console.log('links (auto-selected):', default_out.links);
console.log('links_forward:', default_out.links_forward);
console.log('links_reverse:', default_out.links_reverse);
console.log('links_symmetrized:', default_out.links_symmetrized);
console.log('stats:', default_out.stats);
console.log('scores_forward:', default_out.scores_forward);
console.log('scores_reverse:', default_out.scores_reverse);

console.log('\n=== Test 3: Custom config (forward only) ===');
// Custom config - equivalent to your old test parameters
const config = new eflomal.AlignConfig();
// Set scalar properties using the property names exported by wasm-bindgen
// (wasm-bindgen exposes setters as JS properties without the `set_` prefix)
config.model = 0;           // model: 0
config.score_model = 0;     // score_model: 0
// Option 1: Set iterations individually (these are exported as properties)
config.it1 = 1;
config.it2 = 1;
config.it3 = 1;
// Option 2: Or use the method to set all at once
config.set_iterations(1, 1, 1);

config.n_samplers = 1;      // n_samplers: 1
config.null_prior = 0.0;    // null_prior: 0.0
config.seed = 42n;           // seed: 42

// Option 1: Set direction flags individually (use property names)
config.forward = true;
config.reverse = false;
config.symmetrize = false;
// Option 2: Or use the method
config.set_direction(true, false, false);

// Set output flags
// Disable stats/scores via properties
config.want_stats = false;
config.want_scores = false;
// Or use the method
config.set_outputs(false, false);

const custom_out = eflomal.align_text(
  src,
  tgt,
  config
);

console.log('links_forward:', custom_out.links_forward);
console.log('links_reverse:', custom_out.links_reverse);
console.log('links_symmetrized:', custom_out.links_symmetrized);
console.log('stats:', custom_out.stats);
console.log('scores:', custom_out.scores_forward);

console.log('\n=== Test 4: Bidirectional with symmetrization and all outputs ===');
const full_config = new eflomal.AlignConfig();
full_config.model = 3;
full_config.forward = true;
full_config.reverse = true;
full_config.symmetrize = true;
full_config.want_stats = true;
full_config.want_scores = true;
full_config.seed = 42n;

const full_out = eflomal.align_text(
  src,
  tgt,
  full_config
);

console.log('links_forward:', full_out.links_forward);
console.log('links_reverse:', full_out.links_reverse);
console.log('links_symmetrized:', full_out.links_symmetrized);
console.log('stats:', full_out.stats.slice(0, 10)); // Display first 10 stats for brevity
console.log('scores_forward:', full_out.scores_forward);
console.log('scores_reverse:', full_out.scores_reverse);

//  Don't know how to set priors in the config yet, so skipping this test
// console.log('\n=== Test 5: With priors ===');
// const priors_config = new eflomal.AlignConfig();
// priors_config.priors = "prior data here"; // Set translation priors
// priors_config.model = 3;

// const priors_out = eflomal.align_text(
//   src,
//   tgt,
//   priors_config
// );

// console.log('Alignment with priors:', priors_out.links);