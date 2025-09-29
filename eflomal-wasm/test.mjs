// node --experimental-wasm-modules test.mjs

import * as eflomal from './pkg/eflomal_wasm.js'; // adjust filename if different

const out = eflomal.align_plaintext(
  "hello\nworld",
  "bonjour\nmonde",
  0,0,1,1,1,1,0.0,false,42n,false,false
);

console.log('links_moses:', out.links_moses);
console.log('stats:', out.stats);
console.log('scores:', out.scores);
