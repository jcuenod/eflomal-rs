// Node entry point: instantiate the WASM module from disk so that callers do
// not have to await an initializer. `init` is exported as a no-op so the same
// code works on the web, where the module has to be fetched.
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { initSync } from "./eflomal_wasm.js";

const here = dirname(fileURLToPath(import.meta.url));
const output = initSync({ module: readFileSync(join(here, "eflomal_wasm_bg.wasm")) });

export default async function init() {
  return output;
}

export { align, alignDetailed, formatMoses, initSync } from "./eflomal_wasm.js";
