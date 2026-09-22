// The type check runs with nothing but the TypeScript defaults installed, so
// `node:fs` has no declarations. Rather than pull in @types/node for one call,
// declare the single function type-check-runtime.ts uses to load the WASM.
declare module "node:fs" {
  export function readFileSync(path: URL): Uint8Array;
}
