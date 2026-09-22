// Turn a wasm-pack `pkg` directory into the published npm package: merge in the
// fields wasm-pack cannot derive from Cargo.toml, and stamp the release version.
//
//   node apply-package-overrides.mjs <pkg-dir> <version>
import { readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const [directory, version] = process.argv.slice(2);
if (!directory || !version) {
  console.error("usage: apply-package-overrides.mjs <pkg-dir> <version>");
  process.exit(1);
}

const here = dirname(fileURLToPath(import.meta.url));
const overrides = JSON.parse(readFileSync(join(here, "package-overrides.json"), "utf8"));

// npm validates the published manifest against the repository recorded in the
// provenance statement and rejects the publish outright if they disagree, so a
// repository rename shows up as a burnt version number at the very last step.
// Fail here instead, in every job that assembles the package.
const slug = process.env.GITHUB_REPOSITORY;
if (slug) {
  const canonical = `https://github.com/${slug}`;
  const declared = (overrides.repository?.url ?? "").replace(/^git\+/, "").replace(/\.git$/, "");
  if (declared !== canonical) {
    console.error(
      `package-overrides.json points at ${declared || "<nothing>"}, but this workflow runs in ${canonical}.\n` +
        "npm checks the manifest against the provenance statement and will reject the publish. " +
        "Update repository.url, homepage and bugs.url to match.",
    );
    process.exit(1);
  }
}

const manifest = join(directory, "package.json");
const generated = JSON.parse(readFileSync(manifest, "utf8"));

writeFileSync(manifest, `${JSON.stringify({ ...generated, ...overrides, version }, null, 2)}\n`);
console.log(`${manifest}: ${overrides.name}@${version}`);
