# aither-patch

Sandbox-executed `apply_patch` binary (aither #104): a small, portable
executable that applies contextual patches read from stdin. All parsing,
matching and file writes run inside the target environment; authority over
the filesystem belongs to the enclosing shell sandbox (heel), never to a
host IPC service.

## Usage

```
apply_patch <<'PATCH'
*** Begin Patch
*** Update File: src/main.rs
@@
-old()
+new()
*** End Patch
PATCH
```

- `apply_patch` — read a patch from stdin; print `A`/`M`/`D`/`R` lines for
  each touched file; exit nonzero without writing on any failure.
- `apply_patch --help` — format documentation.
- `apply_patch --version` — deterministic `apply_patch 0.7.0`.

Supported operations: `Add File`, `Update File` (with `*** Move to:`),
`Delete File`, `*** End of File` anchors and `@@` context. Context lines
must match the file verbatim (the engine retries in whitespace-lenient
mode before failing). Ambiguous context is diagnosed, never guessed; every
file is validated before any write, so a bad patch leaves the tree
untouched — a validation guarantee, not a crash-atomicity promise.

## Engine

This crate owns the whole engine. `src/parse.rs` parses the patch into typed
actions; `src/matcher.rs` places each hunk: its old side (context and `-`
lines) must match the file contiguously at or after the previous hunk's end
and after the `@@` anchor line, first with exact whitespace, then
whitespace-insensitively. Zero matches is a conflict naming the hunk and the
first line that did not match; several matches is an ambiguity error listing
the lines. File-level semantics — unconditional delete, literal paths, mode
preservation, all-or-nothing validation — live in `src/lib.rs`.

## Artifact build

Prebuilt delivery lets us optimize the whole artifact. `build-artifact.sh
<target>` builds only this binary with the pinned nightly
`nightly-2026-10-01` (+`rust-src`), `-Zbuild-std=std,panic_abort` and the
workspace `apply-patch` profile (`opt-level=z`, `lto`, `codegen-units=1`,
`strip=symbols`, `panic=abort`; `APPLY_PATCH_ABORT=immediate` adds
`-Cpanic=immediate-abort`). Neither MSRV nor any other crate's release
behavior is affected; ordinary development stays on stable.

Measured executable bytes (`apply_patch` 0.6.0, x86-64):

| build | size |
| --- | --- |
| stable `--release`, stripped | 494,928 |
| `apply-patch` profile (`z`, lto, cu=1) | 394,624 |
| + `panic=abort` | 384,360 |
| + nightly `-Zbuild-std` (panic_unwind abort shim) | 360,440 |
| + `-Zbuild-std` `panic_abort` | 336,240 |
| + `-Cpanic=immediate-abort` | **107,952** |
| aarch64-unknown-linux-musl (same recipe) | 116,832 |
| x86_64-unknown-linux-musl (same recipe) | 123,584 |

Notes from the experiment:

- `opt-level=z` beats `s` (~0.07% smaller).
- `-Zbuild-std-features` (any value, including `optimize_for_size`) fails
  on nightly-2026-10-01 with `E0152 duplicate lang item` — the feature
  list is left unset; `-Zbuild-std=std,panic_abort` is the supported form.
- `immediate-abort` changes *panic* diagnostics only (a bug would abort
  with a location but no message); ordinary patch errors are returned
  `Result`s and keep their full text. It is the shipped default because
  this binary's error surface is validation failures, not panics; set
  `APPLY_PATCH_ABORT=panic` for standard abort diagnostics.
- No packer and no exotic linking: the Linux artifacts are ordinary
  static musl executables (the native Linux build links plain glibc).
- `rust-lld` supplies the linker so musl targets build without a system
  cross toolchain (self-contained crt objects come from the target's
  rust-std component).

Targets: Linux, macOS and Windows, each on x86-64 and ARM64. The Linux
musl artifacts are static and run on any libc, in containers and on SSH
hosts. `artifact.toml` documents the build and usage contract; the
initializer consumes a `ToolDistribution` JSON manifest with pinned URLs
and hashes. Stable release assets are tracked in #113; until publication,
the verified script output must be staged.
