# Releasing

How releases of Intel GPU AI Skills are versioned, validated, and published.

`CONTRIBUTING.md` covers the contributor side: licensing, DCO sign-off,
[branch naming](CONTRIBUTING.md#branch-naming), the fork-and-PR workflow, approvals, and
validation gates. This document covers what maintainers do after a change lands on
`main`.

Maintainers are responsible for release readiness, version selection, tagging, release
notes, and keeping the required security and compliance checks satisfied.

## Versioning

Version numbers follow `vMAJOR.MINOR.PATCH`:

- **PATCH** — backward-compatible bug fixes, documentation, or maintenance
- **MINOR** — new skills or backward-compatible capabilities
- **MAJOR** — breaking changes to a skill interface, script flags, or repository layout

Before 1.0, a MINOR release may include breaking changes, identified under a
**Breaking changes** heading in the release notes. Maintainers select the initial
public release version.

## Cutting a release

Releases are cut from an approved commit on `main` once the checklist below is
complete. After validating `main`, create an annotated, signed tag:

```sh
git switch main && git pull --ff-only
git tag -s vX.Y.Z -m "gpu-ai-skills vX.Y.Z"
git push origin vX.Y.Z
```

Release tags must be annotated and signed so provenance can be verified; key
management follows the maintainers' approved practice. A plain `git tag vX.Y.Z` creates
a lightweight tag with no author, date, or signature and is not acceptable for a
release.

Then publish a GitHub Release with notes based on the changes since the previous
release. Those notes are this project's changelog — there is no `CHANGELOG.md` file to
keep in step.

Once published, a release tag must not be moved, deleted and reused, or repointed to a
different commit. Corrections require a new version.

## Release checklist

- [ ] `bash scripts/check-skills.sh` passes
- [ ] `python3 scripts/generate_agents.py` leaves `agents/AGENTS.md` unchanged
- [ ] The `HOW_TO_TEST.md` acceptance layer has run on an appropriate Intel GPU; record
      the tested device in the release notes
- [ ] `BOM.md` matches the release tree if third-party components changed
- [ ] `README.md` install instructions validated from a clean environment
- [ ] Required CI, security, licensing, and compliance gates are green
- [ ] The release tag is annotated and signed
- [ ] Release notes cover significant changes, breaking changes, known limitations, and
      credit external contributors

## Hotfixes

For a critical issue: assess it immediately, fix it on `main` with regression coverage,
obtain the required review and green CI, then publish a patch release. Do not bundle
unrelated changes into a hotfix.

If the issue is a security vulnerability, follow `SECURITY.md` and its coordinated
disclosure process. Do not disclose details publicly before that process allows it.

## New dependencies and third-party code

The project is intentionally lightweight. Adding a runtime dependency, a dependency
manifest, or vendored or redistributed third-party code affects reproducibility,
licensing, and security review.

- Call out new third-party dependencies or redistributed code in the PR description
- Update `BOM.md` with the component, version, license, and whether it is
  redistributed, vendored, or only referenced
- Pin versions; avoid floating references such as `:latest`
- Components with licensing obligations beyond the project's existing set require
  license review before merge
- Re-run the required security, license, and dependency checks when the dependency
  surface changes

## Quick reference

```text
Release      -> validate main -> select version -> signed tag -> push -> GitHub Release
Critical bug -> expedited assessment -> fix + test -> review/CI -> patch release
Security     -> SECURITY.md -> coordinated handling -> fix -> release
```

Related: `CONTRIBUTING.md` (contributing and triage), `HOW_TO_TEST.md` (testing),
`SECURITY.md` (security reporting), `BOM.md` (third-party component record).
