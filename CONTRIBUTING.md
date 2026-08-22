# Contributing

Thank you for your interest in improving Intel GPU AI Skills. This guide
covers the license terms, the sign-off requirement, the branching workflow,
the review process, and the validation gate every change must pass.

We welcome pull requests of these kinds:

- **New skills**: a run, benchmark, profile, or tooling workflow for Intel
  GPUs that the pack does not cover yet.
- **Fixes to existing skills**: wrong flags, stale image or package
  references, broken commands, missing edge cases.
- **Performance and accuracy improvements**: better launch recipes or
  configuration guidance, backed by numbers from the benchmark skills.
- **Docs and tooling**: the validation gate, install script, templates,
  and repository documentation.

## License

Intel GPU AI Skills is licensed under the terms in [LICENSE](LICENSE).
By contributing to the project, you agree to the license and copyright terms
therein and release your contribution under these terms.

## Sign your work

Please use the sign-off line at the end of each patch. Your signature
certifies that you wrote the patch or otherwise have the right to pass it on
as an open-source patch. The rules are simple: if you can certify the below
(from [developercertificate.org](http://developercertificate.org/)):

```
Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
660 York Street, Suite 102,
San Francisco, CA 94110 USA

Everyone is permitted to copy and distribute verbatim copies of this
license document, but changing it is not allowed.

Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
    have the right to submit it under the open source license
    indicated in the file; or

(b) The contribution is based upon previous work that, to the best
    of my knowledge, is covered under an appropriate open source
    license and I have the right under that license to submit that
    work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am
    permitted to submit under a different license), as indicated
    in the file; or

(c) The contribution was provided directly to me by some other
    person who certified (a), (b) or (c) and I have not modified
    it.

(d) I understand and agree that this project and the contribution
    are public and that a record of the contribution (including all
    personal information I submit with it, including my sign-off) is
    maintained indefinitely and may be redistributed consistent with
    this project or the open source license(s) involved.
```

then add a line to every git commit message:

    Signed-off-by: Joe Smith <joe.smith@email.com>

Use your real name. Pseudonyms and anonymous contributions are not accepted.

If your `user.name` and `user.email` git configs are set, `git commit -s`
adds the sign-off automatically.

## Branching and pull requests

All changes land through pull requests against `main`. Do not commit to
`main` directly.

1. Fork the repository, or clone it directly if you have write access.
2. Create a topic branch from the tip of `main`. Use a short descriptive
   name such as `fix/vllm-xpu-run-typo` or `skill/xpu-newthing`.
3. Make your change. Keep each commit to one logical change and sign it off.
4. Run the validation gate described below and make sure it passes.
5. Push the branch and open a pull request against `main`. Explain what
   changed and why, and link any related issue.
6. Address review feedback with follow-up commits or a rebase. Rebase on
   `main` rather than merging `main` into your branch.

## Review process

- Reviewers are requested automatically through `.github/CODEOWNERS`. You
  can also request a review from any maintainer listed at the end of this
  guide.
- Open work in progress as a draft pull request. Mark it ready for review
  once it is complete and the validation gate passes.
- Simple changes need at least one maintainer approval before merging.
- Complex changes, such as a new skill or a cross-cutting tooling change,
  need at least two maintainer approvals.
- All CI checks must pass. CI runs the same gate as
  `bash scripts/check-skills.sh`, so a green local run means a green CI run.
- When enabling something new, update the relevant tests, and include
  benchmark data when claiming a performance improvement.

## Validation gate

Run the local checker before every change:

```sh
bash scripts/check-skills.sh
```

This runs the repo static gate (`bash tests/static.sh`) and, when
`skill-validator` is installed, the external Agent Skills validator over
every skill directory.

Two tools extend the gate's coverage and are worth installing:

- `ripgrep` (`rg`): the documentation and script pattern checks in
  `tests/static.sh` depend on it.
- `libcst` (`pip install libcst`): needed by the `xpu-port` regression
  test. Without it that test prints a `SKIP` notice and is not run.

The external validator is
[agent-ecosystem/skill-validator](https://github.com/agent-ecosystem/skill-validator);
its README covers installation on any platform.

To make the external validator mandatory instead of optional:

```sh
REQUIRE_SKILL_VALIDATOR=1 bash scripts/check-skills.sh
```

The repo static gate verifies:

- JSON manifests parse
- every `SKILL.md` has `name` and `description` frontmatter, and the
  `name` matches its directory
- Python helper scripts compile and shell scripts parse
- mocked integration checks pass for `xpu-runtime-preflight`,
  `model-can-it-fit`, `xpu-port`, and skill routing
- `agents/AGENTS.md` matches the generator output
- every skill appears in `catalog/bundles.yaml`
- Claude marketplace entries cover every skill
- shipped docs do not reference internal scratch paths
- shipped scripts avoid destructive cleanup patterns
- docs do not advertise unimplemented command behavior
- generated `agents/AGENTS.md` has a stable markdown shape

The external validator checks Agent Skills structure, token size, internal
links, content metrics, and cross-language contamination. The local wrapper
skips external link checks by default to avoid network flake in normal
development and allows the pack's intentional `data/` support directory.

Neither checker proves that an agent will choose the skill correctly or that
the commands in a skill work on a real host. For that, run the relevant
acceptance layer in [HOW_TO_TEST.md](HOW_TO_TEST.md).

## Adding or updating a skill

- start from `template/SKILL.md`
- keep the directory name and the `name` frontmatter identical
- keep `description` specific about what the skill does and when to use it
- quote the whole `description` value when it contains a colon followed by
  a space; an unquoted colon breaks YAML parsing and makes the skill
  invisible to consumers
- keep `SKILL.md` focused; move bulky detail into supporting files
- use public sources or inline local verification recipes
- do not reference `research/` from shipped skills
- avoid destructive commands such as `pkill`, `docker rm -f`, or broad
  `rm -rf`
- add the new skill to `catalog/bundles.yaml` under the right category,
  plus any cross-cutting bundles it belongs to
- run `python3 scripts/generate_agents.py` after skill metadata changes
  (the pre-commit hook below does this automatically)
- run `bash scripts/check-skills.sh` before commit

## Pre-commit hook

A local pre-commit hook keeps `agents/AGENTS.md` in sync with skill
frontmatter so you do not have to remember to regenerate it. Install it once
per clone:

```sh
pip install pre-commit
pre-commit install
```

After that, every `git commit` runs `python3 scripts/generate_agents.py` and
re-stages `agents/AGENTS.md` if it changed, so the refreshed bundle lands in
the same commit. The hook always runs rather than filtering on `SKILL.md`
paths, so staged skill deletions also trigger a regeneration. When nothing
skill-related changed, the generator is a no-op and adds no files to the
commit.

If the hook ever blocks an emergency fix, `git commit --no-verify` bypasses
it for that single commit. Use this only as an escape hatch and follow up
with a normal commit that includes the regenerated `agents/AGENTS.md`;
`bash tests/static.sh` will otherwise fail on drift.

## Maintainers and contributors

Maintainers:

- Yuning Qiu ([@YuningQiu](https://github.com/YuningQiu),
  <yuning.qiu@intel.com>)
- Susan Liu ([@258SusanLiu](https://github.com/258SusanLiu),
  <susan1.liu@intel.com>)
- Chun Tao ([@ctao456](https://github.com/ctao456),
  <chun.tao@intel.com>)
- Gopesh Khandelwal ([@GopeshKh](https://github.com/GopeshKh),
  <gopesh.khandelwal@intel.com>)
- Rahul Unnikrishnan Nair ([@unrahul](https://github.com/unrahul),
  <rahul.unnikrishnan.nair@intel.com>)

Contributors:

- Jerry Zhang ([@inteljerry](https://github.com/inteljerry),
  <jerry.zhang@intel.com>)
- Akash Dhamasia ([@akashdhamasia12](https://github.com/akashdhamasia12),
  <akash.dhamasia@intel.com>)
- Min Sung Kim ([@mins2022](https://github.com/mins2022),
  <min.sung.kim@intel.com>)
- Gilliean Lee ([@gilliean](https://github.com/gilliean),
  <gilliean.lee@intel.com>)
- Kushal Mittal ([@kushal2705](https://github.com/kushal2705),
  <kushal.mittal@intel.com>)
- Vishnu V Ravi (<vishnu.v.ravi@intel.com>)
- Zhiqi Tao ([@zhiqitao-intel](https://github.com/zhiqitao-intel),
  <zhiqi.tao@intel.com>)
- Dunni Aribuki (<dunni.aribuki@intel.com>)
- Yogini Dandekar ([@yogit2020](https://github.com/yogit2020),
  <yogini.dandekar@intel.com>)
