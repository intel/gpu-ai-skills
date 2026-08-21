# Contributing

### License

Intel GPU AI Skills is licensed under the terms in [LICENSE](LICENSE). By contributing to the project, you agree to the license and copyright terms therein and release your contribution under these terms.

### Sign your work

Please use the sign-off line at the end of the patch. Your signature certifies that you wrote the patch or otherwise have the right to pass it on as an open-source patch. The rules are pretty simple: if you can certify
the below (from [developercertificate.org](http://developercertificate.org/)):

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

Then you just add a line to every git commit message:

    Signed-off-by: Joe Smith <joe.smith@email.com>

Use your real name (sorry, no pseudonyms or anonymous contributions.)

If you set your `user.name` and `user.email` git configs, you can sign your
commit automatically with `git commit -s`.

## Development workflow

Run the local checker before every skill change:

```sh
bash scripts/check-skills.sh
```

This runs the repo static gate and, when `skill-validator` is installed,
the external Agent Skills validator.

The static checks are stdlib-only, but the guardrails step of this gate
(`guardrails/check.py`) needs PyYAML, so install it once per environment:

```sh
python3 -m pip install pyyaml
```

Without it that step exits `PyYAML is required: pip install pyyaml` and the
gate fails. See [Layer 1 in HOW_TO_TEST.md](HOW_TO_TEST.md#layer-1--static-checks-10-seconds-no-gpu)
for the full prerequisite list.

Install the external validator:

```sh
brew tap agent-ecosystem/tap
brew install skill-validator
```

To require the external validator, use:

```sh
REQUIRE_SKILL_VALIDATOR=1 bash scripts/check-skills.sh
```

The static gate needs PyYAML (frontmatter checks) and libcst (the `xpu-port`
scan/rewrite regressions). Install both with `pip install pyyaml libcst`; a
missing one is reported as a failure rather than a silent skip, so set
`SKIP_OK=1` if you deliberately want to run without them.

The repo static gate verifies:

- JSON manifests parse
- every `SKILL.md` has `name` and `description`
- every `SKILL.md` frontmatter block parses as YAML (an unquoted colon in a
  description makes the skill invisible to `npx skills` and other consumers —
  quote the whole value when it contains `: `)
- skill name matches its directory
- Python helper scripts compile
- shell scripts parse
- `agents/AGENTS.md` matches the generator
- Claude marketplace entries cover every skill
- shipped docs do not reference internal scratch paths
- shipped scripts avoid destructive cleanup patterns
- docs do not advertise TODO command behavior
- generated `agents/AGENTS.md` has stable markdown shape

The external validator checks Agent Skills structure, token size, internal
links, content metrics, and cross-language contamination. The local wrapper
skips external link checks by default to avoid network flake in normal
development and allows the pack's intentional `data/` support directory.

Neither checker proves that an agent will choose the skill correctly or that
the commands in the skill work on a real host. For that, run the relevant
acceptance layer in `HOW_TO_TEST.md`.

When adding or updating a skill:

- keep the directory name and `name` frontmatter identical
- keep `description` specific about what the skill does and when to use it
- keep `SKILL.md` focused; move bulky detail into supporting files
- use public sources or inline local verification recipes
- do not reference `research/` from shipped skills
- avoid destructive commands such as `pkill`, `docker rm -f`, or broad `rm -rf`
- add the new skill to `catalog/bundles.yaml` under the right category
  (and any cross-cutting bundles it belongs to)
- run `python3 scripts/generate_agents.py` after skill metadata changes
  (the pre-commit hook below does this automatically)
- run `bash scripts/check-skills.sh` before commit

## Pre-commit hook

A local pre-commit hook keeps `agents/AGENTS.md` in sync with skill
frontmatter so you don't have to remember to regenerate it. Install it
once per clone:

```sh
pip install pre-commit
pre-commit install
```

After that, every `git commit` runs `python3 scripts/generate_agents.py`
and re-stages `agents/AGENTS.md` if it changed, so the refreshed bundle
lands in the same commit. The hook runs unconditionally rather than
filtering on `SKILL.md` paths so that staged skill *deletions* — which
pre-commit's default path filter drops — still trigger a regeneration.
When nothing skill-related changed, the generator is a no-op and adds
no files to the commit.

If the hook ever blocks an emergency fix, `git commit --no-verify`
bypasses it for that single commit. Use this only as an escape hatch
and follow up with a normal commit that includes the regenerated
`agents/AGENTS.md`; `bash tests/static.sh` will otherwise fail on drift.
