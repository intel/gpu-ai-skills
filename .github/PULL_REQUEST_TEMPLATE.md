## What changed and why

<!-- Describe the change and the motivation. Link any related issue. -->

## Type of change

<!-- Check all that apply. -->

- [ ] New skill
- [ ] Fix to an existing skill
- [ ] Performance or accuracy improvement (include benchmark data below)
- [ ] Docs or tooling

## Validation

- [ ] `bash scripts/check-skills.sh` passes locally
- [ ] `python3 scripts/generate_agents.py` was run after any `SKILL.md`
      frontmatter change (or the pre-commit hook handled it)
- [ ] For skill behavior changes, the relevant acceptance layer in
      `HOW_TO_TEST.md` was run on real hardware, or the reason it was not
      is explained above
- [ ] Every commit is signed off (`git commit -s`), per the Developer
      Certificate of Origin in CONTRIBUTING.md

## Benchmark data

<!-- Required for performance claims. Delete this section otherwise. -->
