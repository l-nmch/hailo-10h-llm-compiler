---
name: preflight
description: Repo hygiene check before committing or publishing — scans for proprietary material, dates, and leaked internal paths/hostnames. Run before every commit to this repo or push to the wiki.
---

# Preflight check

This repo must stay redistributable and evergreen at all times (see
[CONTRIBUTING.md](../../../CONTRIBUTING.md)). Run these checks on the
staged/changed files before committing — not just once at repo creation.

## 1. No proprietary binaries tracked

```bash
git ls-files | grep -iE '\.(whl|hef|alls|hn)$|firmware'
```
Expect empty output. Also check untracked files about to be added:
```bash
git status --short
```
Anything matching those extensions must never be `git add`ed, even
temporarily.

## 2. No dates or timelines in content

```bash
grep -rniE '\b(20[12][0-9]-[0-9]{2}-[0-9]{2}|january|february|march|april|june|july|september|october|november|december)\b' \
  docs/ README.md CONTRIBUTING.md pipeline/ runtime/ 2>/dev/null
```
Watch for false positives (e.g. the word "may" as a verb) — read matches,
don't just count them. Real dates/changelog language must be rewritten as
evergreen prose.

## 3. No leaked internal infrastructure

```bash
grep -rniE 'hpc0|hpc1|10\.5\.0\.1|/home/[a-z]+-[a-z]+' \
  --include='*.md' --include='*.py' --include='*.ipynb' . 2>/dev/null | grep -v '.git/'
```
Any internal hostname, private IP, or absolute path under a personal
`$HOME` must be genericized (e.g. `$DFC_WORKDIR`, `your-host`) before
committing. A GitHub username in `.github/FUNDING.yml` is fine — that's
attribution, not a leak.

## 4. Large-file sanity

```bash
git ls-files -z | xargs -0 du -h 2>/dev/null | sort -rh | head -10
```
Nothing here should be a multi-MB artifact; the pipeline never writes
outputs inside the repo tree (`$DFC_WORKDIR` is external — see
[CONTRIBUTING.md](../../../CONTRIBUTING.md) code style rules).

## 5. Structural consistency

- Does the README's "Repository layout" tree still match `find . -maxdepth
  2`?
- If a new `docs/findings/*.md` was added, is it linked from
  [docs/findings/index.md](../../../docs/findings/index.md)?
- If [docs/status.md](../../../docs/status.md) claims changed, did the
  underlying stage-by-stage table get updated too?

Only after all five pass: stage, commit, and (if asked) push.
