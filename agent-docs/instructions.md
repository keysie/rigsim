# Instructions (permanent) — rigsim

This file holds the standing rules for working in this repository. Read it fully at the start of every session.
Companion repository: [rigsim-benchmark](https://github.com/keysie/rigsim-benchmark), which has its own agent-docs.

## Permanent instructions

These apply to every agent in every session. The user may override one for a single task; it stays in force afterwards.

### Working with the user

1. **Propose first, build after clearance.** Never just write code. Describe the change (what, where, why, risks,
   tests) in the answer and wait for the user's go-ahead. A proposal is short, not an essay.
   Exception: the agent-docs themselves (rule 3) are updated without asking, on the current sprint branch (rule 7).
2. **Say what you are doing** in one line before any task that takes several tool calls.
3. **Keep agent-docs updated with every answer.** Before finishing an answer, check each file in this folder:
   journal (new decision or finding, dated), open-points (added, reprioritised, closed), findings (new facts added,
   stale ones corrected or removed), briefing (still true? still under 16k tokens?). If nothing changed, change nothing.
   Never leave a statement standing that you know is superseded.
   The briefing's section 0 ("Now") must always hold the direct next steps and the few most recently solved items
   (about five, newest first, merged or closed only; open PRs belong under next steps). Update it whenever the state
   changes, and drop the oldest solved item when adding a new one.
4. **Say what is verified and what is not.** Mark claims as *measured*, *read from source/docs*, *inferred* or
   *user decision*. Never present an inference as a fact.
5. **Keep answers short.** The user prefers short answers. Details belong in these files, not in the chat.

### Git

6. **Conventional commit messages**: `<type>[(<scope>)][!]: <description>`, type one of feat, fix, docs, style,
   refactor, perf, test, build, ci, chore, revert. CI and `.githooks/commit-msg` reject anything else. Once per clone:
   `git config core.hooksPath .githooks`.
7. **Branches are sprints** (user decision 2026-10-31). Code, configuration and docs never go to `main` directly.
   A branch holds one sprint: a stretch of related work with one goal (e.g. "the stick collects the right data"), not a
   single feature or fix. All work of the sprint goes into it — fixes, features, tests **and the agent-docs**, which
   always live on the branch being worked on. Name it `sprint/<goal>`; a sprint touching both repositories uses the same
   branch name in both. Before starting, check for an open sprint branch: work fits its goal → continue there; it does
   not → ask the user whether to close the sprint (PR, user merges) before opening the next one from `main`. Never
   rename a branch with an open PR (GitHub closes the PR). One change per commit; agent-docs commits separate from code
   commits (`docs(agent-docs): …`).
   Keep a PR open for the sprint (draft while work goes on) so the user sees it. The user merges; agents never merge.
8. **Run the tests before and after** a change (commands in the briefing). Never hand over a red branch without saying so.

### Secrets and data

9. **The GitHub token is a secret.** Use it only in shell commands as an HTTP header; never write it into a file, commit,
   git config, remote URL, journal or reply. Each bash call is a fresh shell: re-export it per command. Remind the user
   to revoke it at the end of the session.
   ```
   AUTH=$(printf 'x-access-token:%s' "$GH_TOKEN" | base64 -w0)
   git -c http.extraheader="Authorization: Basic $AUTH" push origin <branch>
   ```
10. **No personal data, hostnames, serial numbers, usernames or other identifying data** in files, commits or docs.
    If the user pastes such data, say so and ask for test data instead.

## Repository-specific rules

11. **Mechanism over fit.** The overhaul's principles (`docs/overhaul/engine.md` §2) are binding for model work: no
    constant exists only to make an end-to-end number come out right; device-specific values come from spec sheets,
    measurements set only a few global constants; where a result disagrees, examine the model first, never patch with
    a per-device constant. Propose model changes against these principles.
12. **Calibrate at the bottom, validate at the top.** Do not feed validation points into constants. Anything that
    changes `backend/tests/test_validation.py` bounds needs an explanation in the PR.
13. **The prediction contract is shared with rigsim-benchmark.** `backend/app/predict.py` (CLI, output format, keys) is
    consumed by that repo's daily job at release tags. Changing it is a cross-repo change: open both PRs, say which
    merges first, and tag only when both agree. Never delete or retag a published `vX.Y.Z`.
14. **Do not touch** `measurements/*.json` (user-submitted records, intake-validated) or `frontend/public/*` generated
    folders by hand.
15. **Do not describe the model as more accurate than the evidence** (README states ±25 % on one machine's path).
