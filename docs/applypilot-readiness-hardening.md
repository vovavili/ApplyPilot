# Apply readiness hardening

ApplyPilot now has a strict readiness gate before auto-apply. A job can be stored,
scored, tailored, and cover-lettered at different times, but it is not allowed
into the apply queue unless all required artifacts pass the same checker.

This exists because manual-review jobs were still able to burn LLM calls and reach
application prep. The worst examples were foreign or ambiguous locations such as
Noida, Seattle, Toronto, Costa Rica, Shanghai, Zug, and "Multiple Locations".
Those rows can stay in the database for inspection, but they must not be applied
without explicit human promotion.

## What "ready" means

A ready job must pass all of these checks:

- `review_status` is not `manual_review`.
- `fit_score` is at least the selected minimum score.
- `application_url`, tailored resume path, and cover letter path are present.
- The location triage result is `accept`, using the same Netherlands, EU, Europe,
  EMEA, and remote rules used during discovery.
- Tailored resume and cover letter files exist and are not empty.
- Artifact filenames include the URL digest, which prevents two jobs with the same
  title from sharing one resume or cover letter.
- No other unapplied job points at the same tailored resume or cover letter path.
- The tailored resume report exists and has `status: "approved"`.
- The cover letter report exists and has `status: "approved"`.
- The cover letter still passes local validation: correct greeting, full body,
  complete sign-off, no obvious truncation, and no LLM self-talk.

`approved_with_judge_warning` is deliberately not ready. It may be fine after a
human reads it, but the bot should not submit it unattended.

## Cover letter generation

Cover letters now use a structured path. The model returns six JSON fields, and
ApplyPilot assembles the greeting, three body paragraphs, and exact sign-off in
code. That makes missing greetings, half-finished letters, and stray chatbot
preambles much harder to sneak through.

The cover stage reads the tailored resume for each job. If that file is missing,
it falls back to `resume.txt`. It also skips jobs whose location triage is not
accepted, even if an older row was never marked `manual_review`.

Cover letters are now checked for useful length, not just basic structure. The
target is roughly 170-285 words: long enough to give real evidence, short enough
to stay on one page. A technically complete but tiny letter is treated as a
failed attempt, retried, and then replaced by the deterministic fallback if the
model keeps under-writing.

`COVER_LLM_MODEL` can be set in `.env` to use a stronger model for cover letters
without changing scoring or resume tailoring. If the value starts with
`claude-`, or if `COVER_LLM_PROVIDER=anthropic`, ApplyPilot uses Anthropic's
Messages API and `ANTHROPIC_API_KEY`. If it is unset, ApplyPilot keeps using the
normal LLM provider and model.

`TAILOR_LLM_MODEL` works the same way for resume tailoring. Your current setup
keeps bulk scoring on `LLM_MODEL=gemini-3.5-flash`, while routing resume
tailoring and cover letters to `claude-opus-4-7`. Discovery, enrichment, and
scoring stay cheaper; the documents a recruiter actually reads get the stronger
model.

Fallback is still there, but only as a safety net. Reports now include
`fallback`, `fallback_reason`, LLM errors, validation errors, and short redacted
excerpts from failed attempts. The batch summary also reports how many generated
letters used fallback.

## Commands

Run the audit before applying:

```powershell
applypilot audit
```

Print the same information as JSON:

```powershell
applypilot audit --json
```

Clean unsafe rows after reviewing the audit:

```powershell
applypilot audit --fix
```

`--fix` writes a SQLite backup in `C:\Users\latvi\.applypilot` before it changes
anything. It marks unsafe geography as `manual_review` and clears broken artifact
paths so ApplyPilot has to regenerate them.

## Test coverage

The hardening tests cover the failure modes that caused the messy run:

- manual-review rows do not get scored, tailored, cover-lettered, or applied
- current bad location examples are blocked
- duplicate artifact paths are blocked
- truncated cover letters are blocked even if the DB has a path
- judge-warning resume reports are blocked
- low salary postings are skipped only after readiness succeeds
- filename prefixes are shared between resume and cover letter generation and
  always include the URL digest

The filename prefix test uses Hypothesis because the invariant matters across
many possible titles, sites, and URLs. A few hand-picked titles would not give
much confidence there.
