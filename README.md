# Boutiqaat AI Search Visibility & Recommendation

Measures whether AI assistants surface Boutiqaat when a shopper asks a buying question, why they do
or do not, and what would have to change.

The headline result from the recorded corpus: **Boutiqaat appears in 15.9% of relevant Trials, in
0 of 36 global-English Trials, and on 0 of the 92 cited web pages that could be fetched.** It is not
being ranked below competitors in grounded answers. It is missing from the sources those answers are
assembled from.

---

## What I cut, and why

Read this first. The brief asks what was deliberately left out; these were choices, not omissions.

- **Multiple AI vendors.** The brief names ChatGPT, Perplexity and Google AI Overviews. Only one
  credential was available, so everything here measures OpenAI models. Spreading three days across
  three integrations would have produced three shallow ones. See `docs/adr/0004`.
- **Crawling Boutiqaat's own site** for structured data, shipping pages or English content coverage.
  It is a second project with its own failure modes, and fetching the pages the model actually cited
  answers the same question more directly.
- **Trend analysis across Runs.** Two Runs exist. A trend view would render an empty chart.
- **Cohen's kappa.** At n=17 comparable labels the statistic is noisy and invites an argument the raw
  agreement rate does not.
- **Automated UI tests.** The views are a thin read layer over a tested API; the highlighting and
  escaping logic, where a real mistake would hide, is tested directly. At this budget the time was
  better spent on the Gold Set.
- **Layered architecture.** Two seams exist, `Provider` and `PageFetcher`, because exactly two things
  leave the process. No repository interface, no service layer, no DI container. SQLite is exercised
  for real in tests.
- **Starting a Run from the UI.** A Run is 32 minutes of paid calls. Execution stays in the CLI; see
  `docs/adr/0003`.

## What the numbers do not support

- **API access is a proxy for consumer ChatGPT.** No memory, no personalization, no account history.
  A real shopper's answer may differ.
- **Three Trials is a small sample.** It is enough to show that answers vary — and that variation
  changed the conclusion, see below — but not to put tight bounds on any single rate.
- **The Share of Voice denominator is a hand-picked list** of eleven competitors, so it carries a
  hand-picked bias by construction.
- **The Arabic tier is overstated.** Arabic `بوتيكات` is also the ordinary plural noun "boutiques".
  Three of the nine Arabic Mentions are phrases like "local perfume boutiques" and do not refer to
  Boutiqaat at all. No metric computed from Mentions could reveal this; it surfaced only when a human
  read the text during Gold Set labelling. The reported Arabic rate of 18.8% should be read as an
  upper bound.
- **Citation Page evidence covers 57% of cited pages.** 70 of 162 could not be fetched: 28 hit the
  per-Answer cap, 27 returned HTTP 403, 6 timed out, 6 were PDFs, 2 were rate-limited, 1 had no
  extractable text. Blocked pages may be systematically different from readable ones — a site that
  blocks scrapers is more likely to be a large commercial retailer — so the 0-of-92 result is bounded
  by that.
- **The Gold Set labels were written after the Judge's output had been seen**, which contaminates the
  comparison. The 88.2% agreement figure is an upper bound and needs reproducing by a human labelling
  blind.

---

## The measurement design

The reasoning matters more than the feature list, so it is stated plainly.

**Mention is a fact; Recommendation is a judgement.** Whether a brand name appears in an Answer is
decided by Alias matching with no model involved, so the headline metric is reproducible and unit
testable. Whether the Answer actively puts Boutiqaat forward is decided by an LLM Judge, only for
Answers already known to contain a Mention. See `docs/adr/0002`.

**Anything countable is computed, never asked of the model.** Asked for Boutiqaat's Rank, the Judge
returned 9 while listing Boutiqaat 7th. Asked for brands "in first-mention order", it returned them
alphabetically, then returned a third different order on the next call. So the Judge supplies which
names are retailer brands and which of four labels fits the prose; the code computes where each name
appears, and therefore the ordering and the Rank. See `docs/adr/0005`.

**Absence is only counted against relevant questions.** Each Query carries a hand-written relevance
label. Three of the 24 are deliberately irrelevant — home gym equipment, a dinner recipe, car
insurance — and Trials on them are excluded from Visibility Rate entirely. Absence from a question
Boutiqaat could never answer is not a failure.

**Answers vary, so each Query is asked three times, and the variation is reported rather than
averaged away.** This is not a detail. At one Trial per Query the apparent finding was "the model
knows Boutiqaat but the web sources do not have it". At three Trials both modes mentioned it once in
three: Boutiqaat is *unreliably present in both*, which is a different commercial problem. Visibility
Rate is therefore computed over Trials, and Consistency — always / sometimes / never — is reported
beside it and never collapsed into it.

**Real answers, recorded, replayed.** Every Answer comes from a live call and is stored verbatim; the
on-disk cache is committed and is the fixture corpus the tests replay with no network. Measuring
against answers we wrote ourselves would be circular. See `docs/adr/0001`. **The `cache/` directory
is deliberately not gitignored** — deleting it breaks the test suite.

**A network error is never evidence.** A cited page that times out, 403s, or is a PDF is recorded as
`unfetched`, never as "Boutiqaat not present", and every claim resting on page evidence states both
counts.

**A cause with no evidence is dropped, not rendered.** Diagnosis candidates come from a fixed set and
each must be supported by something observed in the Run. On this corpus five causes are supported and
one — "mentioned but never recommended" — is correctly dropped, because some Answers do recommend
Boutiqaat.

---

## What it found

| Locale | Mentioned | Relevant Trials | Visibility Rate |
| --- | ---: | ---: | ---: |
| `global_en` | 0 | 36 | **0.0%** |
| `gcc_en` | 11 | 42 | 26.2% |
| `ar` | 9 | 48 | 18.8% (overstated, see above) |

| Provider mode | Mentioned | Relevant Trials | Visibility Rate |
| --- | ---: | ---: | ---: |
| ungrounded (model memory) | 12 | 63 | 19.0% |
| grounded (model + web search) | 8 | 63 | 12.7% |

Share of Voice 4.3% — 20 Mentions against Sephora's 104, Amazon's 65, Noon's 43.

Of the 92 cited pages that could be fetched for Answers where Boutiqaat is absent, **it appears on
none of them**.

Read together: the model recalls Boutiqaat more readily than the web surfaces it, and it is entirely
invisible in global English. The remedy is not "rank better" — it is to exist on the retailer
roundups, listicles and marketplace pages these answers are built from.

A generated sample report is in [REPORT.md](REPORT.md).

---

## Running it

Requires Python 3.11+ and an API key in `.env` as `OPENAI_API_KEY=...`.

```bash
python -m venv .venv && .venv/bin/python -m pip install -e '.[dev]'
export PATH=".venv/bin:$PATH"

python -m avi.cli run --dry-run                 # project the call count and cost, execute nothing
python -m avi.cli run --database corpus.db      # full corpus: 24 Queries x 3 Trials x 2 modes
python -m avi.cli report <run-id> --database corpus.db           # stakeholder Report
python -m avi.cli report <run-id> --database corpus.db --full    # audit Report, every Answer
python -m avi.cli serve --database corpus.db    # read-only API on :8000
streamlit run src/avi/ui.py                     # dashboard and Query drill-down
```

Runs are cache-first: re-running an identical Run makes no network call. A per-Run call budget aborts
on breach.

## Tests

```bash
python -m pytest -q          # 84 tests, no network
python -m mypy src
python -m pytest -q -s --gold-database corpus2.db -m gold_set    # opt-in, costs nothing but needs a corpus
```

The Gold Set suite is excluded from the default run because it grades a model rather than the
pipeline. Everything else replays recorded fixtures and makes no network call.

## Layout

`ingest` runs Queries · `detect` decides Mention by Alias matching · `judge` labels Recommendation
Strength · `citations` classifies and fetches cited pages · `metrics` computes the figures ·
`diagnose` builds evidence-backed findings · `report` renders · `api` serves · `ui` displays.

`CONTEXT.md` is the glossary every module name comes from. `docs/adr/` holds the five decisions that
would otherwise look arbitrary.
