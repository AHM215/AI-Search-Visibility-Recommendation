# Five-minute recording script

Methodology first: the brief weights reasoning above code, and code above the UI. A feature tour
answers the lowest-weighted question. Rehearse once; unrehearsed five-minute recordings run eleven.

## 0:00-1:30 — The measurement design and why

- The question: when a shopper asks an AI assistant where to buy, does Boutiqaat appear, and if not,
  why. Show `CONTEXT.md` briefly: the vocabulary is fixed before any code.
- **Mention is a fact, Recommendation is a judgement.** Alias matching decides presence with no model
  call, so the headline number is reproducible and testable. The Judge only ever sees Answers that
  already contain a Mention. (ADR-0002)
- **Anything countable is computed, not asked.** Show the three live Judge failures: rank 9 while
  listing Boutiqaat 7th; alphabetical order when asked for mention order; a third order on the next
  call. Hence ADR-0005 — the model labels, the code counts.
- **Relevance gate**: three of 24 Queries are deliberately irrelevant, and absence from those is never
  scored.
- **N=3**, and say why in one sentence: at N=1 the finding was "the model knows Boutiqaat, the web
  doesn't"; at N=3 both modes mention it once in three. Repetition changed the conclusion.

## 1:30-3:30 — Live, into one absent Query and its evidence

- `python -m avi.cli run --dry-run` — show the projected 284 calls and $2.84. Cost-awareness is a
  20-line feature; show it.
- Open the UI drill-down on an absent grounded Query. Show the raw Answer text, competitors
  highlighted, Boutiqaat not there, and the Citations listed with their page status.
- Click through to a cited page. Then the number: **0 of 92 fetched cited pages mention Boutiqaat.**
- Immediately give the bound yourself: 70 pages could not be fetched, 27 of them HTTP 403, so the
  evidence covers 57% of cited pages, and blocked pages may be systematically different. Say this
  before anyone asks.

## 3:30-4:30 — The diagnosis and what would change

- Show the Diagnosis section of `REPORT.md`. Five causes supported, each carrying its Answer ids.
- Point at the one that was **dropped** — "mentioned but never recommended" — because the evidence
  refuted it. That is the proof the Diagnosis is not an LLM narrating plausible causes.
- The Locale table: 0 of 36 in global English, 26.2% in GCC English. Visibility is confined to the
  markets Boutiqaat already serves.
- The remedy, in one sentence: this is a source-coverage problem, not a ranking problem. Boutiqaat has
  to exist on the roundups and listicles these answers are built from.

## 4:30-5:00 — What was cut, and what is wrong with it

- One vendor, not three: one credential, three days, and three shallow integrations help nobody.
- API access is a proxy for consumer ChatGPT — no memory, no personalization.
- **The Arabic tier is overstated**: `بوتيكات` is also the ordinary word for "boutiques", and three of
  nine Arabic Mentions are not the brand. No metric could have caught this; a human reading the text
  during Gold Set labelling did.
- Gold Set agreement is 88.2%, and the labels were written after seeing the Judge's output, so treat
  it as an upper bound.

Ending on the flaws you found in your own work is stronger than ending on the features.
