# AI Search Visibility & Recommendation

Measures whether AI assistants surface Boutiqaat when a shopper asks a buying question, why they do or don't, and what would change the answer.

## Language

### The question and the answer

**Query**:
A natural-language buying question a shopper would put to an AI assistant, e.g. "Where can I buy Korean skincare?".
_Avoid_: Prompt, search term, keyword

**Answer**:
One AI assistant's complete text response to a single Query on a single run.
_Avoid_: Result, output, completion

**Query Set**:
The versioned collection of Queries a Run executes, each carrying its Intent, Locale, Specificity and Relevance label. Versioned because changing it makes Runs incomparable.
_Avoid_: Question list, test set, corpus

**Trial**:
One execution of one Query against one Provider. The same Query is run as several Trials because Answers vary between them; that variation is a finding, not noise to be averaged away.
_Avoid_: Attempt, sample, iteration

**Run**:
One complete execution of a Query set against a Provider set, at a fixed point in time. The durable unit everything else belongs to.
_Avoid_: Batch, job, session

**Provider**:
An AI system that produces Answers. Split into two kinds, which are not interchangeable evidence.
_Avoid_: Model, LLM, engine

**Grounded Provider**:
A Provider that retrieves live web content and returns Citations with its Answer. The only source of Citation evidence.
_Avoid_: Search model, RAG provider

**Ungrounded Provider**:
A Provider answering from model weights alone, with no Citations. Evidence of what the model *believes*, not of what the web says.
_Avoid_: Chat model, base model

**Citation**:
A source URL a Grounded Provider attributes its Answer to.
_Avoid_: Reference, link, source

### What we detect in an Answer

**Brand**:
A retailer that can appear in an Answer. Boutiqaat is one Brand among many; Competitors are Brands too.
_Avoid_: Company, merchant, vendor

**Mention**:
A Brand appearing in an Answer at all. A fact, decided by deterministic alias matching, not judgement.
_Avoid_: Hit, match, appearance

**Recommendation**:
An Answer actively putting a Brand forward as a place to buy. Strictly narrower than Mention: every Recommendation is a Mention, but a Brand can be Mentioned in passing, or dismissively, without being Recommended.
_Avoid_: Endorsement, suggestion

**Cited-not-named**:
A Grounded Answer that cites a URL on Boutiqaat's own domain while never naming Boutiqaat in its
text. Recorded and reported beside Visibility Rate, never added into it: the model read the site and
still did not name it, which is a different problem from never having found it.
_Avoid_: Silent citation, unnamed mention

**Absent**:
Boutiqaat not Mentioned in an Answer to a Relevant Query. Absence from an Irrelevant Query is not Absence and is never scored.
_Avoid_: Missing, not found

**Rank**:
Boutiqaat's ordinal position among the Brands Mentioned in one Answer, first Mention first.
_Avoid_: Position, placement, score

**Relevant**:
A property of a Query: whether Boutiqaat could legitimately be a good answer to it, judged before any Answer is examined. Gates every visibility metric.
_Avoid_: Applicable, in-scope, qualified

**Competitor**:
A Brand that competes with Boutiqaat for the same shopper on a Relevant Query. Split into two kinds that are counted separately and never merged mid-Run.
_Avoid_: Rival, alternative, peer

**Seed Competitor**:
A Competitor named in the curated list before a Run starts. Fixes the Share of Voice denominator so Runs stay comparable.
_Avoid_: Known competitor, tracked brand

**Emergent Brand**:
A Brand an Answer names that is not a Seed Competitor. Reported as its own finding — who the AI thinks the market is — never folded into the Share of Voice denominator.
_Avoid_: New competitor, discovered brand

**Alias**:
One surface form a Brand appears as, including misspellings, Arabic script, and domain names. Matching any Alias is a Mention.
_Avoid_: Variant, synonym, spelling

**Judge**:
The LLM step that assigns Recommendation Strength and Rank to an Answer. Distinct from Alias matching, which decides Mention without judgement.
_Avoid_: Classifier, evaluator, grader

### How Queries are organised

**Intent**:
What the shopper is trying to do — find a category, discover a retailer, compare options, or shop for an occasion.
_Avoid_: Query type, purpose

**Locale**:
The market and language a Query is posed in: global English, GCC English, or Arabic. Visibility differs sharply across these, so it is never collapsed.
_Avoid_: Region, market, geo

**Specificity**:
How narrow a Query is, from broad ("best beauty websites") to narrow ("where to buy Korean skincare in Kuwait").
_Avoid_: Granularity, breadth

### Diagnosis

**Diagnosis**:
An evidence-backed account of why Boutiqaat is Absent or weakly Recommended, and what would change it. Every claim carries the Answers and Citations it rests on; a claim with no supporting Evidence is dropped, never rendered.
_Avoid_: Analysis, explanation, insight

**Evidence**:
Something actually observed in a Run — an Answer, a Citation, a Citation Page — that supports or refutes a Diagnosis claim. Plausible reasoning is not Evidence.
_Avoid_: Rationale, support, justification

**Citation Page**:
The fetched content of a Citation, checked for whether Boutiqaat appears on it. Distinguishes "the AI ignores Boutiqaat" from "the sources the AI reads do not contain Boutiqaat".
_Avoid_: Source page, landing page

**Source Type**:
What kind of page a Citation is: retailer site, editorial listicle, marketplace, or review site. The mix reveals which content surfaces AI answers.
_Avoid_: Domain type, category

### What we report

**Visibility**:
The umbrella outcome: whether and how favourably Boutiqaat appears in Answers. Never used alone as a number — always qualified by the specific metric meant.
_Avoid_: Presence, exposure, ranking (as a bare noun)

**Visibility Rate**:
The share of Trials on Relevant Queries whose Answers Mention Boutiqaat. Computed over Trials, not Queries, so a Brand appearing in one Trial of three counts as a third rather than being rounded to visible or invisible.
_Avoid_: Coverage, hit rate

**Share of Voice**:
Boutiqaat's Mentions as a proportion of all Brand Mentions across a set of Answers.
_Avoid_: Market share, SOV

**Recommendation Strength**:
Which of four ordered levels an Answer assigns a Brand: `recommended` (put forward as a place to buy), `listed` (named in a list without endorsement), `passing` (mentioned incidentally), `dismissed` (named negatively or ruled out). Reported as a distribution; never averaged, since the levels are ordinal and an average would invent precision the labels do not carry.
_Avoid_: Sentiment, tone, score

**Consistency**:
How reliably a Query produces a Mention across its Trials: always, sometimes, or never. A Query that is sometimes visible is a different commercial problem from one that never is, so the two are never collapsed into one rate.
_Avoid_: Stability, variance, reliability

**Report**:
The stakeholder-facing document generated from a stored Run. Every claim in it names the Answers it came from, so it is traceable rather than asserted.
_Avoid_: Summary, deliverable, output

**Gold Set**:
Hand-labelled Answers used to measure how far the LLM judge's labels diverge from a human's.
_Avoid_: Test set, ground truth, validation set
