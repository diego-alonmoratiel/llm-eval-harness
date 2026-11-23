# LLM Eval Harness

[![CI](https://github.com/diego-alonmoratiel/llm-eval-harness/actions/workflows/ci.yml/badge.svg)](https://github.com/diego-alonmoratiel/llm-eval-harness/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![Runtime deps](https://img.shields.io/badge/runtime%20deps-httpx-lightgrey)](requirements.txt)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

A single-file evaluation harness that answers one question with evidence:
**which model is good enough for this task, at what latency, and at what price.**

Most "model comparison" is a chat window and a feeling. This does the boring part
properly: fixed criteria, comparable numbers, and a hard line between *the model was
wrong* and *the harness broke*. It runs against any OpenAI-compatible endpoint
(OpenAI, Azure, OpenRouter, Ollama, vLLM) or against a local CLI gateway, so the same
15 cases can be replayed across hosted and self-hosted models.

```text
$ python eval.py --models glm-5.3-flash,deepseek-v4-pro,deepseek-v4.1-flash

15 cases × 3 models × 1 = 45 calls (hermes backend, 3 at a time)

| model | pass | rate | p50 | p95 | errors | cost |
|---|---|---|---|---|---|---|
| `glm-5.3-flash`       | 15/15 | 100% | 6157 ms | 11027 ms | 0 | $0.0055 |
| `deepseek-v4-pro`     | 15/15 | 100% | 7052 ms |  8056 ms | 0 | $0.0255 |
| `deepseek-v4.1-flash` | 13/15 |  87% | 6949 ms |  9188 ms | 0 | $0.0057 |

## Failures worth reading

- `deepseek-v4.1-flash` · triage-01 · got 'software', want 'hardware'
- `deepseek-v4.1-flash` · ground-02 · missing ['2']
```

The headline of that run is the useful part: a flash-tier model at ~5× lower cost matches
the pro tier on 13 of 15 cases, and the two cases it loses are exactly the ones needing
disambiguation and arithmetic. That is a decision, not a vibe. Full output in
[`examples/sample-report.md`](examples/sample-report.md).

## Quick start

```bash
git clone https://github.com/diego-alonmoratiel/llm-eval-harness.git
cd llm-eval-harness
pip install -r requirements.txt        # httpx only; the CLI backend needs nothing
```

Pick a backend:

```bash
# 1. Any OpenAI-compatible endpoint
export OPENAI_BASE_URL=https://openrouter.ai/api/v1
export OPENAI_API_KEY=sk-...
python eval.py --backend openai --models openai/gpt-4.1-mini,anthropic/claude-haiku-4.5

# 2. Local model, zero marginal cost
python eval.py --backend openai --models qwen2.5:3b-instruct \
               --base-url http://localhost:11434/v1

# 3. No API keys at all: shell out to a local agent CLI as the gateway
python eval.py --backend hermes --models deepseek-v4.1-flash,glm-5.3-flash

# Variance instead of a single sample
python eval.py --models X --repeat 3 --jobs 8
```

Every run prints one line per call and writes `report.md` (markdown table, sorted by
pass rate, with the failing cases spelled out) plus `results.jsonl` (one row per call
including the raw model output, so any number in the report can be re-derived or
audited later).

| Flag | Default | Purpose |
|---|---|---|
| `--backend` | `hermes` | `openai` (HTTP) or `hermes` (local CLI) |
| `--models` | required | comma-separated model ids |
| `--repeat` | `1` | attempts per case — the only way to see variance |
| `--jobs` | `3` | parallel calls in flight |
| `--timeout` | `120` | per-call seconds |
| `--base-url` / `--api-key` | env | per-run override of `OPENAI_*` |
| `--out` | `report.md` | report path |

Exit code is `0` only when nothing failed, so it drops straight into CI.

## How it works

```
cases.jsonl ──► dispatcher ──► backend ──► grader ──► metrics ──► report.md
 (fixed set)   (models ×       (openai |   (rule-     (pass, p50,   + results.jsonl
                cases ×        hermes)      based)     p95, cost)
                repeats)
```

Four properties matter more than the code:

1. **Criteria are fixed before any answer is seen.** Cases and expected values live in a
   data file, versioned with the harness. Adding a case after watching a model fail is
   how benchmarks get gamed.
2. **Quality, latency and cost come out of one table.** A model that is 3% more accurate
   for 5× the price is a different answer from one that is 3% better and cheaper.
3. **Infrastructure errors are never scored as model errors.** A 429, a timeout or a dead
   endpoint lands in `errors` and is excluded from the pass rate; otherwise a flaky
   afternoon reads as a dumb model.
4. **No number is invented.** Missing price means `n/a`, not a guess — and a wrong price
   makes every cost conclusion wrong, silently.

Grading is rule-based on purpose: expected labels, JSON field values (numeric tolerance
0.01, case-insensitive strings), required and forbidden substrings, and explicit refusal
behaviour. An LLM judge would add its own error, its own cost and its own variance to a
measurement whose whole point is reproducibility.

## The 15 cases

| Kind | Count | What it tests |
|---|---|---|
| `exact_label` | 5 | Ticket triage into a fixed category set, JSON-only output, English + German |
| `json_fields` | 4 | Structured extraction from invoices and emails, numeric tolerance |
| `must_contain` | 3 | Answering strictly from supplied text — required facts present, invented facts absent |
| `refusal` | 3 | Must refuse an HR data request; must **not** refuse a benign summary |

The refusal group is the interesting one: over-refusal is as much a failure as
under-refusal, so each case declares which side of the line the answer belongs on.

## Adding cases

One JSON object per line in `cases.jsonl`:

```json
{"id":"triage-06","kind":"exact_label","prompt":"...","expected":{"label":"network"}}
{"id":"invoice-05","kind":"json_fields","prompt":"...","expected":{"total":1180.0,"currency":"EUR"}}
{"id":"ground-04","kind":"must_contain","prompt":"...","expected":{"all_of":["30 days"],"none_of":["90 days"]}}
{"id":"refuse-04","kind":"refusal","prompt":"...","expected":{"should_refuse":false}}
```

The loader fails loudly — with the line number — on invalid JSON, a missing field, a
duplicate id, or an empty file, because a silent skip inside a benchmark is a wrong
number in the report.

## Adding prices

`prices.json`, USD per 1M tokens, numbers copied from the vendor page:

```json
{
  "_source": "https://vendor.example/pricing",
  "_verified_on": "2026-11-08",
  "model-id": {"in": 0.15, "out": 0.60}
}
```

Unknown model ⇒ `n/a`. The `_source` and `_verified_on` keys exist so a reviewer can
check the prices instead of trusting them.

## Tests and CI

```bash
python -m unittest discover -s tests -v
```

The suite covers the loader's failure modes, all four graders (including fence-wrapped
and prose-wrapped JSON), numeric tolerance, and the percentile and cost maths. No network
and no API key required. GitHub Actions runs it on Python 3.11, 3.12 and 3.13 on every
push and pull request.

## Limitations

Say these out loud before anyone asks:

- **Rule-based grading is narrow.** Right for classification and extraction, blind to
  tone, nuance, and anything that needs judgement.
- **15 cases is a smoke test, not a benchmark.** A real evaluation is 60+ cases, several
  repeats, and a written rubric for the subjective half.
- **One attempt per case says nothing about variance.** Use `--repeat 3`; a 100% pass rate
  on a single attempt is a point estimate, not a guarantee.
- **Latency is not comparable across backends.** The `openai` backend times the HTTP call;
  the `hermes` backend times a CLI process and includes startup — hence the ~6 s p50 in
  the table above, which is process overhead, not model speed.
- **Refusal detection is phrasing-based.** Only English, German and Spanish refusal
  patterns are recognised; an answer refusing in another language scores as a non-refusal.

## What this demonstrates

A model answering a question is not an evaluation. This harness fixes the criteria before
it sees the answers, reports quality, cost and latency in one comparable table, separates
broken infrastructure from wrong answers, and refuses to print a number it cannot justify.

## License

MIT — see [LICENSE](LICENSE).
