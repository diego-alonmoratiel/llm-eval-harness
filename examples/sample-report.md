# Model comparison

15 cases, 3 models, 1 attempt(s) each, backend `hermes`, temperature 0.

| model | pass | rate | p50 | p95 | errors | cost |
|---|---|---|---|---|---|---|
| `glm-5.3-flash` | 15/15 | 100% | 6157 ms | 11027 ms | 0 | $0.0055 |
| `deepseek-v4-pro` | 15/15 | 100% | 7052 ms | 8056 ms | 0 | $0.0255 |
| `deepseek-v4.1-flash` | 13/15 | 87% | 6949 ms | 9188 ms | 0 | $0.0057 |

## Failures worth reading

- `deepseek-v4.1-flash` · triage-01 · got 'software', want 'hardware'
- `deepseek-v4.1-flash` · ground-02 · missing ['2']

## Limitations

- Rule-based grading only: correct for classification and extraction, blind to tone, nuance and anything requiring judgement.
- No cost column unless `prices.json` is filled in with verified prices.
- 15 cases is a smoke test, not a benchmark. A single attempt per case says nothing about variance.
