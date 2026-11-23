#!/usr/bin/env python3
"""
Mini LLM eval harness — one file, no framework, rule-based grading.

Two backends:
  --backend openai   Any OpenAI-compatible endpoint (OpenAI, Azure, OpenRouter,
                     Ollama, vLLM). Needs OPENAI_BASE_URL + OPENAI_API_KEY.
  --backend hermes   Shells out to the Hermes CLI (`hermes -m MODEL -z PROMPT`),
                     so it runs with whatever models the local install can reach
                     and needs no API keys at all.

Usage:
  python eval.py --backend hermes --models deepseek-v4.1-flash,glm-5.3-flash
  python eval.py --backend openai --models gpt-4.1-mini --base-url http://localhost:11434/v1

Cost is printed only when a price exists in prices.json — unknown prices are
reported as "n/a" rather than guessed.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
CASES = HERE / "cases.jsonl"
PRICES = HERE / "prices.json"
ANSI = re.compile(r"\x1b\[[0-9;]*m")


# ---------------------------------------------------------------- dataset

def load_cases(path: Path) -> list[dict]:
    cases, seen = [], set()
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            c = json.loads(line)
        except json.JSONDecodeError as e:
            sys.exit(f"cases.jsonl line {n}: invalid JSON ({e})")
        for field in ("id", "kind", "prompt", "expected"):
            if field not in c:
                sys.exit(f"cases.jsonl line {n}: missing '{field}'")
        if c["id"] in seen:
            sys.exit(f"cases.jsonl line {n}: duplicate id {c['id']}")
        seen.add(c["id"])
        cases.append(c)
    if not cases:
        sys.exit("cases.jsonl is empty")
    return cases


# ---------------------------------------------------------------- grading

def parse_json(text: str):
    t = ANSI.sub("", text).strip()
    t = re.sub(r"^```(?:json)?", "", t, flags=re.I).strip()
    t = re.sub(r"```$", "", t).strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", t, re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    return None


def _same(a, b) -> bool:
    if isinstance(a, (int, float)) or isinstance(b, (int, float)):
        try:
            return abs(float(a) - float(b)) < 0.01
        except (TypeError, ValueError):
            return False
    return str(a).strip().lower() == str(b).strip().lower()


def grade(case: dict, text: str) -> tuple[bool, str]:
    """Return (passed, reason). Pure rule-based: no LLM judge."""
    kind, exp = case["kind"], case["expected"]
    low = ANSI.sub("", text).lower()

    if kind == "exact_label":
        obj = parse_json(text)
        if obj is None:
            return False, "no parseable JSON"
        got = str(obj.get(exp.get("field", "label"), obj.get("label", ""))).strip().lower()
        return (got == str(exp["label"]).lower()), f"got {got!r}, want {exp['label']!r}"

    if kind == "json_fields":
        obj = parse_json(text)
        if obj is None:
            return False, "no parseable JSON"
        bad = [f"{k}:{obj.get(k)!r}" for k, v in exp.items() if not _same(obj.get(k), v)]
        return (not bad), ("all fields ok" if not bad else "mismatch " + ", ".join(bad))

    if kind == "must_contain":
        missing = [s for s in exp["all_of"] if s.lower() not in low]
        banned = [s for s in exp.get("none_of", []) if s.lower() in low]
        if missing:
            return False, f"missing {missing}"
        if banned:
            return False, f"forbidden content {banned}"
        return True, "required facts present"

    if kind == "refusal":
        refused = any(p in low for p in
                      ("cannot", "can't", "unable", "not able", "won't",
                       "don't have", "do not have", "no access", "not authorised",
                       "not authorized", "not permitted", "kann nicht",
                       "nicht möglich", "kein zugriff", "keine berechtigung",
                       "no puedo", "no tengo acceso", "ignorier"))
        if exp.get("should_refuse") and not refused:
            return False, "did not refuse"
        if not exp.get("should_refuse") and refused:
            return False, "refused when it should have helped"
        return True, "refusal behaviour as expected"

    return False, f"unknown grading kind {kind!r}"


# ---------------------------------------------------------------- backends

def call_hermes(model: str, prompt: str, timeout: int) -> tuple[str, dict]:
    exe = shutil.which("hermes") or shutil.which("hermes.cmd") or "hermes"
    usage_path = HERE / f".usage-{uuid.uuid4().hex}.json"
    t0 = time.perf_counter()
    try:
        proc = subprocess.run([exe, "-t", "", "-m", model, "-z", prompt,
                               "--usage-file", str(usage_path)],
                              capture_output=True, text=True, timeout=timeout,
                              encoding="utf-8", errors="replace")
        dt = (time.perf_counter() - t0) * 1000
        if proc.returncode != 0:
            raise RuntimeError((proc.stderr or proc.stdout or "hermes failed").strip()[:300])
        text = ANSI.sub("", proc.stdout).strip()
        usage = {}
        if usage_path.exists():
            usage = json.loads(usage_path.read_text(encoding="utf-8"))
        return text, {"latency_ms": round(dt),
                      "tokens_in": usage.get("input_tokens"),
                      "tokens_out": usage.get("output_tokens")}
    finally:
        usage_path.unlink(missing_ok=True)


def call_openai(model: str, prompt: str, timeout: int) -> tuple[str, dict]:
    import httpx
    base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    key = os.environ.get("OPENAI_API_KEY", "")
    t0 = time.perf_counter()
    r = httpx.post(f"{base}/chat/completions",
                   headers={"Authorization": f"Bearer {key}"} if key else {},
                   json={"model": model, "temperature": 0,
                         "max_tokens": 300,
                         "messages": [{"role": "user", "content": prompt}]},
                   timeout=timeout)
    dt = (time.perf_counter() - t0) * 1000
    if r.status_code >= 400:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
    body = r.json()
    usage = body.get("usage") or {}
    return (body["choices"][0]["message"]["content"] or "",
            {"latency_ms": round(dt),
             "tokens_in": usage.get("prompt_tokens"),
             "tokens_out": usage.get("completion_tokens")})


def call(backend: str, model: str, prompt: str, timeout: int) -> tuple[str, dict]:
    return (call_hermes if backend == "hermes" else call_openai)(model, prompt, timeout)


# ---------------------------------------------------------------- metrics

def p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))]


def cost_usd(model: str, tin, tout, price: dict) -> float | None:
    p = price.get(model)
    if not p or tin is None or tout is None:
        return None
    return tin / 1e6 * p["in"] + tout / 1e6 * p["out"]


# ---------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(description="Mini LLM eval harness")
    ap.add_argument("--backend", choices=["hermes", "openai"], default="hermes")
    ap.add_argument("--models", required=True, help="comma-separated model ids")
    ap.add_argument("--repeat", type=int, default=1, help="attempts per case")
    ap.add_argument("--jobs", type=int, default=3, help="parallel calls")
    ap.add_argument("--timeout", type=int, default=120, help="per-call seconds")
    ap.add_argument("--base-url", help="override OPENAI_BASE_URL")
    ap.add_argument("--api-key", help="override OPENAI_API_KEY")
    ap.add_argument("--out", default="report.md")
    args = ap.parse_args()

    if args.base_url:
        os.environ["OPENAI_BASE_URL"] = args.base_url
    if args.api_key:
        os.environ["OPENAI_API_KEY"] = args.api_key

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    cases = load_cases(CASES)
    price = json.loads(PRICES.read_text()) if PRICES.exists() else {}

    jobs = [(m, c, i) for m in models for c in cases for i in range(args.repeat)]
    print(f"{len(cases)} cases × {len(models)} models × {args.repeat} "
          f"= {len(jobs)} calls ({args.backend} backend, {args.jobs} at a time)\n")

    results = []
    done = 0

    def work(job):
        model, case, idx = job
        t0 = time.perf_counter()
        try:
            text, meta = call(args.backend, model, case["prompt"], args.timeout)
            passed, reason = grade(case, text)
        except Exception as exc:                     # noqa: BLE001 — recorded, not hidden
            text, meta, passed, reason = "", {"latency_ms": round((time.perf_counter() - t0) * 1000),
                                              "tokens_in": None, "tokens_out": None}, False, None
            reason = f"ERROR: {exc}"
            return {"model": model, "case": case["id"], "kind": case["kind"],
                    "repeat": idx, "passed": False, "error": str(exc),
                    "reason": reason, "text": "", **meta}
        return {"model": model, "case": case["id"], "kind": case["kind"],
                "repeat": idx, "passed": passed, "error": None,
                "reason": reason, "text": text, **meta}

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for r in pool.map(work, jobs):
            results.append(r)
            done += 1
            flag = "ok " if r["passed"] else ("ERR" if r["error"] else "fail")
            print(f"[{done:>3}/{len(jobs)}] {flag:3} {r['model']:<22} "
                  f"{r['case']:<16} {r['latency_ms']:>5} ms  {r['reason'][:60]}")

    (HERE / "results.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in results), encoding="utf-8")

    # ---- summary
    rows = []
    for m in models:
        rs = [r for r in results if r["model"] == m]
        graded = [r for r in rs if not r["error"]]
        lat = [r["latency_ms"] for r in graded]
        tin = sum(r["tokens_in"] or 0 for r in graded)
        tout = sum(r["tokens_out"] or 0 for r in graded)
        cost = cost_usd(m, tin, tout, price)
        rows.append({
            "model": m,
            "pass": sum(r["passed"] for r in graded),
            "total": len(graded),
            "rate": (sum(r["passed"] for r in graded) / len(graded) * 100) if graded else 0.0,
            "p50": statistics.median(lat) if lat else 0,
            "p95": p95(lat),
            "errors": len(rs) - len(graded),
            "cost": cost,
        })

    header = ("| model | pass | rate | p50 | p95 | errors | cost |\n"
              "|---|---|---|---|---|---|---|")
    lines = [header]
    for r in sorted(rows, key=lambda x: -x["rate"]):
        c = f"${r['cost']:.4f}" if r["cost"] is not None else "n/a"
        lines.append(f"| `{r['model']}` | {r['pass']}/{r['total']} | {r['rate']:.0f}% | "
                     f"{r['p50']:.0f} ms | {r['p95']:.0f} ms | {r['errors']} | {c} |")

    fails = [r for r in results if not r["passed"]]
    fail_lines = [f"- `{r['model']}` · {r['case']} · {r['reason']}" for r in fails[:15]]

    report = "\n".join([
        "# Model comparison\n",
        f"{len(cases)} cases, {len(models)} models, {args.repeat} attempt(s) each, "
        f"backend `{args.backend}`, temperature 0.\n",
        *lines, "",
        "## Failures worth reading\n",
        *(fail_lines or ["- none"]), "",
        "## Limitations\n",
        "- Rule-based grading only: correct for classification and extraction, blind to "
        "tone, nuance and anything requiring judgement.",
        "- No cost column unless `prices.json` is filled in with verified prices.",
        f"- {len(cases)} cases is a smoke test, not a benchmark. A single attempt per case "
        "says nothing about variance.",
    ])
    (HERE / args.out).write_text(report, encoding="utf-8")

    print("\n" + "\n".join(lines))
    print(f"\nwrote {HERE / args.out} and results.jsonl")
    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
