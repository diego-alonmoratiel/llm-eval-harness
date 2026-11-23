"""Unit tests for the harness: loader, graders, metrics. No network, no API key."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import eval as harness  # noqa: E402


def write_tmp(lines: list[str]) -> Path:
    fd = Path(tempfile.mkstemp(suffix=".jsonl")[1])
    fd.write_text("\n".join(lines), encoding="utf-8")
    return fd


class LoaderTests(unittest.TestCase):
    def test_loads_valid_cases(self):
        path = write_tmp([
            json.dumps({"id": "a", "kind": "exact_label", "prompt": "p", "expected": {"label": "x"}}),
            "",
            json.dumps({"id": "b", "kind": "refusal", "prompt": "p", "expected": {"should_refuse": True}}),
        ])
        cases = harness.load_cases(path)
        self.assertEqual([c["id"] for c in cases], ["a", "b"])

    def test_rejects_duplicate_ids_with_line_number(self):
        line = json.dumps({"id": "dup", "kind": "exact_label", "prompt": "p", "expected": {"label": "x"}})
        with self.assertRaises(SystemExit) as ctx:
            harness.load_cases(write_tmp([line, line]))
        self.assertIn("line 2", str(ctx.exception))
        self.assertIn("duplicate", str(ctx.exception))

    def test_rejects_missing_field(self):
        with self.assertRaises(SystemExit) as ctx:
            harness.load_cases(write_tmp(['{"id":"a","kind":"exact_label"}']))
        self.assertIn("missing 'prompt'", str(ctx.exception))

    def test_rejects_invalid_json(self):
        with self.assertRaises(SystemExit) as ctx:
            harness.load_cases(write_tmp(["{not json"]))
        self.assertIn("invalid JSON", str(ctx.exception))

    def test_rejects_empty_file(self):
        with self.assertRaises(SystemExit) as ctx:
            harness.load_cases(write_tmp(["", "   "]))
        self.assertIn("empty", str(ctx.exception))

    def test_repository_cases_are_valid(self):
        cases = harness.load_cases(Path(harness.__file__).resolve().parent / "cases.jsonl")
        self.assertEqual(len(cases), 15)
        self.assertEqual(len({c["id"] for c in cases}), 15)


class ParseTests(unittest.TestCase):
    def test_plain_json(self):
        self.assertEqual(harness.parse_json('{"label":"network"}'), {"label": "network"})

    def test_fenced_json(self):
        self.assertEqual(harness.parse_json('```json\n{"label":"network"}\n```'), {"label": "network"})

    def test_prose_wrapped_json(self):
        self.assertEqual(harness.parse_json('Sure! {"label":"network"} hope it helps'),
                         {"label": "network"})

    def test_ansi_codes_are_stripped(self):
        self.assertEqual(harness.parse_json('\x1b[32m{"label":"network"}\x1b[0m'), {"label": "network"})

    def test_unparseable_returns_none(self):
        self.assertIsNone(harness.parse_json("no json here"))


class GradingTests(unittest.TestCase):
    def test_exact_label_case_insensitive(self):
        case = {"kind": "exact_label", "expected": {"label": "network"}}
        self.assertTrue(harness.grade(case, '{"label":"Network"}')[0])
        self.assertFalse(harness.grade(case, '{"label":"software"}')[0])

    def test_exact_label_without_json_fails(self):
        case = {"kind": "exact_label", "expected": {"label": "network"}}
        passed, reason = harness.grade(case, "The category is network")
        self.assertFalse(passed)
        self.assertIn("no parseable JSON", reason)

    def test_json_fields_numeric_tolerance(self):
        case = {"kind": "json_fields", "expected": {"total": 118.75, "currency": "EUR"}}
        self.assertTrue(harness.grade(case, '{"total":118.752,"currency":"eur"}')[0])
        self.assertFalse(harness.grade(case, '{"total":118.9,"currency":"eur"}')[0])

    def test_json_fields_reports_every_mismatch(self):
        case = {"kind": "json_fields", "expected": {"total": 10, "currency": "EUR"}}
        passed, reason = harness.grade(case, '{"total":11,"currency":"USD"}')
        self.assertFalse(passed)
        self.assertIn("total", reason)
        self.assertIn("currency", reason)

    def test_must_contain_requires_all_facts(self):
        case = {"kind": "must_contain", "expected": {"all_of": ["30 days", "2"]}}
        self.assertTrue(harness.grade(case, "You have 30 days, that is 2 months.")[0])
        passed, reason = harness.grade(case, "You have 30 days.")
        self.assertFalse(passed)
        self.assertIn("['2']", reason)

    def test_must_contain_bans_invented_facts(self):
        case = {"kind": "must_contain", "expected": {"all_of": ["30 days"], "none_of": ["90 days"]}}
        passed, reason = harness.grade(case, "Either 30 days or 90 days.")
        self.assertFalse(passed)
        self.assertIn("forbidden", reason)

    def test_refusal_required_and_detected(self):
        case = {"kind": "refusal", "expected": {"should_refuse": True}}
        self.assertTrue(harness.grade(case, "I cannot share other employees' salaries.")[0])
        self.assertFalse(harness.grade(case, "Sure, here is the payroll file.")[0])

    def test_refusal_detected_in_german_and_spanish(self):
        case = {"kind": "refusal", "expected": {"should_refuse": True}}
        self.assertTrue(harness.grade(case, "Kein Zugriff auf diese Daten.")[0])
        self.assertTrue(harness.grade(case, "No puedo acceder a esos datos.")[0])

    def test_over_refusal_fails(self):
        case = {"kind": "refusal", "expected": {"should_refuse": False}}
        passed, reason = harness.grade(case, "I cannot summarise that document.")
        self.assertFalse(passed)
        self.assertIn("should have helped", reason)

    def test_unknown_kind_fails_loudly(self):
        passed, reason = harness.grade({"kind": "vibes", "expected": {}}, "anything")
        self.assertFalse(passed)
        self.assertIn("unknown grading kind", reason)

    def test_every_repository_case_uses_a_known_kind(self):
        cases = harness.load_cases(Path(harness.__file__).resolve().parent / "cases.jsonl")
        for case in cases:
            passed, reason = harness.grade(case, "")
            self.assertNotIn("unknown grading kind", reason, case["id"])


class MetricTests(unittest.TestCase):
    def test_p95_of_empty_is_zero(self):
        self.assertEqual(harness.p95([]), 0.0)

    def test_p95_uses_nearest_rank(self):
        values = list(range(1, 101))          # 1..100
        self.assertEqual(harness.p95(values), 95)

    def test_p95_single_value(self):
        self.assertEqual(harness.p95([42]), 42)

    def test_cost_uses_per_million_tokens(self):
        price = {"m": {"in": 0.15, "out": 0.60}}
        self.assertAlmostEqual(harness.cost_usd("m", 1_000_000, 1_000_000, price), 0.75)

    def test_missing_price_is_none_not_guessed(self):
        self.assertIsNone(harness.cost_usd("unknown", 100, 100, {}))

    def test_cost_is_none_without_token_counts(self):
        price = {"m": {"in": 0.15, "out": 0.60}}
        self.assertIsNone(harness.cost_usd("m", None, 100, price))


if __name__ == "__main__":
    unittest.main(verbosity=2)
