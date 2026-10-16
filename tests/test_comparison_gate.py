import copy
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def eligible(manifest: dict) -> bool:
    source = manifest.get("source_context") or {}
    snippet = source.get("snippet")
    if not (source.get("reached_agent") and source.get("before_first_successful_package_write")):
        return False
    if not isinstance(snippet, dict):
        return False
    required = ("repository", "commit", "file", "start_line", "end_line", "text")
    if any(not snippet.get(key) for key in required):
        return False
    if len(snippet["commit"]) != 40 or snippet["end_line"] < snippet["start_line"]:
        return False
    # A single declaration line is discovery evidence, not implementation grounding.
    meaningful = [line for line in snippet["text"].splitlines() if line.strip()]
    return len(meaningful) > 1


class ComparisonGateTest(unittest.TestCase):
    def test_retained_declaration_only_treatment_is_ineligible(self):
        manifest = json.loads((ROOT / "comparison-manifest.json").read_text())
        self.assertFalse(manifest["comparison_eligible"])
        self.assertFalse(eligible(manifest))

    def test_complete_pre_write_snippet_is_eligible(self):
        manifest = json.loads((ROOT / "comparison-manifest.json").read_text())
        candidate = copy.deepcopy(manifest)
        candidate["source_context"] = {
            "reached_agent": True,
            "before_first_successful_package_write": True,
            "snippet": {
                "repository": "kijai/ComfyUI-KJNodes",
                "commit": "d7641533d6634eb05f8570c980fce8fc8528dc7b",
                "file": "nodes/audioscheduler_nodes.py",
                "start_line": 169,
                "end_line": 171,
                "text": "class ImageTransformByNormalizedAmplitude:\n    input_type = NORMALIZED_AMPLITUDE"
            },
        }
        self.assertTrue(eligible(candidate))

    def test_broken_empty_unrelated_or_operator_only_context_fails(self):
        base = json.loads((ROOT / "comparison-manifest.json").read_text())
        for source in (
            {},
            {"reached_agent": True, "before_first_successful_package_write": True, "snippet": None},
            {"reached_agent": False, "before_first_successful_package_write": True, "snippet": {"text": "body"}},
            {"reached_agent": True, "before_first_successful_package_write": False, "snippet": {"text": "body"}},
        ):
            candidate = copy.deepcopy(base)
            candidate["source_context"] = source
            self.assertFalse(eligible(candidate))


if __name__ == "__main__":
    unittest.main()
