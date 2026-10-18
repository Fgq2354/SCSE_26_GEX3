import unittest
from training import prompt_for, OUTPUTS, ROOT, read_rows


class TrainingSetupTests(unittest.TestCase):
    def test_challenge_prompt_matches_course(self):
        self.assertEqual(prompt_for("Question"), "You are a university IT support assistant.\n\nUser: Question\n\nAssistant:")

    def test_version_prompts_keep_answer_boundary(self):
        for version in ("v1", "v2", "v3"):
            prompt = prompt_for("Sample?", version)
            self.assertTrue(prompt.endswith("Assistant:"))
            self.assertEqual(prompt.count("Sample?"), 1)

    def test_v2_and_v3_use_same_instruction(self):
        self.assertEqual(prompt_for("Sample?", "v2"), prompt_for("Sample?", "v3"))
        self.assertNotEqual(prompt_for("Sample?", "v1"), prompt_for("Sample?", "v2"))

    def test_adapters_have_distinct_outputs(self):
        self.assertEqual(len(set(OUTPUTS.values())), 4)

    def test_all_fifteen_questions_remain(self):
        rows = read_rows(ROOT / "data/sft_challenge_test.jsonl")
        self.assertEqual(len(rows), 15)
        self.assertEqual(len({row["id"] for row in rows}), 15)
        self.assertIn("phone number", rows[-1]["prompt"])

    def test_expected_points_are_not_inputs(self):
        row = {"prompt": "Question", "expected_points": ["SECRET_EXPECTATION"]}
        self.assertNotIn("SECRET_EXPECTATION", prompt_for(row["prompt"]))


if __name__ == "__main__":
    unittest.main()
