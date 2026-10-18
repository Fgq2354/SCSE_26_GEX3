"""Check the three exports against recorded token sequences."""
import argparse
from pathlib import Path
from training import ROOT, OUTPUTS, digest, read_rows, prompt_for
import json


def verify(tokenizer_loader):
    questions = read_rows(ROOT / "data/sft_challenge_test.jsonl")
    for version in ("v1", "v2", "v3"):
        record = json.loads((ROOT / f"run_records/{version}_answers.json").read_text(encoding="utf-8"))
        training = json.loads((ROOT / f"run_records/{version}.json").read_text(encoding="utf-8"))
        domain = json.loads((ROOT / "run_records/domain.json").read_text(encoding="utf-8"))
        adapter = ROOT / "models" / OUTPUTS[version]
        assert digest(adapter / "adapter_model.safetensors") == record["adapter_sha256"]
        assert training["adapter_sha256"] == record["adapter_sha256"]
        assert training["base_sha256"] == record["base_sha256"] == domain["base_sha256"]
        assert training["domain_sha256"] == domain["adapter_sha256"]
        target = ROOT / f"SFT_{version.upper()}_Test.txt"
        assert digest(target) == record["export_sha256"], target
        assert digest(ROOT / "data/sft_challenge_test.jsonl") == record["questions_sha256"]
        assert len(record["answers"]) == len(questions) == 15
        tokenizer = tokenizer_loader(ROOT / "models" / OUTPUTS[version])
        for question, answer in zip(questions, record["answers"]):
            assert question["id"] == answer["id"]
            assert answer["input_prompt"] == prompt_for(question["prompt"])
            assert tokenizer(answer["input_prompt"])["input_ids"] == answer["input_ids"]
            decoded = tokenizer.decode(answer["generated_ids"], skip_special_tokens=True)
            assert decoded == answer["response"]
            whole = tokenizer.decode(answer["input_ids"] + answer["generated_ids"], skip_special_tokens=True)
            assert whole == answer["full_text"] and whole in target.read_text(encoding="utf-8")
        print(f"{version}: 15 answers match recorded model tokens")


if __name__ == "__main__":
    from transformers import AutoTokenizer
    verify(lambda path: AutoTokenizer.from_pretrained(path, local_files_only=True))
