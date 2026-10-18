"""Check inputs and completion masks without loading model weights."""
from training import ROOT, arguments, base_identity, prepare_data, prompt_for, save_json, digest


if __name__ == "__main__":
    args = arguments().parse_args()
    from transformers import AutoTokenizer
    from trl.trainer.sft_trainer import DataCollatorForLanguageModeling
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, local_files_only=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    result = {"base_sha256": base_identity(args.base_model.resolve()), "stages": {}}
    collator = DataCollatorForLanguageModeling(
        pad_token_id=tokenizer.pad_token_id, completion_only_loss=True)
    for version in ("domain", "v1", "v2", "v3"):
        data = prepare_data(version, tokenizer)
        summary = {key: len(rows) for key, rows in data.items()}
        if version != "domain":
            for split in data.values():
                for row in split:
                    prefix = tokenizer(row["prompt"])["input_ids"]
                    full = tokenizer(row["prompt"] + row["completion"] + tokenizer.eos_token)["input_ids"][:256]
                    assert full[:len(prefix)] == prefix, "Tokenizer changed the prompt/completion boundary"
                    assert len(full) > len(prefix), "No completion tokens remain"
                    mask = [0] * len(prefix) + [1] * (len(full) - len(prefix))
                    labels = collator([{"input_ids": full, "completion_mask": mask}])["labels"][0].tolist()
                    assert all(value == -100 for value in labels[:len(prefix)])
                    assert labels[len(prefix):] == full[len(prefix):]
        result["stages"][version] = summary
        print(version, summary)
    result["data_sha256"] = {p.name: digest(p) for p in sorted((ROOT / "data").glob("*.jsonl"))}
    save_json(ROOT / "run_records/preparation.json", result)
    print("Input and completion-mask checks passed. No model weights were loaded.")
