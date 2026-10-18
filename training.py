"""Training and answer export for the supplied course datasets."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "2"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_OFFLINE"] = "1"

ROOT = Path(__file__).resolve().parent
SEED = 48391
OUTPUTS = {"domain": "domain_adapter", "v1": "specialized_adapter",
           "v2": "specialized_adapter_v2", "v3": "specialized_adapter_v3"}
BASIC = "You are a university IT support assistant."
FOCUSED = (
    "You are a university IT support assistant. "
    "Answer only the user's question using supported university IT information. "
    "If the requested fact is not available, say that it is not specified rather than inventing it. "
    "Keep the answer concise."
)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def prompt_for(question, version="v1"):
    instruction = BASIC if version == "v1" else FOCUSED
    return instruction + "\n\nUser: " + question + "\n\nAssistant:"


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def hardware(startup=False):
    import psutil
    free_ram = psutil.virtual_memory().available / 2**30
    result = subprocess.run(
        ["nvidia-smi", "--id=0", "--query-gpu=temperature.gpu,memory.free",
         "--format=csv,noheader,nounits"], capture_output=True, text=True,
        check=True, timeout=10)
    temperature, free_vram = map(float, result.stdout.strip().split(","))
    if free_ram < (4.5 if startup else 1.25) or temperature >= 80:
        raise RuntimeError(f"Resource limit: RAM {free_ram:.2f} GiB, GPU {temperature} C")
    if startup and (free_vram < 4500 or temperature >= 75):
        raise RuntimeError(f"GPU not ready: {free_vram} MiB free, {temperature} C")
    return temperature


def cool_down():
    temperature = hardware()
    started = time.monotonic()
    if temperature >= 72:
        while temperature > 65:
            if time.monotonic() - started > 600:
                raise RuntimeError("GPU did not cool within ten minutes")
            time.sleep(3)
            temperature = hardware()


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=Path, default=ROOT / "models/qwen3-0.6b")
    return parser


def runtime(seed=SEED):
    import torch
    from transformers import set_seed
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("This configuration requires a CUDA GPU with bfloat16 support")
    set_seed(seed)
    return torch


def prepare_data(stage, tokenizer):
    from datasets import DatasetDict, load_dataset
    if stage in ("domain", "v1"):
        name = "domain_corpus_clean.jsonl" if stage == "domain" else "sft_clean.jsonl"
        data = load_dataset("json", data_files=str(ROOT / "data" / name), split="train")
        split = data.train_test_split(test_size=0.1, seed=SEED)
        data = DatasetDict(train=split["train"], validation=split["test"])
    else:
        data = load_dataset("json", data_files={
            part: str(ROOT / "data" / f"sft_{stage}_{part}.jsonl")
            for part in ("train", "validation")})
    columns = data["train"].column_names
    if stage == "domain":
        return data.map(lambda row: tokenizer(row["text"], truncation=True, max_length=256),
                        remove_columns=columns)
    # Separate prompt and completion fields let TRL mask question tokens correctly.
    return data.map(lambda row: {
        "prompt": prompt_for(row["prompt"], stage),
        "completion": " " + row["completion"],
    }, remove_columns=columns)


def base_identity(base):
    config = json.loads((base / "config.json").read_text(encoding="utf-8"))
    if config.get("model_type") != "qwen3" or config.get("hidden_size") != 1024:
        raise ValueError("Expected the original Qwen3-0.6B model")
    weights = list(base.glob("*.safetensors"))
    if not weights or (base / "adapter_config.json").exists():
        raise ValueError("Use original base weights, not another trained adapter")
    return {p.name: digest(p) for p in [base / "config.json", *sorted(weights)]}


def train(stage):
    args = arguments().parse_args()
    hardware(startup=True)
    seed = SEED if stage in ("domain", "v1") else 42
    torch = runtime(seed)
    from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                              TrainerCallback, TrainingArguments,
                              DataCollatorForLanguageModeling)
    from peft import LoraConfig, PeftModel, get_peft_model
    from trl import SFTConfig, SFTTrainer
    base = args.base_model.resolve()
    identity = base_identity(base)
    destination = ROOT / "models" / OUTPUTS[stage]
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Output already exists: {destination}")
    parent = ROOT / "models/domain_adapter"
    if stage != "domain":
        proof = json.loads((ROOT / "run_records/domain.json").read_text(encoding="utf-8"))
        if proof["adapter_sha256"] != digest(parent / "adapter_model.safetensors"):
            raise ValueError("Domain adapter does not match this project's training record")
        if proof["base_sha256"] != identity:
            raise ValueError("Base model changed after domain training")
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "right"
    data = prepare_data(stage, tokenizer)
    model = AutoModelForCausalLM.from_pretrained(
        base, dtype=torch.bfloat16, device_map={"": 0}, low_cpu_mem_usage=True,
        attn_implementation="sdpa", local_files_only=True)
    if stage == "domain":
        model = get_peft_model(model, LoraConfig(
            r=8, lora_alpha=16, lora_dropout=0.05, target_modules="all-linear",
            task_type="CAUSAL_LM"))
    else:
        model = PeftModel.from_pretrained(model, parent, is_trainable=True)
    model.config.use_cache = False
    model.enable_input_require_grads()

    class Guard(TrainerCallback):
        def on_step_begin(self, args, state, control, **kwargs):
            cool_down()

    epochs = {"domain": 2, "v1": 3, "v2": 1, "v3": 1}[stage]
    rate = 1e-4 if stage in ("domain", "v1") else 5e-5
    settings = dict(
        output_dir=str(destination), num_train_epochs=epochs,
        per_device_train_batch_size=2, per_device_eval_batch_size=2,
        learning_rate=rate, logging_steps=20 if stage in ("domain", "v1") else 10,
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=1,
        report_to="none", seed=seed, data_seed=seed, bf16=True, fp16=False,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=0, dataloader_pin_memory=False)
    common = dict(model=model, train_dataset=data["train"],
                  eval_dataset=data["validation"], callbacks=[Guard()])
    if stage == "domain":
        trainer = Trainer(
            args=TrainingArguments(**settings),
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False), **common)
    else:
        trainer = SFTTrainer(
            args=SFTConfig(**settings, max_length=256, completion_only_loss=True,
                           use_liger_kernel=False),
            processing_class=tokenizer, **common)
        for split in (trainer.train_dataset, trainer.eval_dataset):
            for item in split:
                mask = item["completion_mask"]
                if not any(mask) or all(mask):
                    raise ValueError("Invalid completion-only loss mask")
    started = time.time()
    result = trainer.train()
    model.save_pretrained(destination)
    tokenizer.save_pretrained(destination)
    save_json(ROOT / f"run_records/{stage}.json", {
        "stage": stage, "seed": seed, "epochs": epochs, "learning_rate": rate,
        "train_rows": len(data["train"]), "validation_rows": len(data["validation"]),
        "started_unix": started, "finished_unix": time.time(),
        "metrics": result.metrics, "history": trainer.state.log_history,
        "base_sha256": identity,
        "adapter_sha256": digest(destination / "adapter_model.safetensors"),
        "domain_sha256": digest(parent / "adapter_model.safetensors") if stage != "domain" else None,
        "data_sha256": {p.name: digest(p) for p in sorted((ROOT / "data").glob("*.jsonl"))},
        "script_sha256": {p.name: digest(p) for p in sorted(ROOT.glob("*.py"))}})
    print(f"Saved {stage} adapter and training record", flush=True)


def export_answers():
    parser = arguments()
    parser.add_argument("--version", choices=("v1", "v2", "v3"), required=True)
    args = parser.parse_args()
    hardware(startup=True)
    torch = runtime()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    version = args.version
    target = ROOT / f"SFT_{version.upper()}_Test.txt"
    if target.exists():
        raise FileExistsError(f"Answer export already exists: {target}")
    base = args.base_model.resolve()
    adapter = ROOT / "models" / OUTPUTS[version]
    proof = json.loads((ROOT / f"run_records/{version}.json").read_text(encoding="utf-8"))
    if proof["adapter_sha256"] != digest(adapter / "adapter_model.safetensors"):
        raise ValueError("Adapter differs from the training record")
    identity = base_identity(base)
    if proof["base_sha256"] != identity:
        raise ValueError("The training and inference base models differ")
    tokenizer = AutoTokenizer.from_pretrained(adapter, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        base, dtype=torch.bfloat16, device_map={"": 0}, low_cpu_mem_usage=True,
        local_files_only=True, attn_implementation="sdpa")
    model = PeftModel.from_pretrained(model, adapter).eval()
    rows = read_rows(ROOT / "data/sft_challenge_test.jsonl")
    records, blocks = [], []
    for position, row in enumerate(rows, 1):
        cool_down()
        # The challenge prompt is identical across versions. Expected answers are not inputs.
        prompt = prompt_for(row["prompt"])
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=100, do_sample=False)
        full_tokens = output[0].tolist()
        full_text = tokenizer.decode(full_tokens, skip_special_tokens=True)
        generated = full_tokens[inputs.input_ids.shape[1]:]
        records.append({
            "id": row["id"], "question": row["prompt"], "input_prompt": prompt,
            "input_ids": inputs.input_ids[0].tolist(), "generated_ids": generated,
            "response": tokenizer.decode(generated, skip_special_tokens=True),
            "full_text": full_text})
        blocks.append("\n====================\nQUESTION:\n" + row["prompt"]
                      + "\n\nEXPECTED POINTS:\n" + str(row["expected_points"])
                      + "\n\nMODEL:\n" + full_text + "\n")
        save_json(ROOT / f"run_records/{version}_answers.partial.json", records)
        print(f"{version} question {position}/{len(rows)}", flush=True)
    text = "".join(blocks)
    target.write_text(text, encoding="utf-8")
    save_json(ROOT / f"run_records/{version}_answers.json", {
        "version": version, "base_sha256": identity,
        "adapter_sha256": proof["adapter_sha256"],
        "questions_sha256": digest(ROOT / "data/sft_challenge_test.jsonl"),
        "export_sha256": digest(target), "answers": records})
    print(f"Saved {target.name}", flush=True)
