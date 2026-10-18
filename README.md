# University IT fine-tuning

This project uses the supplied SCSE GEX3 datasets to compare three supervised fine-tuning runs of Qwen3-0.6B. Each run starts from the domain adapter trained in this project. The data and challenge questions are unchanged.

Use Python 3.12 and install CUDA-enabled PyTorch 2.5.1 before installing the other packages in requirements.txt. Download the original model with download_qwen.py, or supply its directory with --base-model.

Run the course scripts in this order:

```powershell
python domain_train.py
python sft_lora.py
python test_specialized.py --version v1
python sft_lora_v2.py
python test_specialized.py --version v2
python sft_lora_v3.py
python test_specialized.py --version v3
python verify_submission.py
```

The trainer keeps the supplied epoch counts, batch sizes and learning rates. Domain training and V1 retain seed 48391; V2 and V3 retain the Trainer default of 42. It uses CUDA bfloat16 and gradient checkpointing for this laptop. V2 and V3 now pass separate prompt and completion fields to TRL so that the loss excludes question tokens. This preserves the original training text while fixing the loss-mask setup.

The test script keeps the same prompt and 100-token generation limit for all versions. It saves every actual response, including wrong answers, to SFT_V1_Test.txt, SFT_V2_Test.txt and SFT_V3_Test.txt. Expected points are printed for comparison, never supplied to the model. Records include generated token IDs and hashes linking each export to its adapter.

Use run_guarded.ps1 for monitored local runs. The wrapper checks RAM and GPU temperature and terminates only its own worker if limits are exceeded. These software checks cannot guarantee hardware safety.

## Recorded run

Domain training and all three fine-tuning runs finished on 18 October 2026 on an RTX 3060 Laptop GPU. The three TXT files contain 15 challenge responses each. All 45 responses were checked against the recorded generated token sequences and the corresponding adapter hashes.

The responses are experimental results, not a reliable IT support guide. V1 gives incorrect advice on several questions, including print-job timing and visitor Wi-Fi. V2 often repeats policy-like text without answering the question. V3 recalls some policy facts but still adds unsupported advice and invents a phone number in the final question. These answers have not been corrected in the TXT files. Low validation loss on this repetitive dataset does not establish reliable reasoning on the challenge questions.

The assignment's fine-tuning submission consists of SFT_V1_Test.txt, SFT_V2_Test.txt and SFT_V3_Test.txt. The accompanying retrieval project is [scse_26_gex3_rag](https://github.com/Fgq2354/scse_26_gex3_rag). Repository upload is separate from submitting the files and link to the course portal.
