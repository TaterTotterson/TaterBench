# Tater Bench Results

Accuracy and real-world speed for models running through Tater's llama.cpp and MLX engines.

> Raw accuracy and speed stay visible beside the composite Tater Score. Compare speed only on matching hardware, suite, context, and quantization.

| Model | Engine | Mode | Tater Score | Accuracy | Gen tok/s | TTFT | Load | Peak RSS | Hardware | Suite |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| TaterTotterson/gemma-4-26B-A4B-it-GGUF-Tater-NoThink | llama.cpp | MTP (+41.6%) | 92.14 | 92.85% | 122.89 | 0.23s | 2.82s | 18.4 GiB | Apple M3 Ultra / 96.0 GiB | tater-core-0.1 |
| TaterTotterson/gemma-4-26B-A4B-it-GGUF-Tater-NoThink | llama.cpp | BASELINE | 86.44 | 92.85% | 86.80 | 0.23s | 7.67s | 17.6 GiB | Apple M3 Ultra / 96.0 GiB | tater-core-0.1 |
| TaterTotterson/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF-Tater-NoThink | llama.cpp | BASELINE | 86.38 | 88.12% | 105.12 | 0.25s | 9.69s | 15.8 GiB | Apple M3 Ultra / 96.0 GiB | tater-core-0.1 |
| TaterTotterson/Qwen3.6-35B-A3B-MTP-GGUF-Tater-NoThink | llama.cpp | MTP (+34.6%) | 84.01 | 85.63% | 110.16 | 0.24s | 2.29s | 23.1 GiB | Apple M3 Ultra / 96.0 GiB | tater-core-0.1 |
| TaterTotterson/Gemma-4-26B-A4B-IT-UD-Q4_K_XL-mlx-Tater-NoThink | mlx | BASELINE | 84.00 | 91.50% | 64.61 | 0.13s | 8.40s | 17.9 GiB | Apple M3 Ultra / 96.0 GiB | tater-core-0.1 |
| TaterTotterson/Qwen3.6-35B-A3B-MTP-GGUF-Tater-NoThink | llama.cpp | BASELINE | 80.15 | 85.63% | 81.82 | 0.24s | 11.95s | 19.2 GiB | Apple M3 Ultra / 96.0 GiB | tater-core-0.1 |
| TaterTotterson/gemma-4-26B-A4B-it-GGUF-Tater-NoThink | llama.cpp | DFLASH (-50.8%) | 78.56 | 92.85% | 42.74 | 0.23s | 2.80s | 20.7 GiB | Apple M3 Ultra / 96.0 GiB | tater-core-0.1 |
| TaterTotterson/Qwen3.8-27B-GGUF-Tater-NoThink | llama.cpp | BASELINE | 73.97 | 92.85% | 27.33 | 1.16s | 2.05s | 20.0 GiB | Apple M3 Ultra / 96.0 GiB | tater-core-0.1 |
| TaterTotterson/Qwen3.8-27B-GGUF-Tater-NoThink | llama.cpp | MTP (-4.6%) | 73.14 | 92.85% | 26.07 | 1.17s | 3.57s | 23.7 GiB | Apple M3 Ultra / 96.0 GiB | tater-core-0.1 |

## Method

Tater Score is a 100-point composite: 70 points for task accuracy, 20 for generation speed, 5 for time to first token, and 5 for peak-memory efficiency. Performance and efficiency are normalized within matching hardware, suite, context, and prompt profile.

Tater Bench uses deterministic Tater-style routing, strict tool-call, synthesis, chat, and Spudex scenarios. Each result records the model, engine, speculative mode, suite version, hardware fingerprint, context, and raw per-scenario response.

MTP, DFlash, and DSpark percentages compare generation speed against the matching baseline run on the same hardware and suite.
