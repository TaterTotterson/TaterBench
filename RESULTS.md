# Tater Bench Results

Cross-device average accuracy and real-world speed for models running through Tater's llama.cpp and MLX engines.

> Repeat submissions are averaged per hardware type first, then hardware-type averages are combined so one frequently tested device cannot outweigh the others.

| Model | Engine | Mode | Avg Tater Score | Avg Accuracy | Avg Gen tok/s | Avg TTFT | Devices | Runs | Suite |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| TaterTotterson/gemma-4-26B-A4B-it-GGUF-Tater-NoThink | llama.cpp | MTP (+50.1%) | 82.00 | 90.84% | 91.00 | 0.59s | 2 | 4 | tater-core-0.2 |
| TaterTotterson/Qwen3.6-35B-A3B-MTP-GGUF-Tater-NoThink | llama.cpp | MTP (+38.2%) | 80.19 | 87.87% | 88.80 | 0.70s | 2 | 4 | tater-core-0.2 |
| TaterTotterson/gemma-4-26B-A4B-it-GGUF-Tater-NoThink | llama.cpp | DFLASH (-28.1%) | 74.16 | 90.84% | 39.78 | 0.62s | 2 | 4 | tater-core-0.2 |
| TaterTotterson/Gemma-4-26B-A4B-IT-UD-Q4_K_XL-mlx-Tater-NoThink | mlx | BASELINE | 73.63 | 95.33% | 49.77 | 0.26s | 1 | 2 | tater-core-0.2 |
| TaterTotterson/Qwen3.6-35B-A3B-MTP-GGUF-Tater-NoThink | llama.cpp | BASELINE | 67.69 | 75.83% | 65.65 | 0.67s | 2 | 4 | tater-core-0.2 |
| TaterTotterson/Qwen3.8-27B-GGUF-Tater-NoThink | llama.cpp | MTP (+42.6%) | 67.58 | 88.84% | 22.82 | 2.58s | 2 | 4 | tater-core-0.2 |
| unsloth/Qwen3.5-0.8B-GGUF | llama.cpp | BASELINE | 67.20 | 53.14% | 214.21 | 0.10s | 1 | 1 | tater-core-0.2 |
| TaterTotterson/Qwen3.8-27B-GGUF-Tater-NoThink | llama.cpp | BASELINE | 61.37 | 81.04% | 18.91 | 2.54s | 2 | 4 | tater-core-0.2 |
| TaterTotterson/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF-Tater-NoThink | llama.cpp | BASELINE | 60.08 | 61.27% | 80.31 | 0.72s | 2 | 4 | tater-core-0.2 |
| TaterTotterson/gemma-4-26B-A4B-it-GGUF-Tater-NoThink | llama.cpp | BASELINE | 56.44 | 60.94% | 63.28 | 0.57s | 2 | 4 | tater-core-0.2 |

## Devices

| Device | Submitted runs |
|---|---:|
| AMD RYZEN AI MAX+ 395 w/ Radeon 8060S · 64.0 GiB | 16 |
| Apple M3 Ultra · 96.0 GiB | 19 |

## Method

Tater Score is a 100-point composite: 70 points for task accuracy, 20 for generation speed, 5 for time to first token, and 5 for peak-memory efficiency. Performance and efficiency are normalized within matching hardware, suite, context, and prompt profile.

Tater Bench uses a versioned, frozen synthetic Tater runtime for routing, strict tool-call, synthesis, chat, and Spudex scenarios. Repeated runs are averaged by hardware type before every represented hardware type receives equal weight. Each result records the model, engine, speculative mode, suite version, prompt profile, hardware fingerprint, context, and raw per-scenario response.

MTP, DFlash, and DSpark percentages compare generation speed against the matching baseline run on the same hardware and suite.
