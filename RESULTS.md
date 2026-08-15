# Tater Bench Results

Cross-device average accuracy and real-world speed for models running through Tater's llama.cpp and MLX engines.

> Repeat submissions are averaged per hardware type first, then hardware-type averages are combined so one frequently tested device cannot outweigh the others.

| Model | Engine | Mode | Avg Tater Score | Avg Accuracy | Avg Gen tok/s | Avg TTFT | Devices | Runs | Suite |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| TaterTotterson/gemma-4-26B-A4B-it-GGUF-Tater-NoThink | llama.cpp | MTP (+49.5%) | 90.94 | 90.84% | 90.51 | 0.60s | 2 | 2 | tater-core-0.2 |
| TaterTotterson/Qwen3.6-35B-A3B-MTP-GGUF-Tater-NoThink | llama.cpp | MTP (+53.5%) | 87.05 | 87.87% | 89.18 | 0.75s | 2 | 2 | tater-core-0.2 |
| TaterTotterson/Gemma-4-26B-A4B-IT-UD-Q4_K_XL-mlx-Tater-NoThink | mlx | BASELINE | 85.75 | 95.33% | 50.01 | 0.27s | 1 | 1 | tater-core-0.2 |
| TaterTotterson/Qwen3.6-35B-A3B-MTP-GGUF-Tater-NoThink | llama.cpp | BASELINE | 80.97 | 87.87% | 61.17 | 0.74s | 2 | 2 | tater-core-0.2 |
| TaterTotterson/gemma-4-26B-A4B-it-GGUF-Tater-NoThink | llama.cpp | DFLASH (-28.7%) | 79.61 | 90.84% | 39.38 | 0.63s | 2 | 2 | tater-core-0.2 |
| TaterTotterson/Qwen3.8-27B-GGUF-Tater-NoThink | llama.cpp | BASELINE | 71.22 | 89.43% | 17.81 | 2.75s | 2 | 2 | tater-core-0.2 |
| TaterTotterson/Qwen3.8-27B-GGUF-Tater-NoThink | llama.cpp | MTP (+37.1%) | 70.09 | 88.84% | 19.98 | 7.15s | 2 | 2 | tater-core-0.2 |
| TaterTotterson/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF-Tater-NoThink | llama.cpp | BASELINE | 67.97 | 61.27% | 80.77 | 0.72s | 2 | 2 | tater-core-0.2 |
| TaterTotterson/gemma-4-26B-A4B-it-GGUF-Tater-NoThink | llama.cpp | BASELINE | 64.27 | 60.94% | 63.17 | 0.57s | 2 | 2 | tater-core-0.2 |

## Devices

| Device | Submitted runs |
|---|---:|
| AMD RYZEN AI MAX+ 395 w/ Radeon 8060S · 14.8 GiB | 8 |
| Apple M3 Ultra · 96.0 GiB | 9 |

## Method

Tater Score is a 100-point composite: 70 points for task accuracy, 20 for generation speed, 5 for time to first token, and 5 for peak-memory efficiency. Performance and efficiency are normalized within matching hardware, suite, context, and prompt profile.

Tater Bench uses a versioned, frozen synthetic Tater runtime for routing, strict tool-call, synthesis, chat, and Spudex scenarios. Repeated runs are averaged by hardware type before every represented hardware type receives equal weight. Each result records the model, engine, speculative mode, suite version, prompt profile, hardware fingerprint, context, and raw per-scenario response.

MTP, DFlash, and DSpark percentages compare generation speed against the matching baseline run on the same hardware and suite.
