# Tater Bench Results

Cross-device average accuracy and real-world speed for models running through Tater's llama.cpp and MLX engines.

> Repeat submissions are averaged per hardware type first, then hardware-type averages are combined so one frequently tested device cannot outweigh the others.

| Model | Engine | Mode | Avg Tater Score | Avg Accuracy | Avg Gen tok/s | Avg TTFT | Devices | Runs | Suite |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| TaterTotterson/gemma-4-26B-A4B-it-GGUF-Tater-NoThink | llama.cpp | MTP (+49.9%) | 89.91 | 90.84% | 91.01 | 0.59s | 2 | 2 | tater-core-0.2 |
| TaterTotterson/Qwen3.6-35B-A3B-MTP-GGUF-Tater-NoThink | llama.cpp | MTP (+38.8%) | 87.44 | 87.87% | 89.99 | 0.70s | 2 | 2 | tater-core-0.2 |
| TaterTotterson/Gemma-4-26B-A4B-IT-UD-Q4_K_XL-mlx-Tater-NoThink | mlx | BASELINE | 85.75 | 95.33% | 50.01 | 0.27s | 1 | 1 | tater-core-0.2 |
| TaterTotterson/gemma-4-26B-A4B-it-GGUF-Tater-NoThink | llama.cpp | DFLASH (-28.0%) | 78.57 | 90.84% | 39.89 | 0.62s | 2 | 2 | tater-core-0.2 |
| TaterTotterson/Qwen3.6-35B-A3B-MTP-GGUF-Tater-NoThink | llama.cpp | BASELINE | 74.09 | 75.83% | 65.95 | 0.67s | 2 | 2 | tater-core-0.2 |
| TaterTotterson/Qwen3.8-27B-GGUF-Tater-NoThink | llama.cpp | MTP (+41.1%) | 70.54 | 88.84% | 22.67 | 2.57s | 2 | 2 | tater-core-0.2 |
| TaterTotterson/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF-Tater-NoThink | llama.cpp | BASELINE | 67.85 | 61.27% | 80.94 | 0.71s | 2 | 2 | tater-core-0.2 |
| TaterTotterson/Qwen3.8-27B-GGUF-Tater-NoThink | llama.cpp | BASELINE | 65.05 | 81.09% | 19.03 | 2.53s | 2 | 2 | tater-core-0.2 |
| TaterTotterson/gemma-4-26B-A4B-it-GGUF-Tater-NoThink | llama.cpp | BASELINE | 63.34 | 60.94% | 63.37 | 0.57s | 2 | 2 | tater-core-0.2 |

## Devices

| Device | Submitted runs |
|---|---:|
| AMD RYZEN AI MAX+ 395 w/ Radeon 8060S · 64.0 GiB | 8 |
| Apple M3 Ultra · 96.0 GiB | 9 |

## Method

Tater Score is a 100-point composite: 70 points for task accuracy, 20 for generation speed, 5 for time to first token, and 5 for peak-memory efficiency. Performance and efficiency are normalized within matching hardware, suite, context, and prompt profile.

Tater Bench uses a versioned, frozen synthetic Tater runtime for routing, strict tool-call, synthesis, chat, and Spudex scenarios. Repeated runs are averaged by hardware type before every represented hardware type receives equal weight. Each result records the model, engine, speculative mode, suite version, prompt profile, hardware fingerprint, context, and raw per-scenario response.

MTP, DFlash, and DSpark percentages compare generation speed against the matching baseline run on the same hardware and suite.
