# Tater Bench Results

Cross-device average accuracy and real-world speed for models running through Tater's llama.cpp and MLX engines.

> Repeat submissions are averaged per hardware type first, then hardware-type averages are combined so one frequently tested device cannot outweigh the others.

| Model | Engine | Mode | Avg Tater Score | Avg Accuracy | Avg Gen tok/s | Avg TTFT | Devices | Runs | Suite |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| TaterTotterson/gemma-4-26B-A4B-it-GGUF-Tater-NoThink | llama.cpp | MTP (+30.2%) | 91.52 | 91.78% | 108.76 | 0.52s | 1 | 1 | tater-core-0.2 |
| TaterTotterson/gemma-4-26B-A4B-it-GGUF-Tater-NoThink | llama.cpp | BASELINE | 87.14 | 91.78% | 83.53 | 0.51s | 1 | 1 | tater-core-0.2 |
| TaterTotterson/Qwen3.6-35B-A3B-MTP-GGUF-Tater-NoThink | llama.cpp | MTP (+27.4%) | 85.76 | 86.98% | 101.07 | 0.55s | 1 | 1 | tater-core-0.2 |
| TaterTotterson/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF-Tater-NoThink | llama.cpp | BASELINE | 85.76 | 85.13% | 102.12 | 0.56s | 1 | 1 | tater-core-0.2 |
| TaterTotterson/Gemma-4-26B-A4B-IT-UD-Q4_K_XL-mlx-Tater-NoThink | mlx | BASELINE | 85.75 | 95.33% | 50.01 | 0.27s | 1 | 1 | tater-core-0.2 |
| TaterTotterson/Qwen3.6-35B-A3B-MTP-GGUF-Tater-NoThink | llama.cpp | BASELINE | 82.27 | 86.98% | 79.31 | 0.53s | 1 | 1 | tater-core-0.2 |
| TaterTotterson/gemma-4-26B-A4B-it-GGUF-Tater-NoThink | llama.cpp | DFLASH (-56.4%) | 77.66 | 91.78% | 36.41 | 0.53s | 1 | 1 | tater-core-0.2 |
| TaterTotterson/Qwen3.8-27B-GGUF-Tater-NoThink | llama.cpp | BASELINE | 72.49 | 88.84% | 27.08 | 2.73s | 1 | 1 | tater-core-0.2 |
| TaterTotterson/Qwen3.8-27B-GGUF-Tater-NoThink | llama.cpp | MTP (-10.7%) | 70.83 | 88.84% | 24.18 | 2.79s | 1 | 1 | tater-core-0.2 |

## Devices

| Device | Submitted runs |
|---|---:|
| Apple M3 Ultra · 96.0 GiB | 9 |

## Method

Tater Score is a 100-point composite: 70 points for task accuracy, 20 for generation speed, 5 for time to first token, and 5 for peak-memory efficiency. Performance and efficiency are normalized within matching hardware, suite, context, and prompt profile.

Tater Bench uses a versioned, frozen synthetic Tater runtime for routing, strict tool-call, synthesis, chat, and Spudex scenarios. Repeated runs are averaged by hardware type before every represented hardware type receives equal weight. Each result records the model, engine, speculative mode, suite version, prompt profile, hardware fingerprint, context, and raw per-scenario response.

MTP, DFlash, and DSpark percentages compare generation speed against the matching baseline run on the same hardware and suite.
