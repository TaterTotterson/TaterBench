# Tater Bench Results

Cross-device average accuracy and real-world speed for models running through Tater's llama.cpp and MLX engines.

> Repeat submissions are averaged per hardware type first, then hardware-type averages are combined so one frequently tested device cannot outweigh the others.

| Model | Engine | Mode | Fitness | Avg Tater Score | Avg Accuracy | Avg Gen tok/s | Avg TTFT | Devices | Runs | Suite |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| TaterTotterson/Gemma-4-26B-A4B-IT-UD-Q4_K_XL-mlx-Tater-NoThink | mlx | BASELINE | Tater Ready | 88.43 | 95.33% | 49.77 | 0.26s | 1 | 2 | tater-core-0.2 |
| TaterTotterson/gemma-4-26B-A4B-it-GGUF-Tater-NoThink | llama.cpp | MTP (+50.1%) | Tater Ready | 88.10 | 90.84% | 91.00 | 0.59s | 2 | 4 | tater-core-0.2 |
| TaterTotterson/gemma-4-26B-A4B-it-GGUF-Tater-NoThink | llama.cpp | DFLASH (-28.1%) | Tater Ready | 85.42 | 90.84% | 39.78 | 0.62s | 2 | 4 | tater-core-0.2 |
| TaterTotterson/Qwen3.8-27B-GGUF-Tater-NoThink | llama.cpp | MTP (+42.6%) | Tater Ready | 82.22 | 88.84% | 22.82 | 2.58s | 2 | 4 | tater-core-0.2 |
| TaterTotterson/Qwen3.6-35B-A3B-MTP-GGUF-Tater-NoThink | llama.cpp | MTP (+38.2%) | Limited — Not Ready | 85.83 | 87.87% | 88.80 | 0.70s | 2 | 4 | tater-core-0.2 |
| TaterTotterson/Qwen3.8-27B-GGUF-Tater-NoThink | llama.cpp | BASELINE | Mixed by Hardware | 73.78 | 81.04% | 18.91 | 2.54s | 2 | 4 | tater-core-0.2 |
| TaterTotterson/Qwen3.6-35B-A3B-MTP-GGUF-Tater-NoThink | llama.cpp | BASELINE | Mixed by Hardware | 72.36 | 75.83% | 65.65 | 0.67s | 2 | 4 | tater-core-0.2 |
| TaterTotterson/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF-Tater-NoThink | llama.cpp | BASELINE | Mixed by Hardware | 62.46 | 61.27% | 80.31 | 0.72s | 2 | 4 | tater-core-0.2 |
| TaterTotterson/gemma-4-26B-A4B-it-GGUF-Tater-NoThink | llama.cpp | BASELINE | Mixed by Hardware | 60.83 | 60.94% | 63.28 | 0.57s | 2 | 4 | tater-core-0.2 |
| unsloth/Qwen3.5-0.8B-GGUF | llama.cpp | BASELINE | Not Fit — Provisional | 59.38 | 53.14% | 214.21 | 0.10s | 1 | 1 | tater-core-0.2 |

## Devices

| Device | Submitted runs |
|---|---:|
| AMD RYZEN AI MAX+ 395 w/ Radeon 8060S · 64.0 GiB | 16 |
| Apple M3 Ultra · 96.0 GiB | 19 |

## Method

Tater Score is a 100-point composite: 90 points for category-weighted accuracy (35 tool accuracy, 25 routing, 15 Spudex, 10 synthesis, and 5 chat), 7 for generation speed, 2 for time to first token, and 1 for peak-memory efficiency. Performance and efficiency are normalized within matching hardware, suite, context, and prompt profile.

Fitness is a separate reliability verdict. Tater Ready requires at least 85% overall accuracy, 85% tool accuracy, 80% routing, and 80% in every remaining category. Limited results miss a readiness gate; Not Fit results have overall, tool, or routing accuracy below 70%. Aggregate labels are unanimous: a model is Tater Ready or Not Fit only when every underlying result has that verdict, while device disagreements are labeled Mixed by Hardware. Results based on fewer than two runs are provisional.

Tater Bench uses a versioned, frozen synthetic Tater runtime for routing, strict tool-call, synthesis, chat, and Spudex scenarios. Repeated runs are averaged by hardware type before every represented hardware type receives equal weight. Each result records the model, engine, speculative mode, suite version, prompt profile, hardware fingerprint, context, and raw per-scenario response.

MTP, DFlash, and DSpark percentages compare generation speed against the matching baseline run on the same hardware and suite.
