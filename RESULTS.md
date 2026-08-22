# Tater Bench Results

Cross-device average accuracy and real-world speed for models running through Tater's llama.cpp and MLX engines.

> 27 duplicate runs with identical graded outcomes on the same hardware type are omitted, keeping the newest representative. Distinct outcomes are averaged per hardware type before hardware-type averages are combined.

| Model | Engine | Mode | Fitness | Avg Tater Score | Avg Accuracy | Avg Gen tok/s | Avg TTFT | Devices | Unique Runs | Observations | Suite |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| TaterTotterson/Gemma-4-26B-A4B-IT-UD-Q4_K_XL-mlx-Tater-NoThink | mlx | BASELINE | Tater Ready | 88.46 | 97.02% | 51.69 | 0.21s | 1 | 1 | 2 | tater-core-0.3 |
| TaterTotterson/gemma-4-26B-A4B-it-GGUF-Tater-NoThink | llama.cpp | MTP (+50.7%) | Tater Ready | 85.34 | 93.03% | 90.85 | 0.62s | 2 | 2 | 4 | tater-core-0.3 |
| TaterTotterson/gemma-4-26B-A4B-it-GGUF-Tater-NoThink | llama.cpp | DFLASH (-23.7%) | Tater Ready | 84.13 | 93.03% | 42.48 | 0.64s | 2 | 2 | 4 | tater-core-0.3 |
| TaterTotterson/Qwen3.8-27B-GGUF-Tater-NoThink | llama.cpp | MTP (+42.7%) | Tater Ready | 82.03 | 91.30% | 22.82 | 3.10s | 2 | 2 | 4 | tater-core-0.3 |
| TaterTotterson/Qwen3.6-35B-A3B-MTP-GGUF-Tater-NoThink | llama.cpp | MTP (+41.5%) | Mixed by Hardware | 80.81 | 88.99% | 90.57 | 0.82s | 2 | 2 | 4 | tater-core-0.3 |
| unsloth/gemma-4-E2B-it-qat-GGUF | llama.cpp | MTP (+57.4%) | Limited — Not Ready | 76.81 | 81.99% | 174.75 | 0.09s | 2 | 2 | 4 | tater-core-0.3 |
| TaterTotterson/Qwen3.8-27B-GGUF-Tater-NoThink | llama.cpp | BASELINE | Mixed by Hardware | 60.11 | 71.84% | 18.70 | 3.03s | 2 | 2 | 4 | tater-core-0.3 |
| TaterTotterson/Qwen3.6-35B-A3B-MTP-GGUF-Tater-NoThink | llama.cpp | BASELINE | Mixed by Hardware | 58.73 | 70.17% | 64.80 | 0.76s | 2 | 2 | 4 | tater-core-0.3 |
| TaterTotterson/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF-Tater-NoThink | llama.cpp | BASELINE | Mixed by Hardware | 55.59 | 63.96% | 79.40 | 0.81s | 2 | 2 | 4 | tater-core-0.3 |
| TaterTotterson/gemma-4-26B-A4B-it-GGUF-Tater-NoThink | llama.cpp | BASELINE | Mixed by Hardware | 52.96 | 59.31% | 62.78 | 0.59s | 2 | 2 | 4 | tater-core-0.3 |
| unsloth/gemma-4-12B-it-qat-GGUF | llama.cpp | BASELINE | Mixed by Hardware | 52.95 | 61.29% | 43.30 | 1.09s | 2 | 3 | 4 | tater-core-0.3 |
| unsloth/gemma-4-E4B-it-qat-GGUF | llama.cpp | BASELINE | Mixed by Hardware | 48.92 | 54.79% | 75.64 | 0.14s | 2 | 3 | 4 | tater-core-0.3 |
| unsloth/gemma-4-E2B-it-qat-GGUF | llama.cpp | BASELINE | Mixed by Hardware | 43.59 | 48.30% | 115.50 | 0.08s | 2 | 3 | 4 | tater-core-0.3 |
| unsloth/Qwen3.5-0.8B-GGUF | llama.cpp | BASELINE | Not Fit | 37.78 | 45.52% | 198.74 | 0.12s | 2 | 2 | 4 | tater-core-0.3 |
| LiquidAI/LFM2.5-Audio-1.5B-GGUF | llama.cpp | BASELINE | Not Fit | 33.67 | 33.47% | 285.62 | 0.14s | 2 | 2 | 4 | tater-core-0.3 |
| allenai/OLMoE-1B-7B-0924-Instruct-GGUF | llama.cpp | BASELINE | Not Fit | 32.17 | 32.03% | 213.91 | 0.08s | 2 | 3 | 4 | tater-core-0.3 |

## Devices

| Device | Unique published runs | Observations |
|---|---:|---:|
| AMD RYZEN AI MAX+ 395 w/ Radeon 8060S · 64.0 GiB | 19 | 30 |
| Apple M3 Ultra · 96.0 GiB | 16 | 32 |

## Method

Tater Score starts with a 100-point raw formula: 90 points for category-weighted accuracy (35 Astraeus routing and tool selection, 25 Thanatos tool execution, 15 Spudex, 10 synthesis, and 5 chat), 7 for generation speed, 2 for time to first token, and 1 for peak-memory efficiency. Limited results are capped at 79.9 and Not Fit results at 49.9 before repeated runs and hardware results are averaged. Performance and efficiency are normalized within matching hardware, suite, context, and prompt profile.

Fitness is a separate reliability verdict. Tater Ready requires at least 85% overall accuracy, 85% tool accuracy, 80% routing, and 80% in every remaining category. Limited results miss a readiness gate; Not Fit results have overall, tool, or routing accuracy below 70%. Aggregate labels are unanimous: a model is Tater Ready or Not Fit only when every underlying result has that verdict, while device disagreements are labeled Mixed by Hardware. Results based on fewer than two observations are provisional.

Tater Bench uses a versioned, frozen synthetic Tater runtime for routing, strict tool-call, synthesis, chat, and Spudex scenarios. Runs are considered duplicates only when the hardware type, model configuration, accuracy summary, and every graded scenario outcome match; timing variation is ignored. The newest duplicate is published while the original batch files remain untouched. Hidden duplicates still count as observations for reproducibility and provisional-status checks. Outcome-distinct runs are averaged by hardware type before every represented hardware type receives equal weight. Each result records the model, engine, speculative mode, suite version, prompt profile, hardware fingerprint, context, and raw per-scenario response.

MTP, DFlash, and DSpark percentages compare generation speed against the matching baseline run on the same hardware and suite.
