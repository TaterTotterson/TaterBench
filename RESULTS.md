# Tater Bench Results

Best observed hardware result for each model and mode running through Tater's llama.cpp and MLX engines.

> 27 duplicate runs with identical graded outcomes on the same hardware type are omitted, keeping the newest representative. Distinct outcomes are averaged within each hardware type, then Overall selects the best hardware result for each model and mode.

| Model | Engine | Mode | Fitness | Tater Score | Accuracy | Gen tok/s | TTFT | Best Hardware | Tested Devices | Unique Runs | Observations | Suite |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---|
| TaterTotterson/Gemma-4-26B-A4B-IT-UD-Q4_K_XL-mlx-Tater-NoThink | mlx | BASELINE | Tater Ready | 88.46 | 97.02% | 51.69 | 0.21s | Apple M3 Ultra · 96.0 GiB | 1 | 1 | 2 | tater-core-0.3 |
| TaterTotterson/gemma-4-26B-A4B-it-GGUF-Tater-NoThink | llama.cpp | MTP (+30.5%) | Tater Ready | 85.80 | 93.84% | 106.30 | 0.54s | Apple M3 Ultra · 96.0 GiB | 2 | 1 | 2 | tater-core-0.3 |
| TaterTotterson/gemma-4-26B-A4B-it-GGUF-Tater-NoThink | llama.cpp | BASELINE | Tater Ready | 85.30 | 93.84% | 81.46 | 0.53s | Apple M3 Ultra · 96.0 GiB | 2 | 1 | 2 | tater-core-0.3 |
| TaterTotterson/gemma-4-26B-A4B-it-GGUF-Tater-NoThink | llama.cpp | DFLASH (-52.6%) | Tater Ready | 84.41 | 93.84% | 38.57 | 0.54s | Apple M3 Ultra · 96.0 GiB | 2 | 1 | 2 | tater-core-0.3 |
| TaterTotterson/Qwen3.8-27B-GGUF-Tater-NoThink | llama.cpp | MTP (+93.0%) | Tater Ready | 82.17 | 91.30% | 21.31 | 2.76s | AMD RYZEN AI MAX+ 395 w/ Radeon 8060S · 64.0 GiB | 2 | 1 | 2 | tater-core-0.3 |
| TaterTotterson/Qwen3.8-27B-GGUF-Tater-NoThink | llama.cpp | BASELINE | Tater Ready | 81.94 | 91.30% | 26.35 | 3.36s | Apple M3 Ultra · 96.0 GiB | 2 | 1 | 2 | tater-core-0.3 |
| TaterTotterson/Qwen3.6-35B-A3B-MTP-GGUF-Tater-NoThink | llama.cpp | MTP (+32.2%) | Tater Ready | 81.73 | 89.70% | 101.70 | 0.63s | Apple M3 Ultra · 96.0 GiB | 2 | 1 | 2 | tater-core-0.3 |
| TaterTotterson/Qwen3.6-35B-A3B-MTP-GGUF-Tater-NoThink | llama.cpp | BASELINE | Tater Ready | 81.24 | 89.70% | 76.91 | 0.60s | Apple M3 Ultra · 96.0 GiB | 2 | 1 | 2 | tater-core-0.3 |
| TaterTotterson/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF-Tater-NoThink | llama.cpp | BASELINE | Limited — Not Ready | 79.90 | 90.02% | 98.72 | 0.66s | Apple M3 Ultra · 96.0 GiB | 2 | 1 | 2 | tater-core-0.3 |
| unsloth/gemma-4-12B-it-qat-GGUF | llama.cpp | BASELINE | Limited — Not Ready | 79.90 | 89.23% | 59.89 | 1.24s | Apple M3 Ultra · 96.0 GiB | 2 | 1 | 2 | tater-core-0.3 |
| unsloth/gemma-4-E4B-it-qat-GGUF | llama.cpp | BASELINE | Limited — Not Ready | 79.26 | 88.10% | 93.26 | 0.16s | Apple M3 Ultra · 96.0 GiB | 2 | 1 | 2 | tater-core-0.3 |
| unsloth/gemma-4-E2B-it-qat-GGUF | llama.cpp | MTP (+95.9%) | Limited — Not Ready | 79.10 | 82.54% | 193.81 | 0.08s | AMD RYZEN AI MAX+ 395 w/ Radeon 8060S · 64.0 GiB | 2 | 1 | 2 | tater-core-0.3 |
| unsloth/gemma-4-E2B-it-qat-GGUF | llama.cpp | BASELINE | Limited — Not Ready | 74.06 | 81.43% | 130.94 | 0.09s | Apple M3 Ultra · 96.0 GiB | 2 | 1 | 2 | tater-core-0.3 |
| unsloth/Qwen3.5-0.8B-GGUF | llama.cpp | BASELINE | Not Fit | 49.90 | 65.24% | 215.04 | 0.12s | Apple M3 Ultra · 96.0 GiB | 2 | 1 | 2 | tater-core-0.3 |
| LiquidAI/LFM2.5-Audio-1.5B-GGUF | llama.cpp | BASELINE | Not Fit | 45.04 | 46.24% | 343.60 | 0.16s | Apple M3 Ultra · 96.0 GiB | 2 | 1 | 2 | tater-core-0.3 |
| allenai/OLMoE-1B-7B-0924-Instruct-GGUF | llama.cpp | BASELINE | Not Fit | 40.91 | 42.96% | 263.60 | 0.06s | Apple M3 Ultra · 96.0 GiB | 2 | 1 | 2 | tater-core-0.3 |

## Devices

| Device | Unique published runs | Observations |
|---|---:|---:|
| AMD RYZEN AI MAX+ 395 w/ Radeon 8060S · 64.0 GiB | 19 | 30 |
| Apple M3 Ultra · 96.0 GiB | 16 | 32 |

## Method

Tater Score starts with a 100-point raw formula: 90 points for category-weighted accuracy (35 Astraeus routing and tool selection, 25 Thanatos tool execution, 15 Spudex, 10 synthesis, and 5 chat), 7 for generation speed, 2 for time to first token, and 1 for peak-memory efficiency. Limited results are capped at 79.9 and Not Fit results at 49.9 before outcome-distinct runs are averaged within a hardware type. Performance and efficiency are normalized within matching hardware, suite, context, and prompt profile.

Fitness is a separate reliability verdict. Tater Ready requires at least 85% overall accuracy, 85% tool accuracy, 80% routing, and 80% in every remaining category. Limited results miss a readiness gate; Not Fit results have overall, tool, or routing accuracy below 70%. Overall uses the verdict from the selected best hardware result; every device-specific verdict remains visible in its hardware tab. Results based on fewer than two observations on the selected hardware are provisional.

Tater Bench uses a versioned, frozen synthetic Tater runtime for routing, strict tool-call, synthesis, chat, and Spudex scenarios. Runs are considered duplicates only when the hardware type, model configuration, accuracy summary, and every graded scenario outcome match; timing variation is ignored. The newest duplicate is published while the original batch files remain untouched. Hidden duplicates still count as observations for reproducibility and provisional-status checks. Outcome-distinct runs are averaged within each hardware type. Overall selects the best fitness-qualified hardware result for each model and mode, while hardware tabs retain every device-specific result. Each result records the model, engine, speculative mode, suite version, prompt profile, hardware fingerprint, context, and raw per-scenario response.

MTP, DFlash, and DSpark percentages compare generation speed against the matching baseline run on the same hardware and suite.
