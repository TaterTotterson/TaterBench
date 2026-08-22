# Tater Bench Results

Best observed hardware result for each model and mode running through Tater's llama.cpp and MLX engines.

> 31 duplicate runs with identical graded outcomes on the same hardware type are omitted, keeping the newest representative. Distinct outcomes are averaged within each hardware type, then Overall selects the best hardware result for each model and mode.

| Model | Inputs | Engine | Mode | Fitness | Tater Score | Required Accuracy | Gen tok/s | TTFT | Best Hardware | Tested Devices | Unique Runs | Observations | Suite |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---|
| TaterTotterson/Gemma-4-26B-A4B-IT-UD-Q4_K_XL-mlx-Tater-NoThink | Text only | mlx | BASELINE | Tater Ready | 88.23 | 96.14% | 51.69 | 0.21s | Apple M3 Ultra · 96.0 GiB | 1 | 1 | 2 | tater-core-0.3 |
| TaterTotterson/gemma-4-26B-A4B-it-GGUF-Tater-NoThink | Vision, Video | llama.cpp | MTP (+30.5%) | Tater Ready | 86.11 | 92.94% | 106.30 | 0.54s | Apple M3 Ultra · 96.0 GiB | 2 | 1 | 2 | tater-core-0.3 |
| TaterTotterson/gemma-4-26B-A4B-it-GGUF-Tater-NoThink | Vision, Video | llama.cpp | BASELINE | Tater Ready | 85.61 | 92.94% | 81.46 | 0.53s | Apple M3 Ultra · 96.0 GiB | 2 | 1 | 2 | tater-core-0.3 |
| TaterTotterson/gemma-4-26B-A4B-it-GGUF-Tater-NoThink | Vision, Video | llama.cpp | DFLASH (-52.6%) | Tater Ready | 84.72 | 92.94% | 38.57 | 0.54s | Apple M3 Ultra · 96.0 GiB | 2 | 1 | 2 | tater-core-0.3 |
| TaterTotterson/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF-Tater-NoThink | Vision, Video | llama.cpp | BASELINE | Tater Ready | 84.06 | 90.87% | 98.72 | 0.66s | Apple M3 Ultra · 96.0 GiB | 2 | 1 | 2 | tater-core-0.3 |
| TaterTotterson/Qwen3.8-27B-GGUF-Tater-NoThink | Vision, Video | llama.cpp | MTP (+93.0%) | Tater Ready | 82.47 | 90.66% | 21.31 | 2.76s | AMD RYZEN AI MAX+ 395 w/ Radeon 8060S · 64.0 GiB | 2 | 1 | 2 | tater-core-0.3 |
| unsloth/gemma-4-12B-it-qat-GGUF | Vision, Video, Audio | llama.cpp | BASELINE | Tater Ready | 82.43 | 89.97% | 59.89 | 1.24s | Apple M3 Ultra · 96.0 GiB | 2 | 1 | 2 | tater-core-0.3 |
| TaterTotterson/Qwen3.6-35B-A3B-MTP-GGUF-Tater-NoThink | Vision, Video | llama.cpp | MTP (+50.8%) | Tater Ready | 82.40 | 88.25% | 79.44 | 1.01s | AMD RYZEN AI MAX+ 395 w/ Radeon 8060S · 64.0 GiB | 2 | 1 | 2 | tater-core-0.3 |
| TaterTotterson/Qwen3.8-27B-GGUF-Tater-NoThink | Vision, Video | llama.cpp | BASELINE | Tater Ready | 82.24 | 90.66% | 26.35 | 3.36s | Apple M3 Ultra · 96.0 GiB | 2 | 1 | 2 | tater-core-0.3 |
| TaterTotterson/Qwen3.6-35B-A3B-MTP-GGUF-Tater-NoThink | Vision, Video | llama.cpp | BASELINE | Tater Ready | 81.26 | 88.25% | 76.91 | 0.60s | Apple M3 Ultra · 96.0 GiB | 2 | 1 | 2 | tater-core-0.3 |
| ornith-ai/Ornith-1.5-35B-A3B-GGUF | Vision, Video | llama.cpp | BASELINE | Tater Ready | 80.89 | 87.70% | 83.34 | 0.62s | Apple M3 Ultra · 96.0 GiB | 2 | 1 | 2 | tater-core-0.3 |
| unsloth/gemma-4-E2B-it-qat-GGUF | Vision, Video | llama.cpp | MTP (+95.9%) | Limited — Not Ready | 79.90 | 82.17% | 193.81 | 0.08s | AMD RYZEN AI MAX+ 395 w/ Radeon 8060S · 64.0 GiB | 2 | 1 | 2 | tater-core-0.3 |
| unsloth/gemma-4-E4B-it-qat-GGUF | Vision, Video | llama.cpp | BASELINE | Limited — Not Ready | 79.87 | 85.51% | 93.26 | 0.16s | Apple M3 Ultra · 96.0 GiB | 2 | 1 | 2 | tater-core-0.3 |
| ornith-ai/Ornith-1.5-9B-GGUF | Vision | llama.cpp | BASELINE | Limited — Not Ready | 79.35 | 86.12% | 76.24 | 1.01s | Apple M3 Ultra · 96.0 GiB | 2 | 1 | 2 | tater-core-0.3 |
| unsloth/gemma-4-E2B-it-qat-GGUF | Vision, Video | llama.cpp | BASELINE | Limited — Not Ready | 76.86 | 80.51% | 130.94 | 0.09s | Apple M3 Ultra · 96.0 GiB | 2 | 1 | 2 | tater-core-0.3 |
| unsloth/Qwen3.5-0.8B-GGUF | Vision, Video | llama.cpp | BASELINE | Not Fit | 49.90 | 66.62% | 215.04 | 0.12s | Apple M3 Ultra · 96.0 GiB | 2 | 1 | 2 | tater-core-0.3 |
| LiquidAI/LFM2.5-Audio-1.5B-GGUF | Vision, Video, Audio | llama.cpp | BASELINE | Not Fit | 49.90 | 48.55% | 343.60 | 0.16s | Apple M3 Ultra · 96.0 GiB | 2 | 1 | 2 | tater-core-0.3 |
| allenai/OLMoE-1B-7B-0924-Instruct-GGUF | Vision, Video | llama.cpp | BASELINE | Not Fit | 43.24 | 39.62% | 263.60 | 0.06s | Apple M3 Ultra · 96.0 GiB | 2 | 1 | 2 | tater-core-0.3 |

## Devices

| Device | Unique published runs | Observations |
|---|---:|---:|
| AMD RYZEN AI MAX+ 395 w/ Radeon 8060S · 64.0 GiB | 21 | 34 |
| Apple M3 Ultra · 96.0 GiB | 18 | 36 |

## Method

Tater Score starts with a 100-point raw formula: 90 points for required, category-weighted accuracy (40 Astraeus routing and tool selection, 30 Thanatos tool execution, 15 synthesis, and 5 chat), 7 for generation speed, 2 for time to first token, and 1 for peak-memory efficiency. Spudex is reported as an optional capability and contributes no score or readiness penalty. Limited results are capped at 79.9 and Not Fit results at 49.9 before outcome-distinct runs are averaged within a hardware type. Performance and efficiency are normalized within matching hardware, suite, context, and prompt profile.

Fitness is a separate reliability verdict. Tater Ready requires at least 85% weighted required accuracy, 85% tool accuracy, 80% routing, 80% synthesis, and 80% chat. Limited results miss a required readiness gate; Not Fit results have required, tool, or routing accuracy below 70%. Spudex is not a readiness gate. Overall uses the verdict from the selected best hardware result; every device-specific verdict remains visible in its hardware tab. Results based on fewer than two observations on the selected hardware are provisional.

Model input badges come from Tater's downloaded-model registry. Vision, Video, and Audio are informational and are not scored by this text/tool suite. Audio can include speech or music input and does not imply that a model generates music.

Tater Bench uses a versioned, frozen synthetic Tater runtime for routing, strict tool-call, synthesis, chat, and Spudex scenarios. Runs are considered duplicates only when the hardware type, model configuration, accuracy summary, and every graded scenario outcome match; timing variation is ignored. The newest duplicate is published while the original batch files remain untouched. Hidden duplicates still count as observations for reproducibility and provisional-status checks. Outcome-distinct runs are averaged within each hardware type. Overall selects the best fitness-qualified hardware result for each model and mode, while hardware tabs retain every device-specific result. Each result records the model, engine, speculative mode, suite version, prompt profile, hardware fingerprint, context, and raw per-scenario response.

MTP, DFlash, and DSpark percentages compare generation speed against the matching baseline run on the same hardware and suite.
