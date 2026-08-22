# Tater Bench Results

Best observed hardware result for each model and mode running through Tater's llama.cpp and MLX engines.

> 27 duplicate runs with identical graded outcomes on the same hardware type are omitted, keeping the newest representative. Distinct outcomes are averaged within each hardware type, then Overall selects the best hardware result for each model and mode.

| Model | Inputs | Engine | Mode | Fitness | Tater Score | Required Accuracy | Gen tok/s | TTFT | Best Hardware | Tested Devices | Unique Runs | Observations | Suite |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---|
| TaterTotterson/Gemma-4-26B-A4B-IT-UD-Q4_K_XL-mlx-Tater-NoThink | Text only | mlx | BASELINE | Tater Ready | 88.61 | 96.14% | 55.09 | 0.23s | Apple M3 Ultra · 96.0 GiB | 1 | 1 | 2 | tater-core-0.4 |
| TaterTotterson/gemma-4-26B-A4B-it-GGUF-Tater-NoThink | Vision, Video | llama.cpp | MTP (+70.8%) | Tater Ready | 87.21 | 93.00% | 75.34 | 0.71s | AMD RYZEN AI MAX+ 395 w/ Radeon 8060S · 64.0 GiB | 2 | 1 | 2 | tater-core-0.4 |
| TaterTotterson/gemma-4-26B-A4B-it-GGUF-Tater-NoThink | Vision, Video | llama.cpp | BASELINE | Tater Ready | 86.12 | 93.00% | 79.73 | 0.53s | Apple M3 Ultra · 96.0 GiB | 2 | 1 | 2 | tater-core-0.4 |
| TaterTotterson/gemma-4-26B-A4B-it-GGUF-Tater-NoThink | Vision, Video | llama.cpp | DFLASH (+9.0%) | Tater Ready | 85.98 | 93.00% | 48.10 | 0.74s | AMD RYZEN AI MAX+ 395 w/ Radeon 8060S · 64.0 GiB | 2 | 1 | 2 | tater-core-0.4 |
| unsloth/gemma-4-12B-it-qat-GGUF | Vision, Video, Audio | llama.cpp | BASELINE | Tater Ready | 84.45 | 91.70% | 62.96 | 1.22s | Apple M3 Ultra · 96.0 GiB | 2 | 1 | 2 | tater-core-0.4 |
| TaterTotterson/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF-Tater-NoThink | Vision, Video | llama.cpp | BASELINE | Tater Ready | 84.42 | 90.62% | 97.59 | 0.66s | Apple M3 Ultra · 96.0 GiB | 2 | 1 | 2 | tater-core-0.4 |
| TaterTotterson/Qwen3.6-35B-A3B-MTP-GGUF-Tater-NoThink | Vision, Video | llama.cpp | MTP (+48.9%) | Tater Ready | 82.96 | 88.00% | 77.87 | 0.94s | AMD RYZEN AI MAX+ 395 w/ Radeon 8060S · 64.0 GiB | 2 | 1 | 2 | tater-core-0.4 |
| TaterTotterson/Qwen3.8-27B-GGUF-Tater-NoThink | Vision, Video | llama.cpp | MTP (+93.4%) | Tater Ready | 82.49 | 90.41% | 21.33 | 2.76s | AMD RYZEN AI MAX+ 395 w/ Radeon 8060S · 64.0 GiB | 2 | 1 | 2 | tater-core-0.4 |
| TaterTotterson/Qwen3.8-27B-GGUF-Tater-NoThink | Vision, Video | llama.cpp | BASELINE | Tater Ready | 82.20 | 90.41% | 27.18 | 3.30s | Apple M3 Ultra · 96.0 GiB | 2 | 1 | 2 | tater-core-0.4 |
| TaterTotterson/Qwen3.6-35B-A3B-MTP-GGUF-Tater-NoThink | Vision, Video | llama.cpp | BASELINE | Tater Ready | 81.49 | 88.00% | 75.91 | 0.61s | Apple M3 Ultra · 96.0 GiB | 2 | 1 | 2 | tater-core-0.4 |
| unsloth/gemma-4-E4B-it-qat-GGUF | Vision, Video | llama.cpp | BASELINE | Limited — Not Ready | 79.90 | 85.19% | 99.13 | 0.16s | Apple M3 Ultra · 96.0 GiB | 2 | 1 | 2 | tater-core-0.4 |
| unsloth/gemma-4-E2B-it-qat-GGUF | Vision, Video | llama.cpp | MTP (+83.2%) | Limited — Not Ready | 79.90 | 84.02% | 181.02 | 0.09s | AMD RYZEN AI MAX+ 395 w/ Radeon 8060S · 64.0 GiB | 2 | 1 | 2 | tater-core-0.4 |
| unsloth/gemma-4-E2B-it-qat-GGUF | Vision, Video | llama.cpp | BASELINE | Limited — Not Ready | 79.90 | 83.52% | 141.61 | 0.09s | Apple M3 Ultra · 96.0 GiB | 2 | 1 | 2 | tater-core-0.4 |
| unsloth/Qwen3.5-0.8B-GGUF | Vision, Video | llama.cpp | BASELINE | Not Fit | 49.90 | 67.35% | 229.34 | 0.12s | Apple M3 Ultra · 96.0 GiB | 2 | 1 | 2 | tater-core-0.4 |
| allenai/OLMoE-1B-7B-0924-Instruct-GGUF | Vision, Video | llama.cpp | BASELINE | Not Fit | 46.36 | 41.28% | 262.69 | 0.06s | Apple M3 Ultra · 96.0 GiB | 2 | 1 | 2 | tater-core-0.4 |

## Devices

| Device | Unique published runs | Observations |
|---|---:|---:|
| AMD RYZEN AI MAX+ 395 w/ Radeon 8060S · 64.0 GiB | 16 | 28 |
| Apple M3 Ultra · 96.0 GiB | 15 | 30 |

## Method

Tater Score starts with a 100-point raw formula: 90 points for required, category-weighted accuracy (40 Astraeus routing and tool selection, 30 Thanatos tool execution, 15 synthesis, and 5 chat), 7 for generation speed, 2 for time to first token, and 1 for peak-memory efficiency. Spudex is reported as an optional capability and contributes no score or readiness penalty. Limited results are capped at 79.9 and Not Fit results at 49.9 before outcome-distinct runs are averaged within a hardware type. Performance and efficiency are normalized within matching hardware, suite, context, and prompt profile.

Fitness is a separate reliability verdict. Tater Ready requires at least 85% weighted required accuracy, 85% tool accuracy, 80% routing, 80% synthesis, and 80% chat. Limited results miss a required readiness gate; Not Fit results have required, tool, or routing accuracy below 70%. Spudex is not a readiness gate. Overall uses the verdict from the selected best hardware result; every device-specific verdict remains visible in its hardware tab. Results based on fewer than two observations on the selected hardware are provisional.

Astraeus execution steps must include a non-empty NL instruction. For NL-first TaterShop Verbas, Thanatos must call the correct function and populate its declared query or request field, but the NL value's wording is intentionally not graded. Structured arguments such as platforms, destinations, URLs, and filenames are still checked against the scenario contract.

Model input badges come from Tater's downloaded-model registry. Vision, Video, and Audio are informational and are not scored by this text/tool suite. Audio can include speech or music input and does not imply that a model generates music.

Tater Bench uses a versioned, frozen synthetic Tater runtime for routing, strict tool-call, synthesis, chat, and Spudex scenarios. Runs are considered duplicates only when the hardware type, model configuration, accuracy summary, and every graded scenario outcome match; timing variation is ignored. The newest duplicate is published while the original batch files remain untouched. Hidden duplicates still count as observations for reproducibility and provisional-status checks. Outcome-distinct runs are averaged within each hardware type. Overall selects the best fitness-qualified hardware result for each model and mode, while hardware tabs retain every device-specific result. Each result records the model, engine, speculative mode, suite version, prompt profile, hardware fingerprint, context, and raw per-scenario response.

MTP, DFlash, and DSpark percentages compare generation speed against the matching baseline run on the same hardware and suite.
