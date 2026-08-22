<p align="center">
  <img src="assets/tater-bench-mascot.png" width="210" alt="Tater mascot">
</p>

# Tater Bench

Tater Bench is a standalone accuracy and performance benchmark for local models installed by [Tater](https://github.com/TaterTotterson/Tater). It reads Tater's model registry, launches each model with its proper llama.cpp or MLX engine, runs versioned Tater-style scenarios, and stores publish-safe results.

Tater does not need to be running, and Tater Bench never changes Tater's model files or settings.

## What it measures

Tater Bench records capability and performance separately, then combines them into a transparent Tater Score for the leaderboard:

- **Tater Score:** a 100-point raw formula with 90 points for required, category-weighted accuracy (40 Astraeus routing and tool selection, 30 Thanatos tool execution, 15 synthesis, and 5 chat), 7 for generation speed, 2 for time to first token, and 1 for peak-memory efficiency. Spudex is reported as an optional capability and contributes no score or readiness penalty. Limited results are capped at 79.9 and Not Fit results at 49.9 before aggregation.
- **Fitness:** Tater Ready requires at least 85% weighted required accuracy, 85% tool accuracy, 80% routing, 80% synthesis, and 80% chat. Limited results miss a required readiness gate; Not Fit results fall below 70% required, tool, or routing accuracy. Spudex is not a readiness gate.
- **Accuracy:** Astraeus routing, tool selection, and ordered planning against a curated non-overlapping catalog; Thanatos execution and arguments for a locked tool contract; normal chat and behavioral recall from frozen Memory, Personal, Guardian, Music, and Tater Tube Core context; and Hermes result synthesis. Optional Spudex action decisions are tested and displayed separately without affecting the score.
- **Speed:** model load time, time to first token, prompt speed, generation speed, complete scenario latency, and peak engine RSS.
- **Speculation:** llama.cpp targets run once without speculative decoding and again with every compatible MTP, DFlash, or DSpark draft found beside the target GGUF.
- **Model inputs:** Tater-declared Vision, Video, and Audio compatibility is published with each result. Audio can include speech or music input; these badges do not imply media generation and are not scored by the text/tool suite.
- **Hardware:** OS, CPU, architecture, core count, memory, GPU/backend, engine version, and an anonymous hardware fingerprint.

MLX models currently run a baseline pass. MTP, DFlash, and DSpark are tested on compatible llama.cpp GGUF targets because those are the speculative methods Tater exposes for that engine.

Engine support and model availability are separate: Tater's bundled llama.cpp can contain a decoder even when no compatible draft GGUF is installed. Tater Bench only schedules a speculative variant when it discovers the matching MTP, DFlash, or DSpark data for that target model.

## Install

Python 3.10 or newer is required. The benchmark runner itself has no third-party Python dependencies.

~~~bash
git clone https://github.com/TaterTotterson/TaterBench.git
cd TaterBench
python -m pip install -e .
~~~

Tater must already have installed the models and engine runtimes you want to test.

## Discover models

~~~bash
tater-bench doctor
tater-bench models
tater-bench hardware
~~~

Discovery starts from:

~~~text
~/.taterassistant/agent_lab/models/llm/downloaded_models.json
~~~

If that registry is unavailable, Tater Bench safely scans Tater's llama.cpp and MLX model caches. Vision projectors and speculative draft GGUFs are attached to their target model and are not listed as independent benchmark targets.

When a matching projector is installed, llama.cpp loads it with the target just as Tater does. The core v0.3 score remains text/tool focused; a separately scored vision suite can be added without changing existing scores.

Capability badges come from Tater's downloaded-model registry. New benchmark runs preserve Vision, Video, and Audio flags in their raw result metadata; the published registry snapshot supplies the same fields to older results without changing their measured scores.

## Frozen synthetic Tater runtime

The core v0.3 suite uses a fixed synthetic runtime profile rather than importing prompts or state from the live Tater installation. Every model receives the same Tater identity, date, platform, fake filesystem and hardware state, tool results, and conversation context.

The profile retains every Verba and Core present in the Tater Shop snapshot dated August 14, 2026. All 63 Verbas, 18 built-in tools, 27 Core-provided kernel tools, 12 synthetic Portals, and 10 Cores remain available to the fixture; every Core is also marked running. For routing scenarios, Astraeus receives only the small, explicitly curated set of non-overlapping tools relevant to that turn. A defensive capability-family filter also removes an overlapping tool if one is accidentally added later. This prevents duplicate capabilities such as generic and provider-specific music controls from lowering an otherwise correct model's score. Declared equivalent tools receive full credit, explicitly declared less-preferred tools can receive partial credit, and unrelated or invented tools still lose credit. Chat retains full system-awareness context, and scored behavioral scenarios require the model to use frozen fake facts from Memory, Personal, Guardian, Music, and Tater Tube without exposing internal context labels.

The catalog is stored in `taterbench/fixtures/tater-shop-2026-08-14.json`. Changing the catalog or its prompt selection requires a new dated prompt profile and suite version so results produced under different prompt conditions never share a comparison cohort. Maintainers can create a future snapshot with:

~~~bash
python scripts/snapshot_tater_shop.py --snapshot-date YYYY-MM-DD --output taterbench/fixtures/tater-shop-YYYY-MM-DD.json
~~~

## Run benchmarks

Run one selected model with its baseline and installed speculative variants:

~~~bash
tater-bench run --model Qwen3.8-27B --yes
~~~

Run every discovered llama.cpp and MLX target, one at a time:

~~~bash
tater-bench run --all --yes
~~~

Useful focused runs:

~~~bash
tater-bench run --all --provider llama_cpp --variant baseline --variant mtp --variant dflash --yes
tater-bench run --all --provider mlx_lm --baseline-only --yes
tater-bench run --model Gemma-4 --limit 3 --yes
~~~

Close Tater before an official run. A second loaded model or active generation process can distort memory, model-load, and throughput measurements.

On Linux AMD Ryzen AI Max / Strix Halo systems, Tater Bench mirrors Tater's ROCm defaults by launching the target with `--n-gpu-layers all` and speculative drafts with `--spec-draft-ngl all`. Other systems retain llama.cpp automatic placement. Every result records both settings so the execution path remains auditable. Explicit overrides are available when needed:

~~~bash
export TATER_BENCH_LLAMA_N_GPU_LAYERS=auto
export TATER_BENCH_LLAMA_DRAFT_N_GPU_LAYERS=auto
~~~

Tater's existing `TATER_LLAMA_CPP_N_GPU_LAYERS`, `TATER_LLAMA_CPP_DRAFT_N_GPU_LAYERS`, and `TATER_LLAMA_CPP_STRIX_HALO_FULL_OFFLOAD` settings are also honored. Linux source installations under `~/Tater` and their `.runtime/llama.cpp` engine are discovered automatically.

## Results

Each batch is saved as a versioned JSON file under `results/`. After a run, Tater Bench automatically regenerates:

- `RESULTS.md` — a compact leaderboard rendered directly by GitHub.
- `docs/results.json` — aggregated machine-readable results.
- `docs/index.html` — the Tater-themed ranked dashboard for GitHub Pages.

Regenerate reports at any time:

~~~bash
tater-bench report
~~~

The HTML report is optional. GitHub can display the Markdown leaderboard and raw JSON without GitHub Pages.

New benchmark runs do not overwrite earlier submissions. During report generation, runs with the same hardware type, model configuration, accuracy summary, and graded scenario outcomes are treated as duplicates even when their timing differs. The newest representative is published and the raw batch files remain untouched. Outcome-distinct runs are averaged within their hardware type. The dashboard provides:

- **Overall** for each model and mode's best fitness-qualified hardware result.
- One tab per hardware type, such as Apple M3 Ultra, for that device's averaged results.
- A unique-runs dropdown on each result for every outcome-distinct measurement behind the selected hardware result or device average.
- Tater Ready, Limited, Not Fit, and Mixed Results verdicts with the failed accuracy gates shown directly on each result.
- Base, MTP, DFlash, DSpark, llama.cpp, and MLX filters whenever matching results exist.
- Vision, Video, Audio, or Text-only input badges for every model configuration.
- Sorting by Tater Score, accuracy, generation speed, TTFT, memory, test count, or model name.

Hardware types are grouped by OS family, architecture, CPU, core counts, memory, and GPU configuration. Patch-level OS, Python, and driver changes remain visible in the individual result metadata but do not split otherwise matching devices into separate leaderboard tabs.

## Fair comparisons

Compare generation speed only when hardware, engine version, context size, quantization, and suite version match. The MTP/DFlash/DSpark speedup shown in reports is calculated only against a matching baseline on the same hardware profile and suite.

The Tater Score is calculated only within matching hardware-profile, suite, context, and prompt-profile groups. Its 90-point accuracy component weights Astraeus routing and tool selection most heavily, followed by Thanatos tool execution; its speed, TTFT, and memory components are normalized to the best measured value in that group. Duplicate graded outcomes are removed before scoring and outcome-distinct submissions are averaged within their hardware type. Overall then selects each model and mode's best device result, preferring the stronger fitness verdict and then the higher final score. The winning hardware is shown directly, while every other device remains available in its own tab. Hidden duplicates still count as observations for reproducibility and provisional-status checks.

Fitness is deliberately strict. A readiness cap is applied before outcome-distinct runs are averaged within a hardware type, so a Limited result cannot display 80 or higher and a Not Fit result cannot display 50 or higher. Overall uses the verdict and score from the selected best hardware result instead of penalizing a model for a weaker device. Hardware tabs still expose those weaker results—for example, a model can be Tater Ready overall from Apple while remaining Not Fit in the Halo tab. Results with fewer than two observations on the selected hardware are marked provisional, and older results without all four required accuracy categories remain Unrated.

Speculative decoding is expected to preserve answers, but every speculative pass is graded independently so quality regressions remain visible.

## Privacy

Published profiles omit the hostname, username, home directory, absolute model paths, and Tater credentials. Benchmark prompts use synthetic people, devices, URLs, and tool results. The CLI can show local paths during discovery, but saved result files use repository IDs and filenames only.

## License

Tater Bench is licensed under the GNU Affero General Public License v3.0 or later.
