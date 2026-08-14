<p align="center">
  <img src="assets/tater-bench-mascot.png" width="210" alt="Tater mascot">
</p>

# Tater Bench

Tater Bench is a standalone accuracy and performance benchmark for local models installed by [Tater](https://github.com/TaterTotterson/Tater). It reads Tater's model registry, launches each model with its proper llama.cpp or MLX engine, runs versioned Tater-style scenarios, and stores publish-safe results.

Tater does not need to be running, and Tater Bench never changes Tater's model files or settings.

## What it measures

Tater Bench keeps capability and performance separate:

- **Accuracy:** Astraeus routing, ordered planning, Thanatos tool selection and arguments, normal chat, Hermes result synthesis, and Spudex action decisions.
- **Speed:** model load time, time to first token, prompt speed, generation speed, complete scenario latency, and peak engine RSS.
- **Speculation:** llama.cpp targets run once without speculative decoding and again with every compatible MTP, DFlash, or DSpark draft found beside the target GGUF.
- **Hardware:** OS, CPU, architecture, core count, memory, GPU/backend, engine version, and an anonymous hardware fingerprint.

MLX models currently run a baseline pass. MTP, DFlash, and DSpark are tested on compatible llama.cpp GGUF targets because those are the speculative methods Tater exposes for that engine.

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

When a matching projector is installed, llama.cpp loads it with the target just as Tater does. The core v0.1 score remains text/tool focused; a separately scored vision suite can be added without changing existing scores.

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

## Results

Each batch is saved as a versioned JSON file under `results/`. After a run, Tater Bench automatically regenerates:

- `RESULTS.md` — the primary leaderboard rendered directly by GitHub.
- `docs/results.json` — aggregated machine-readable results.
- `docs/index.html` — an optional Tater-themed static dashboard.

Regenerate reports at any time:

~~~bash
tater-bench report
~~~

The HTML report is optional. GitHub can display the Markdown leaderboard and raw JSON without GitHub Pages.

## Fair comparisons

Compare generation speed only when hardware, engine version, context size, quantization, and suite version match. The MTP/DFlash/DSpark speedup shown in reports is calculated only against a matching baseline on the same hardware and suite.

Accuracy is reported as a separate 0–100 score. Speculative decoding is expected to preserve answers, but every speculative pass is graded independently so quality regressions remain visible.

## Privacy

Published profiles omit the hostname, username, home directory, absolute model paths, and Tater credentials. Benchmark prompts use synthetic people, devices, URLs, and tool results. The CLI can show local paths during discovery, but saved result files use repository IDs and filenames only.

## License

Tater Bench is licensed under the GNU Affero General Public License v3.0 or later.
