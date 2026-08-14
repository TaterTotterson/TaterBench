# Tater Bench Results

Accuracy and real-world speed for models running through Tater's llama.cpp and MLX engines.

> Accuracy and speed are intentionally separate. Compare speed only on matching hardware, suite, context, and quantization.

| Model | Engine | Mode | Accuracy | Gen tok/s | TTFT | Load | Peak RSS | Hardware | Suite |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| _No published benchmark runs yet_ | | | | | | | | | |


## Method

Tater Bench uses deterministic Tater-style routing, strict tool-call, synthesis, chat, and Spudex scenarios. Each result records the model, engine, speculative mode, suite version, hardware fingerprint, context, and raw per-scenario response.

MTP, DFlash, and DSpark percentages compare generation speed against the matching baseline run on the same hardware and suite.
