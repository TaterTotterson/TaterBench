from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from .hardware import bytes_label


TATER_SCORE_WEIGHTS = {
    "accuracy": 70.0,
    "generation_speed": 20.0,
    "ttft": 5.0,
    "memory": 5.0,
}


def load_batches(results_dir: str | Path) -> list[dict[str, Any]]:
    root = Path(results_dir)
    batches: list[dict[str, Any]] = []
    if not root.is_dir():
        return batches
    for path in sorted(root.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict) and isinstance(payload.get("runs"), list):
            payload["_source"] = path.name
            batches.append(payload)
    return batches


def flatten_runs(batches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for batch in batches:
        hardware = batch.get("hardware") if isinstance(batch.get("hardware"), dict) else {}
        for run in batch.get("runs") or []:
            if not isinstance(run, dict):
                continue
            rows.append({**run, "hardware": hardware, "result_source": batch.get("_source", "")})
    rows.sort(key=lambda row: str(row.get("finished_at") or ""), reverse=True)
    return rows


def _speedup(run: dict[str, Any], rows: list[dict[str, Any]]) -> float:
    variant = str((run.get("variant") or {}).get("name") or "baseline")
    if variant == "baseline":
        return 0.0
    model_id = str((run.get("model") or {}).get("id") or "")
    hardware_id = str((run.get("hardware") or {}).get("hardware_id") or "")
    suite_version = str((run.get("suite") or {}).get("version") or "")
    current = float((run.get("performance") or {}).get("median_generation_tokens_per_second") or 0.0)
    for candidate in rows:
        if str((candidate.get("variant") or {}).get("name") or "") != "baseline":
            continue
        if str((candidate.get("model") or {}).get("id") or "") != model_id:
            continue
        if str((candidate.get("hardware") or {}).get("hardware_id") or "") != hardware_id:
            continue
        if str((candidate.get("suite") or {}).get("version") or "") != suite_version:
            continue
        baseline = float((candidate.get("performance") or {}).get("median_generation_tokens_per_second") or 0.0)
        return ((current / baseline) - 1.0) * 100.0 if current and baseline else 0.0
    return 0.0


def _comparison_key(run: dict[str, Any]) -> tuple[str, str, int, str]:
    hardware = run.get("hardware") or {}
    suite = run.get("suite") or {}
    configuration = run.get("configuration") or {}
    return (
        str(hardware.get("hardware_id") or ""),
        str(suite.get("version") or ""),
        int(configuration.get("context_size") or 0),
        str(configuration.get("prompt_profile") or ""),
    )


def _positive_metric(run: dict[str, Any], name: str) -> float:
    if name == "memory":
        value = run.get("peak_rss_bytes")
    else:
        value = (run.get("performance") or {}).get(name)
    try:
        number = float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return number if number > 0 else 0.0


def _tater_score(run: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, float]:
    cohort = [
        candidate
        for candidate in rows
        if str(candidate.get("status") or "complete") == "complete"
        and _comparison_key(candidate) == _comparison_key(run)
    ]
    accuracy = max(0.0, min(100.0, float((run.get("accuracy") or {}).get("score") or 0.0)))
    speed = _positive_metric(run, "median_generation_tokens_per_second")
    ttft = _positive_metric(run, "median_ttft_seconds")
    memory = _positive_metric(run, "memory")
    cohort_speeds = [_positive_metric(candidate, "median_generation_tokens_per_second") for candidate in cohort]
    cohort_ttfts = [_positive_metric(candidate, "median_ttft_seconds") for candidate in cohort]
    cohort_memory = [_positive_metric(candidate, "memory") for candidate in cohort]
    max_speed = max(cohort_speeds or [0.0])
    min_ttft = min((value for value in cohort_ttfts if value > 0), default=0.0)
    min_memory = min((value for value in cohort_memory if value > 0), default=0.0)
    components = {
        "accuracy": (accuracy / 100.0) * TATER_SCORE_WEIGHTS["accuracy"],
        "generation_speed": (speed / max_speed) * TATER_SCORE_WEIGHTS["generation_speed"] if max_speed else 0.0,
        "ttft": (min_ttft / ttft) * TATER_SCORE_WEIGHTS["ttft"] if ttft and min_ttft else 0.0,
        "memory": (min_memory / memory) * TATER_SCORE_WEIGHTS["memory"] if memory and min_memory else 0.0,
    }
    rounded = {name: round(value, 2) for name, value in components.items()}
    rounded["total"] = round(sum(components.values()), 2)
    return rounded


def aggregate_payload(batches: list[dict[str, Any]]) -> dict[str, Any]:
    rows = flatten_runs(batches)
    public_rows: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["speedup_percent"] = round(_speedup(row, rows), 2)
        score = _tater_score(row, rows)
        item["tater_score"] = score["total"]
        item["tater_score_components"] = {name: value for name, value in score.items() if name != "total"}
        public_rows.append(item)
    timestamps = [str(batch.get("created_at") or "") for batch in batches if str(batch.get("created_at") or "")]
    return {
        "generated_at": max(timestamps) if timestamps else "",
        "run_count": len(public_rows),
        "runs": public_rows,
    }


def _fmt(value: Any, digits: int = 2) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def render_markdown(aggregate: dict[str, Any]) -> str:
    rows = sorted(
        aggregate.get("runs") or [],
        key=lambda run: (
            -float(run.get("tater_score") or 0.0),
            str((run.get("model") or {}).get("repo_id") or ""),
        ),
    )
    lines = [
        "# Tater Bench Results",
        "",
        "Accuracy and real-world speed for models running through Tater's llama.cpp and MLX engines.",
        "",
        "> Raw accuracy and speed stay visible beside the composite Tater Score. Compare speed only on matching hardware, suite, context, and quantization.",
        "",
        "| Model | Engine | Mode | Tater Score | Accuracy | Gen tok/s | TTFT | Load | Peak RSS | Hardware | Suite |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for run in rows:
        model = run.get("model") or {}
        variant = run.get("variant") or {}
        performance = run.get("performance") or {}
        hardware = run.get("hardware") or {}
        engine = run.get("engine") or {}
        mode = str(variant.get("name") or "baseline").upper()
        speedup = float(run.get("speedup_percent") or 0.0)
        if mode != "BASELINE" and speedup:
            mode += f" ({speedup:+.1f}%)"
        hardware_label = f"{hardware.get('cpu') or hardware.get('architecture')} / {bytes_label(hardware.get('memory_bytes') or 0)}"
        label = str(model.get("repo_id") or model.get("label") or model.get("filename") or "Unknown").replace("|", "\\|")
        lines.append(
            "| "
            + " | ".join(
                [
                    label,
                    str(engine.get("engine") or model.get("provider") or ""),
                    mode,
                    _fmt(run.get("tater_score")),
                    _fmt((run.get("accuracy") or {}).get("score")) + "%",
                    _fmt(performance.get("median_generation_tokens_per_second")),
                    _fmt(performance.get("median_ttft_seconds")) + "s",
                    _fmt(run.get("load_seconds")) + "s",
                    bytes_label(run.get("peak_rss_bytes") or 0),
                    hardware_label.replace("|", "\\|"),
                    str((run.get("suite") or {}).get("version") or ""),
                ]
            )
            + " |"
        )
    if not rows:
        lines.extend(["| _No published benchmark runs yet_ | | | | | | | | | | |", ""])
    lines.extend(
        [
            "",
            "## Method",
            "",
            "Tater Score is a 100-point composite: 70 points for task accuracy, 20 for generation speed, 5 for time to first token, and 5 for peak-memory efficiency. Performance and efficiency are normalized within matching hardware, suite, context, and prompt profile.",
            "",
            "Tater Bench uses deterministic Tater-style routing, strict tool-call, synthesis, chat, and Spudex scenarios. Each result records the model, engine, speculative mode, suite version, hardware fingerprint, context, and raw per-scenario response.",
            "",
            "MTP, DFlash, and DSpark percentages compare generation speed against the matching baseline run on the same hardware and suite.",
            "",
        ]
    )
    return "\n".join(lines)


def _chart_label(repo_id: str, provider: str) -> str:
    identity = repo_id.lower()
    if "qwen3.8-27b" in identity:
        return "Qwen 3.8 27B"
    if "qwen3.6-35b" in identity:
        return "Qwen 3.6 35B A3B"
    if "nemotron-3.5" in identity:
        return "Nemotron 3.5 30B A3B"
    if "gemma-4-26b" in identity:
        return "Gemma 4 26B" + (" MLX" if provider == "mlx_lm" else "")
    return repo_id.rsplit("/", 1)[-1][:36]


def render_html(aggregate: dict[str, Any]) -> str:
    rows = sorted(
        aggregate.get("runs") or [],
        key=lambda run: (
            -float(run.get("tater_score") or 0.0),
            -float((run.get("accuracy") or {}).get("score") or 0.0),
            -float((run.get("performance") or {}).get("median_generation_tokens_per_second") or 0.0),
            str((run.get("model") or {}).get("repo_id") or ""),
        ),
    )
    cards: list[str] = []
    leaderboard_items: list[str] = []
    present_modes: set[str] = set()
    present_engines: set[str] = set()
    for index, run in enumerate(rows):
        rank = index + 1
        model = run.get("model") or {}
        variant = run.get("variant") or {}
        performance = run.get("performance") or {}
        hardware = run.get("hardware") or {}
        engine = run.get("engine") or {}
        categories = (run.get("accuracy") or {}).get("categories") or {}
        provider = str(model.get("provider") or "")
        mode = str(variant.get("name") or "baseline").lower()
        repo_id = str(model.get("repo_id") or model.get("label") or "Unknown model")
        card_id = f"result-{rank}"
        tater_score = float(run.get("tater_score") or 0.0)
        score_components = run.get("tater_score_components") or {}
        category_html = "".join(
            f'<span><b>{html.escape(str(name).replace("_", " ").title())}</b>{_fmt(score)}%</span>'
            for name, score in categories.items()
        )
        cards.append(
            f'''<article id="{card_id}" class="result-card" data-engine="{html.escape(provider)}" data-mode="{html.escape(mode)}">
              <div class="card-head"><div><p class="eyebrow"><span class="detail-rank">#{rank}</span> · {html.escape(str(engine.get("engine") or provider))} · {html.escape(mode.upper())}</p><h2>{html.escape(repo_id)}</h2><p>{html.escape(str(model.get("filename") or ""))}</p></div><div class="score">{_fmt(tater_score, 1)}<small>Tater score</small></div></div>
              <div class="metrics"><span><b>{_fmt((run.get("accuracy") or {}).get("score"), 1)}%</b> accuracy</span><span><b>{_fmt(performance.get("median_generation_tokens_per_second"))}</b> tok/s</span><span><b>{_fmt(performance.get("median_ttft_seconds"))}s</b> TTFT</span><span><b>{_fmt(run.get("load_seconds"))}s</b> load</span><span><b>{bytes_label(run.get("peak_rss_bytes") or 0)}</b> peak RSS</span></div>
              <div class="categories">{category_html}</div>
              <footer>{html.escape(str(hardware.get("cpu") or hardware.get("architecture") or "Unknown hardware"))} · {bytes_label(hardware.get("memory_bytes") or 0)} · Score mix: {_fmt(score_components.get("accuracy"), 1)} accuracy + {_fmt(score_components.get("generation_speed"), 1)} speed + {_fmt(score_components.get("ttft"), 1)} TTFT + {_fmt(score_components.get("memory"), 1)} memory</footer>
            </article>'''
        )
        accuracy = float((run.get("accuracy") or {}).get("score") or 0.0)
        speed = float(performance.get("median_generation_tokens_per_second") or 0.0)
        if str(run.get("status") or "complete") == "complete" and accuracy > 0 and speed > 0:
            label = _chart_label(repo_id, provider)
            engine_label = str(engine.get("engine") or provider)
            leader_class = " is-leader" if rank == 1 else ""
            leaderboard_items.append(
                f'''<a class="score-entry{leader_class}" href="#{card_id}" data-engine="{html.escape(provider)}" data-mode="{html.escape(mode)}" data-label="{html.escape(label)}, {tater_score:.1f} Tater Score" style="--bar-score:{tater_score}%;--bar-color:var(--mode-{html.escape(mode)})" aria-label="Rank {rank}: {html.escape(label)}, {tater_score:.1f} Tater Score">
                  <span class="score-rank">#{rank}</span><span class="score-bar-stage"><span class="score-bar"><b>{tater_score:.1f}</b></span></span><span class="score-copy"><strong>{html.escape(label)}</strong><small>{html.escape(engine_label)} · {html.escape(mode.upper())}</small></span>
                </a>'''
            )
            present_modes.add(mode)
            present_engines.add(provider)
    cards_html = "\n".join(cards) or '<div class="empty">No benchmark results have been published yet.</div>'
    leaderboard_html = "\n".join(leaderboard_items) or '<div class="empty">No completed benchmark runs yet.</div>'
    mode_labels = {"baseline": "Baseline", "mtp": "MTP", "dflash": "DFlash", "dspark": "DSpark"}
    engine_labels = {"llama_cpp": "llama.cpp", "mlx_lm": "MLX"}
    filter_parts = ['<button class="active" type="button" aria-pressed="true" data-filter="all">All</button>']
    filter_parts.extend(
        f'<button type="button" aria-pressed="false" data-filter="{html.escape(mode)}">{label}</button>'
        for mode, label in mode_labels.items()
        if mode in present_modes
    )
    filter_parts.extend(
        f'<button type="button" aria-pressed="false" data-filter="{html.escape(engine)}">{label}</button>'
        for engine, label in engine_labels.items()
        if engine in present_engines
    )
    filters_html = "".join(filter_parts)
    template = '''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Tater Bench</title>
<meta name="description" content="Ranked accuracy and speed results for local models running through Tater.">
<style>
:root{--ink:#f7f7f5;--muted:#aaa9a2;--line:#31302c;--orange:#ff7a18;--orange2:#ffb13b;--blue:#42b8ff;--purple:#b38cff;--green:#56d68b;--gold:#ffd166;--grid:#2b2a27;--mode-baseline:var(--blue);--mode-mtp:var(--orange);--mode-dflash:var(--purple);--mode-dspark:var(--green)}
*{box-sizing:border-box}[hidden]{display:none!important}html{scroll-behavior:smooth}body{margin:0;background:radial-gradient(circle at 75% 0,#2b1608 0,transparent 34%),#090909;color:var(--ink);font:15px/1.55 Inter,ui-sans-serif,system-ui,sans-serif}.shell{width:min(1180px,calc(100% - 32px));margin:auto;padding:34px 0 70px}header{display:grid;grid-template-columns:1fr auto;align-items:center;gap:28px;border-bottom:1px solid var(--line);padding-bottom:28px}.brand{display:flex;align-items:center;gap:18px}.brand img{width:88px;height:88px;border-radius:50%;border:2px solid var(--orange);box-shadow:0 0 34px #ff7a1840}.eyebrow{margin:0 0 5px;color:var(--orange2);font-size:12px;font-weight:800;letter-spacing:.13em;text-transform:uppercase}h1{margin:0;font-size:clamp(36px,7vw,72px);line-height:.95;letter-spacing:-.055em}.lede{max-width:620px;color:var(--muted);font-size:17px}.summary{text-align:right}.summary b{display:block;color:var(--orange);font-size:34px}.controls{display:flex;gap:10px;flex-wrap:wrap;margin:26px 0}button{border:1px solid var(--line);border-radius:999px;background:#111;color:var(--muted);padding:9px 15px;cursor:pointer}button.active,button:hover{border-color:var(--orange);color:#fff;background:#271509}.leaderboard{margin:10px 0 30px;padding:24px;background:linear-gradient(145deg,#171717,#0e0e0e);border:1px solid var(--line);border-radius:24px}.chart-heading{display:flex;align-items:flex-end;justify-content:space-between;gap:18px;margin-bottom:14px}.chart-heading h2{margin:0;font-size:clamp(23px,4vw,34px);letter-spacing:-.025em}.chart-heading p{margin:4px 0 0;color:var(--muted)}.score-formula{max-width:630px;text-align:right;color:var(--muted);font-size:12px}.score-formula b{color:var(--ink)}.score-plot{position:relative;padding-left:44px;margin-top:22px}.score-grid{position:absolute;z-index:0;left:44px;right:0;top:0;height:320px}.score-grid span{position:absolute;left:0;right:0;bottom:var(--grid-position);height:1px;background:var(--grid)}.score-grid i{position:absolute;right:calc(100% + 10px);top:-8px;color:var(--muted);font-size:10px;font-style:normal}.score-bars{position:relative;z-index:1;display:grid;grid-template-columns:repeat(var(--bar-count),minmax(72px,1fr));gap:10px;min-height:430px}.score-entry{position:relative;display:grid;grid-template-rows:320px auto;text-decoration:none;color:var(--ink);min-width:0}.score-rank{position:absolute;z-index:2;top:6px;left:50%;transform:translateX(-50%);color:#ffffffb8;font-size:11px;font-weight:900}.score-bar-stage{display:flex;align-items:flex-end;justify-content:center;height:320px}.score-bar{display:flex;align-items:flex-start;justify-content:center;width:min(64px,78%);height:var(--bar-score);min-height:34px;padding-top:13px;border-radius:7px 7px 2px 2px;background:linear-gradient(180deg,color-mix(in srgb,var(--bar-color),white 13%),var(--bar-color));box-shadow:0 8px 30px color-mix(in srgb,var(--bar-color),transparent 75%);transition:filter .18s,transform .18s}.score-bar b{font-size:15px;color:#fff;text-shadow:0 1px 4px #0008}.score-copy{display:block;padding:12px 3px 0;text-align:center;line-height:1.25}.score-copy strong{display:block;font-size:12px;overflow-wrap:anywhere}.score-copy small{display:block;margin-top:5px;color:var(--muted);font-size:10px}.score-entry:hover .score-bar,.score-entry:focus .score-bar{filter:brightness(1.18);transform:translateY(-3px)}.score-entry:focus{outline:2px solid var(--orange2);outline-offset:3px;border-radius:5px}.score-entry.is-leader .score-bar{background:linear-gradient(180deg,#ffe5a3,var(--gold) 34%,var(--orange));box-shadow:0 0 32px #ffb13b66}.score-entry.is-leader .score-rank{color:var(--gold)}.score-note{margin:10px 0 0;color:var(--muted);font-size:12px}.results-heading{margin:8px 0 14px;font-size:25px}main{display:grid;gap:16px;min-width:0}.result-card{min-width:0;scroll-margin-top:16px;background:linear-gradient(145deg,#191919,#101010);border:1px solid var(--line);border-radius:22px;padding:22px;box-shadow:0 18px 60px #0006}.result-card:target{border-color:var(--orange);box-shadow:0 0 0 1px var(--orange),0 18px 60px #0006}.card-head{display:flex;justify-content:space-between;gap:18px;min-width:0}.card-head>div{min-width:0}.result-card h2{margin:0;font-size:21px;overflow-wrap:anywhere}.card-head p:not(.eyebrow){margin:4px 0;color:var(--muted);overflow-wrap:anywhere}.score{min-width:94px;text-align:center;color:var(--orange);font-size:36px;font-weight:900;line-height:1}.score small{display:block;color:var(--muted);font-size:10px;letter-spacing:.12em;text-transform:uppercase;margin-top:7px}.metrics,.categories{display:grid;grid-template-columns:repeat(auto-fit,minmax(145px,1fr));gap:10px;margin-top:19px;min-width:0}.metrics span,.categories span{min-width:0;background:#0c0c0c;border:1px solid #272622;border-radius:13px;padding:11px;color:var(--muted)}.metrics b,.categories b{display:block;color:var(--ink);font-size:15px}footer{margin-top:18px;color:#777;font-size:12px}.empty{padding:70px;text-align:center;border:1px dashed var(--line);border-radius:20px;color:var(--muted)}
@media(max-width:820px){header{grid-template-columns:1fr}.summary{text-align:left}.leaderboard{padding:18px}.chart-heading{align-items:flex-start;flex-direction:column}.score-formula{text-align:left}.score-plot{padding-left:0}.score-grid{display:none}.score-bars{display:flex;flex-direction:column;gap:13px;min-height:0}.score-entry{display:grid;grid-template-columns:minmax(106px,1fr) minmax(140px,2fr);grid-template-rows:auto;height:auto;gap:12px;align-items:center}.score-copy{position:relative;grid-column:1;grid-row:1;padding:0 0 0 27px;text-align:left}.score-bar-stage{grid-column:2;grid-row:1;height:30px;justify-content:flex-start;background:#090909;border:1px solid var(--line);border-radius:6px;overflow:hidden}.score-bar{align-items:center;justify-content:flex-end;width:var(--bar-score);height:100%;min-width:48px;padding:0 8px;border-radius:4px}.score-rank{top:0;left:0;right:auto;transform:none;color:#ffffff88}.score-entry.is-leader .score-rank{color:var(--gold)}.brand img{width:68px;height:68px}.card-head{align-items:flex-start}}@media(max-width:430px){.shell{width:min(100% - 20px,1180px)}.leaderboard{padding:14px}.score-entry{grid-template-columns:minmax(94px,1fr) minmax(128px,1.55fr);gap:9px}.card-head{flex-direction:column}.score{text-align:left}.metrics,.categories{grid-template-columns:1fr 1fr}}
</style></head><body><div class="shell"><header><div><div class="brand"><img src="tater-mascot.png" alt="Tater mascot"><div><p class="eyebrow">Local model field test</p><h1>Tater Bench</h1></div></div><p class="lede">Tater-style accuracy and real measured speed across llama.cpp, MLX, and speculative decoding configurations.</p></div><div class="summary"><b>__RUN_COUNT__</b>published runs</div></header><nav class="controls" aria-label="Filter results">__FILTERS__</nav>
<section class="leaderboard" aria-labelledby="leaderboard-title"><div class="chart-heading"><div><p class="eyebrow">Tater leaderboard</p><h2 id="leaderboard-title">Best models for Tater</h2><p>One score ranks every tested model and decoding configuration.</p></div><p class="score-formula"><b>100 points:</b> 70 accuracy · 20 generation speed · 5 TTFT · 5 memory efficiency</p></div><div class="score-plot"><div class="score-grid" aria-hidden="true"><span style="--grid-position:0%"><i>0</i></span><span style="--grid-position:20%"><i>20</i></span><span style="--grid-position:40%"><i>40</i></span><span style="--grid-position:60%"><i>60</i></span><span style="--grid-position:80%"><i>80</i></span><span style="--grid-position:100%"><i>100</i></span></div><div class="score-bars" id="score-bars" style="--bar-count:__BAR_COUNT__">__LEADERBOARD__</div></div><p class="score-note">Higher is better. Speed, TTFT, and memory are normalized only against runs with matching hardware, suite, context, and prompt profile. Select a bar for the full result.</p></section>
<h2 class="results-heading" id="results-heading">All results — best to lowest Tater Score</h2><main>__CARDS__</main></div>
<script>
(() => {
  const buttons = [...document.querySelectorAll('button[data-filter]')];
  const scoreBars = document.getElementById('score-bars');
  const leaderboardTitle = document.getElementById('leaderboard-title');
  const resultsHeading = document.getElementById('results-heading');
  function applyFilter(button) {
    const activeFilter = button.dataset.filter;
    const filterLabel = button.textContent.trim();
    buttons.forEach(item => { const selected = item === button; item.classList.toggle('active', selected); item.setAttribute('aria-pressed', selected ? 'true' : 'false'); });
    let visibleRank = 0;
    document.querySelectorAll('.score-entry').forEach(entry => {
      const visible = activeFilter === 'all' || entry.dataset.mode === activeFilter || entry.dataset.engine === activeFilter;
      entry.hidden = !visible;
      if (visible) {
        visibleRank += 1;
        entry.querySelector('.score-rank').textContent = `#${visibleRank}`;
        entry.setAttribute('aria-label', `Rank ${visibleRank}: ${entry.dataset.label}`);
        entry.classList.toggle('is-leader', visibleRank === 1);
      } else {
        entry.classList.remove('is-leader');
      }
    });
    scoreBars.style.setProperty('--bar-count', Math.max(visibleRank, 1));
    let resultRank = 0;
    document.querySelectorAll('.result-card').forEach(card => {
      const visible = activeFilter === 'all' || card.dataset.mode === activeFilter || card.dataset.engine === activeFilter;
      card.hidden = !visible;
      if (visible) { resultRank += 1; card.querySelector('.detail-rank').textContent = `#${resultRank}`; }
    });
    leaderboardTitle.textContent = activeFilter === 'all' ? 'Best models for Tater' : `Best ${filterLabel} results for Tater`;
    resultsHeading.textContent = `${activeFilter === 'all' ? 'All' : filterLabel} results — best to lowest Tater Score`;
  }
  buttons.forEach(button => button.addEventListener('click', () => applyFilter(button)));
})();
</script></body></html>'''
    return (
        template.replace("__RUN_COUNT__", str(len(rows)))
        .replace("__FILTERS__", filters_html)
        .replace("__BAR_COUNT__", str(max(1, len(leaderboard_items))))
        .replace("__LEADERBOARD__", leaderboard_html)
        .replace("__CARDS__", cards_html)
    )


def write_reports(
    *,
    results_dir: str | Path = "results",
    markdown_path: str | Path = "RESULTS.md",
    docs_dir: str | Path = "docs",
) -> dict[str, Path]:
    aggregate = aggregate_payload(load_batches(results_dir))
    markdown = Path(markdown_path)
    docs = Path(docs_dir)
    docs.mkdir(parents=True, exist_ok=True)
    markdown.write_text(render_markdown(aggregate), encoding="utf-8")
    json_path = docs / "results.json"
    html_path = docs / "index.html"
    json_path.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    html_path.write_text(render_html(aggregate), encoding="utf-8")
    return {"markdown": markdown, "json": json_path, "html": html_path}
