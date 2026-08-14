from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from .hardware import bytes_label


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


def aggregate_payload(batches: list[dict[str, Any]]) -> dict[str, Any]:
    rows = flatten_runs(batches)
    public_rows: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["speedup_percent"] = round(_speedup(row, rows), 2)
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
    rows = aggregate.get("runs") or []
    lines = [
        "# Tater Bench Results",
        "",
        "Accuracy and real-world speed for models running through Tater's llama.cpp and MLX engines.",
        "",
        "> Accuracy and speed are intentionally separate. Compare speed only on matching hardware, suite, context, and quantization.",
        "",
        "| Model | Engine | Mode | Accuracy | Gen tok/s | TTFT | Load | Peak RSS | Hardware | Suite |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
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
        lines.extend(["| _No published benchmark runs yet_ | | | | | | | | | |", ""])
    lines.extend(
        [
            "",
            "## Method",
            "",
            "Tater Bench uses deterministic Tater-style routing, strict tool-call, synthesis, chat, and Spudex scenarios. Each result records the model, engine, speculative mode, suite version, hardware fingerprint, context, and raw per-scenario response.",
            "",
            "MTP, DFlash, and DSpark percentages compare generation speed against the matching baseline run on the same hardware and suite.",
            "",
        ]
    )
    return "\n".join(lines)


def render_html(aggregate: dict[str, Any]) -> str:
    rows = aggregate.get("runs") or []
    cards: list[str] = []
    for run in rows:
        model = run.get("model") or {}
        variant = run.get("variant") or {}
        perf = run.get("performance") or {}
        hardware = run.get("hardware") or {}
        categories = (run.get("accuracy") or {}).get("categories") or {}
        category_html = "".join(
            f'<span><b>{html.escape(str(name).replace("_", " ").title())}</b>{_fmt(score)}%</span>'
            for name, score in categories.items()
        )
        cards.append(
            f'''<article class="result-card" data-engine="{html.escape(str(model.get("provider") or ""))}" data-mode="{html.escape(str(variant.get("name") or "baseline"))}">
              <div class="card-head"><div><p class="eyebrow">{html.escape(str((run.get("engine") or {}).get("engine") or model.get("provider") or ""))} · {html.escape(str(variant.get("name") or "baseline").upper())}</p><h2>{html.escape(str(model.get("repo_id") or model.get("label") or "Unknown model"))}</h2><p>{html.escape(str(model.get("filename") or ""))}</p></div><div class="score">{_fmt((run.get("accuracy") or {}).get("score"), 1)}<small>accuracy</small></div></div>
              <div class="metrics"><span><b>{_fmt(perf.get("median_generation_tokens_per_second"))}</b> tok/s</span><span><b>{_fmt(perf.get("median_ttft_seconds"))}s</b> TTFT</span><span><b>{_fmt(run.get("load_seconds"))}s</b> load</span><span><b>{bytes_label(run.get("peak_rss_bytes") or 0)}</b> peak RSS</span></div>
              <div class="categories">{category_html}</div>
              <footer>{html.escape(str(hardware.get("cpu") or hardware.get("architecture") or "Unknown hardware"))} · {bytes_label(hardware.get("memory_bytes") or 0)}</footer>
            </article>'''
        )
    cards_html = "\n".join(cards) or '<div class="empty">No benchmark results have been published yet.</div>'
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Tater Bench</title>
<meta name="description" content="Accuracy and speed results for local models running through Tater.">
<style>
:root{{--ink:#f7f7f5;--muted:#aaa9a2;--panel:#151515;--line:#31302c;--orange:#ff7a18;--orange2:#ffb13b;--blue:#42b8ff}}*{{box-sizing:border-box}}body{{margin:0;background:radial-gradient(circle at 75% 0,#2b1608 0,transparent 34%),#090909;color:var(--ink);font:15px/1.55 Inter,ui-sans-serif,system-ui,sans-serif}}.shell{{width:min(1180px,calc(100% - 32px));margin:auto;padding:34px 0 70px}}header{{display:grid;grid-template-columns:1fr auto;align-items:center;gap:28px;border-bottom:1px solid var(--line);padding-bottom:28px}}.brand{{display:flex;align-items:center;gap:18px}}.brand img{{width:88px;height:88px;border-radius:50%;border:2px solid var(--orange);box-shadow:0 0 34px #ff7a1840}}.eyebrow{{margin:0 0 5px;color:var(--orange2);font-size:12px;font-weight:800;letter-spacing:.13em;text-transform:uppercase}}h1{{margin:0;font-size:clamp(36px,7vw,72px);line-height:.95;letter-spacing:-.055em}}.lede{{max-width:620px;color:var(--muted);font-size:17px}}.summary{{text-align:right}}.summary b{{display:block;color:var(--orange);font-size:34px}}.controls{{display:flex;gap:10px;flex-wrap:wrap;margin:26px 0}}button{{border:1px solid var(--line);border-radius:999px;background:#111;color:var(--muted);padding:9px 15px;cursor:pointer}}button.active,button:hover{{border-color:var(--orange);color:white;background:#271509}}main{{display:grid;gap:16px}}.result-card{{background:linear-gradient(145deg,#191919,#101010);border:1px solid var(--line);border-radius:22px;padding:22px;box-shadow:0 18px 60px #0006}}.card-head{{display:flex;justify-content:space-between;gap:18px}}h2{{margin:0;font-size:21px}}.card-head p:not(.eyebrow){{margin:4px 0;color:var(--muted)}}.score{{min-width:94px;text-align:center;color:var(--orange);font-size:36px;font-weight:900;line-height:1}}.score small{{display:block;color:var(--muted);font-size:10px;letter-spacing:.12em;text-transform:uppercase;margin-top:7px}}.metrics,.categories{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-top:19px}}.metrics span,.categories span{{background:#0c0c0c;border:1px solid #272622;border-radius:13px;padding:11px;color:var(--muted)}}.metrics b,.categories b{{display:block;color:var(--ink);font-size:15px}}footer{{margin-top:18px;color:#777;font-size:12px}}.empty{{padding:70px;text-align:center;border:1px dashed var(--line);border-radius:20px;color:var(--muted)}}@media(max-width:720px){{header{{grid-template-columns:1fr}}.summary{{text-align:left}}.metrics,.categories{{grid-template-columns:repeat(2,1fr)}}.brand img{{width:68px;height:68px}}.card-head{{align-items:flex-start}}}}
</style></head><body><div class="shell"><header><div><div class="brand"><img src="tater-mascot.png" alt="Tater mascot"><div><p class="eyebrow">Local model field test</p><h1>Tater Bench</h1></div></div><p class="lede">Tater-style accuracy and real measured speed across llama.cpp, MLX, baseline, MTP, and DFlash.</p></div><div class="summary"><b>{len(rows)}</b>published runs</div></header><nav class="controls" aria-label="Filter results"><button class="active" data-filter="all">All</button><button data-filter="baseline">Baseline</button><button data-filter="mtp">MTP</button><button data-filter="dflash">DFlash</button><button data-filter="llama_cpp">llama.cpp</button><button data-filter="mlx_lm">MLX</button></nav><main>{cards_html}</main></div>
<script>document.querySelectorAll('button[data-filter]').forEach(b=>b.addEventListener('click',()=>{{document.querySelectorAll('button').forEach(x=>x.classList.remove('active'));b.classList.add('active');const f=b.dataset.filter;document.querySelectorAll('.result-card').forEach(c=>{{c.hidden=!(f==='all'||c.dataset.mode===f||c.dataset.engine===f)}})}}));</script></body></html>'''


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
