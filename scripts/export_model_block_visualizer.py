"""Export Interactive Model Block-Graph Visualizer HTML & JSON (`Wave 44–45`).

Usage:
    /home/festivity/agy/QuantumDLBlocks/.venv/bin/python scripts/export_model_block_visualizer.py \\
        --preset w44_compact_r2_d128c_L2_b32 \\
        --html-out architecture_visualizer.html \\
        --json-out .cache/w44_block_graph.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from qdlblocks.benchmarks.model_block_visualizer import (
    DEFAULT_PRESET_KEY,
    build_architecture_visualizer_payload,
)

# ==============================================================================
# Top-of-File Named Constants (Rule 7 of docs/SCRIPT_BEST_PRACTICES.md)
# ==============================================================================
DEFAULT_HTML_OUT: str = "architecture_visualizer.html"
DEFAULT_JSON_OUT: str = ".cache/w44_block_graph.json"

HTML_TEMPLATE: str = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>QuantumDLBlocks — Block Architecture Visualizer (w44_compact_r2_d128c_L2_b32)</title>
  <style>
    :root {
      --bg-main: #070b14;
      --bg-panel: #0f1626;
      --bg-card: #131c31;
      --bg-card-hover: #192542;
      --border: #26334d;
      --text-main: #e6edf3;
      --text-muted: #8b949e;
      --accent-cyan: #38bdf8;
      --accent-green: #10b981;
      --accent-amber: #f59e0b;
      --accent-purple: #a855f7;
      --accent-pink: #ec4899;
      --accent-blue: #3b82f6;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      padding: 16px 20px 36px 20px;
      background: var(--bg-main);
      color: var(--text-main);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      font-size: 13px;
      line-height: 1.45;
    }
    .top-header {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      flex-wrap: wrap;
      gap: 12px;
      padding: 14px 18px;
      background: linear-gradient(135deg, rgba(19,28,49,0.96), rgba(15,22,38,0.96));
      border: 1px solid var(--border);
      border-radius: 10px;
      margin-bottom: 14px;
    }
    .title-main {
      font-size: 17px;
      font-weight: 800;
      color: #f0f6fc;
      letter-spacing: -0.2px;
    }
    .subtitle-main {
      color: var(--accent-cyan);
      font-size: 12px;
      font-weight: 600;
      margin-top: 3px;
      font-family: "JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, monospace;
    }
    .controls-bar {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
      padding: 12px 16px;
      background: var(--bg-panel);
      border: 1px solid var(--border);
      border-radius: 10px;
      margin-bottom: 14px;
    }
    .ctrl-label {
      font-size: 11px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      color: var(--text-muted);
    }
    .btn-pill {
      background: #162038;
      color: #c9d1d9;
      border: 1px solid #2e3e5e;
      border-radius: 6px;
      padding: 6px 11px;
      font-size: 12px;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.15s ease;
    }
    .btn-pill:hover {
      background: #1e2c4c;
      color: #fff;
    }
    .btn-pill.active {
      background: rgba(56, 189, 248, 0.18);
      border-color: var(--accent-cyan);
      color: #fff;
      box-shadow: 0 0 10px rgba(56, 189, 248, 0.25);
    }
    .highway-banner {
      padding: 10px 15px;
      border-radius: 8px;
      border-left: 4px solid var(--accent-cyan);
      background: rgba(19, 28, 49, 0.85);
      border-top: 1px solid var(--border);
      border-right: 1px solid var(--border);
      border-bottom: 1px solid var(--border);
      margin-bottom: 14px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 10px;
    }
    .schematic-box {
      background: var(--bg-panel);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 12px 16px;
      margin-bottom: 16px;
      overflow-x: auto;
    }
    .main-grid {
      display: grid;
      grid-template-columns: 1.35fr 1fr;
      gap: 16px;
      align-items: start;
    }
    @media (max-width: 1180px) {
      .main-grid {
        grid-template-columns: 1fr;
      }
    }
    .stage-card {
      background: var(--bg-panel);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 14px 16px;
      margin-bottom: 12px;
      position: relative;
    }
    .stage-header {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 10px;
      margin-bottom: 10px;
      padding-bottom: 8px;
      border-bottom: 1px solid rgba(255,255,255,0.07);
    }
    .stage-title {
      font-size: 14px;
      font-weight: 800;
      color: #f0f6fc;
    }
    .stage-sub {
      font-size: 11.5px;
      color: var(--text-muted);
      margin-top: 2px;
    }
    .stage-param-badge {
      background: rgba(56, 189, 248, 0.12);
      border: 1px solid rgba(56, 189, 248, 0.35);
      color: var(--accent-cyan);
      padding: 3px 9px;
      border-radius: 999px;
      font-size: 11px;
      font-weight: 700;
      white-space: nowrap;
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    }
    .subblocks-list {
      display: flex;
      flex-direction: column;
      gap: 8px;
    }
    .subblock-item {
      background: var(--bg-card);
      border: 1px solid #253352;
      border-radius: 8px;
      padding: 10px 12px;
      cursor: pointer;
      transition: all 0.14s ease;
      position: relative;
    }
    .subblock-item:hover {
      background: var(--bg-card-hover);
      border-color: #435d91;
    }
    .subblock-item.selected {
      background: #1a2849;
      border-color: var(--accent-cyan);
      box-shadow: 0 0 0 1px var(--accent-cyan), 0 4px 16px rgba(0, 0, 0, 0.35);
    }
    .subblock-item.dimmed {
      opacity: 0.32;
    }
    .subblock-top {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
    }
    .subblock-title {
      font-weight: 700;
      font-size: 13px;
      color: #f0f6fc;
    }
    .subblock-sub {
      font-size: 11.5px;
      color: #9ca3af;
      margin-top: 2px;
    }
    .shape-ribbon {
      margin-top: 7px;
      padding: 5px 8px;
      background: rgba(7, 11, 20, 0.65);
      border: 1px solid rgba(255,255,255,0.06);
      border-radius: 5px;
      font-family: "JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 11px;
      color: #93c5fd;
      display: flex;
      justify-content: space-between;
      flex-wrap: wrap;
      gap: 6px;
    }
    .wire-connector {
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      margin: -4px 0 8px 0;
      color: #38bdf8;
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 11px;
      font-weight: 600;
    }
    .badge-cat {
      font-size: 10px;
      font-weight: 700;
      text-transform: uppercase;
      padding: 2px 7px;
      border-radius: 4px;
      letter-spacing: 0.4px;
    }
    .cat-input { background: rgba(148,163,184,0.18); color: #cbd5e1; border: 1px solid rgba(148,163,184,0.35); }
    .cat-embedding { background: rgba(16,185,129,0.16); color: #34d399; border: 1px solid rgba(16,185,129,0.38); }
    .cat-unitary { background: rgba(236,72,153,0.16); color: #f472b6; border: 1px solid rgba(236,72,153,0.38); }
    .cat-attention { background: rgba(56,189,248,0.16); color: #38bdf8; border: 1px solid rgba(56,189,248,0.38); }
    .cat-ffn { background: rgba(245,158,11,0.16); color: #fbbf24; border: 1px solid rgba(245,158,11,0.38); }
    .cat-mixer { background: rgba(168,85,247,0.16); color: #c084fc; border: 1px solid rgba(168,85,247,0.38); }
    .cat-readout_prep { background: rgba(59,130,246,0.16); color: #60a5fa; border: 1px solid rgba(59,130,246,0.38); }
    .cat-readout_born { background: rgba(20,184,166,0.18); color: #2dd4bf; border: 1px solid rgba(20,184,166,0.42); }
    .inspector-panel {
      background: var(--bg-panel);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 16px 18px;
      position: sticky;
      top: 14px;
    }
    .sec-heading {
      font-size: 11px;
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: 0.6px;
      color: var(--accent-cyan);
      margin: 14px 0 6px 0;
    }
    .math-box {
      background: #080d18;
      border: 1px solid #22304c;
      border-left: 3px solid var(--accent-purple);
      border-radius: 6px;
      padding: 10px 12px;
      font-family: "JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 11.5px;
      color: #e9d5ff;
      overflow-x: auto;
      white-space: pre-wrap;
    }
    .why-box {
      background: rgba(16, 185, 129, 0.08);
      border: 1px solid rgba(16, 185, 129, 0.28);
      border-radius: 6px;
      padding: 10px 12px;
      color: #d1fae5;
      font-size: 12.5px;
    }
    .param-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 11.5px;
      margin-top: 6px;
    }
    .param-table th, .param-table td {
      border-bottom: 1px solid #1f2b44;
      padding: 6px 7px;
      text-align: left;
      vertical-align: top;
    }
    .param-table th {
      color: var(--text-muted);
      font-weight: 700;
      background: rgba(255,255,255,0.02);
    }
    .mono {
      font-family: "JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, monospace;
    }
    .bottom-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
      margin-top: 18px;
    }
    @media (max-width: 1080px) {
      .bottom-grid {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>
<body>
  <div id="app"></div>
  <script>
    window.__ARCH_GRAPH_PAYLOAD__ = __PAYLOAD_JSON__;
  </script>
  <script>
    (function() {
      const state = {
        payload: window.__ARCH_GRAPH_PAYLOAD__,
        presetKey: (window.__ARCH_GRAPH_PAYLOAD__ && window.__ARCH_GRAPH_PAYLOAD__.default_preset) || "w44_compact_r2_d128c_L2_b32",
        highwayId: "all",
        selectedSubBlockId: "l0_gram_stinespring"
      };

      const urlParams = new URLSearchParams(window.location.search);
      if (urlParams.get("preset") && state.payload.presets[urlParams.get("preset")]) {
        state.presetKey = urlParams.get("preset");
      }
      if (urlParams.get("highway")) {
        state.highwayId = urlParams.get("highway");
      }
      if (urlParams.get("block")) {
        state.selectedSubBlockId = urlParams.get("block");
      }

      function formatInt(n) {
        return Number(n || 0).toLocaleString();
      }

      function getCurrentPreset() {
        return state.payload.presets[state.presetKey] || Object.values(state.payload.presets)[0];
      }

      function findSubBlock(preset, subId) {
        for (const st of preset.stages) {
          for (const sb of st.sub_blocks) {
            if (sb.id === subId) return { stage: st, subBlock: sb };
          }
        }
        const firstStage = preset.stages[0];
        return { stage: firstStage, subBlock: firstStage.sub_blocks[0] };
      }

      function renderSchematicSvg(preset, activeHighway) {
        const stages = preset.stages;
        const boxW = 175;
        const boxH = 62;
        const gap = 36;
        const padX = 24;
        const totalW = Math.max(960, padX * 2 + stages.length * boxW + (stages.length - 1) * gap);
        const svgH = 178;
        const yBox = 64;

        let rects = "";
        let arrows = "";
        const centers = [];

        stages.forEach((st, idx) => {
          const x = padX + idx * (boxW + gap);
          const cx = x + boxW / 2;
          centers.push({ x, cx, st });

          const hasActiveNode = st.sub_blocks.some(sb => activeHighway.node_ids.includes(sb.id));
          const stroke = hasActiveNode ? activeHighway.color : "#2e3e5e";
          const fill = hasActiveNode ? "rgba(25, 37, 66, 0.95)" : "rgba(15, 22, 38, 0.55)";

          rects += `
            <g style="cursor:pointer" onclick="window.__selectStageFirstBlock('${st.sub_blocks[0].id}')">
              <rect x="${x}" y="${yBox}" width="${boxW}" height="${boxH}" rx="8" fill="${fill}" stroke="${stroke}" stroke-width="${hasActiveNode ? 2.2 : 1.2}" />
              <text x="${cx}" y="${yBox + 20}" text-anchor="middle" fill="#f0f6fc" font-size="11.5" font-weight="700">Stage ${st.stage_num}</text>
              <text x="${cx}" y="${yBox + 36}" text-anchor="middle" fill="#38bdf8" font-size="10.5" font-family="monospace">${st.title.split("—")[1] ? st.title.split("—")[1].trim().slice(0, 23) : st.title.slice(0, 23)}</text>
              <text x="${cx}" y="${yBox + 51}" text-anchor="middle" fill="#8b949e" font-size="10" font-family="monospace">${(st.total_params / 1e6).toFixed(2)}M (${st.pct_of_model}%)</text>
            </g>
          `;

          if (idx < stages.length - 1) {
            const x1 = x + boxW;
            const x2 = x + boxW + gap;
            arrows += `
              <line x1="${x1 + 2}" y1="${yBox + boxH / 2}" x2="${x2 - 6}" y2="${yBox + boxH / 2}" stroke="#38bdf8" stroke-width="2" />
              <polygon points="${x2 - 2},${yBox + boxH / 2} ${x2 - 9},${yBox + boxH / 2 - 4} ${x2 - 9},${yBox + boxH / 2 + 4}" fill="#38bdf8" />
            `;
          }
        });

        // Highway overlay arcs:
        // 1. Anchor |psi_0> skip arc from Stage 0 to Stage 2..Readout Prep
        const c0 = centers[0].cx;
        const cPrep = centers[centers.length - 2].cx;
        const cReadout = centers[centers.length - 1].cx;
        const anchorOpacity = (state.highwayId === "all" || state.highwayId === "anchor_psi0") ? 0.95 : 0.18;
        const ngramOpacity = (state.highwayId === "all" || state.highwayId === "ngram_hperp") ? 0.95 : 0.18;
        const tiedOpacity = (state.highwayId === "all" || state.highwayId === "tied_vocab") ? 0.95 : 0.18;

        const arcs = `
          <!-- Anchor |psi_0> & V^(0) Skip Arc (Top) -->
          <path d="M ${c0} ${yBox} Q ${(c0 + cPrep) / 2} 8 ${cPrep} ${yBox}" fill="none" stroke="#f59e0b" stroke-width="2" stroke-dasharray="5,3" opacity="${anchorOpacity}" />
          <text x="${(c0 + cPrep) / 2}" y="22" text-anchor="middle" fill="#f59e0b" font-size="10.5" font-weight="700" opacity="${anchorOpacity}">
            ⚓ Layer-0 Anchor Skip Highway: |ψ_0⟩ &amp; V^(0) ➔ V-Residual, Conjugate Readout Skip, H_⊥ Self-Echo &amp; Phase Router
          </text>

          <!-- 900M N-Gram H_perp Highway (Bottom Inner) -->
          <path d="M ${c0} ${yBox + boxH} Q ${(c0 + cPrep) / 2} ${yBox + boxH + 34} ${cPrep} ${yBox + boxH}" fill="none" stroke="#a855f7" stroke-width="2" stroke-dasharray="4,3" opacity="${ngramOpacity}" />
          <text x="${(c0 + cPrep) / 2}" y="${yBox + boxH + 26}" text-anchor="middle" fill="#c084fc" font-size="10" font-weight="700" opacity="${ngramOpacity}">
            📚 900M-Token Calibrated Bigram (8,192) &amp; Hashed Trigram (16,384) H_⊥ Channels
          </text>

          <!-- Tied 16^4 Wavefunction Loop (Bottom Outer) -->
          <path d="M ${c0} ${yBox + boxH} Q ${(c0 + cReadout) / 2} ${yBox + boxH + 62} ${cReadout} ${yBox + boxH}" fill="none" stroke="#10b981" stroke-width="2.2" opacity="${tiedOpacity}" />
          <text x="${(c0 + cReadout) / 2}" y="${yBox + boxH + 46}" text-anchor="middle" fill="#34d399" font-size="10.5" font-weight="700" opacity="${tiedOpacity}">
            🔁 Tied 16^4 = 65,536 Wavefunction Table: embed_re/im (12.88M params = 76.4%) Shared Between Input Lookup &amp; Chunked Born Readout
          </text>
        `;

        return `
          <svg width="${totalW}" height="${svgH}" viewBox="0 0 ${totalW} ${svgH}" style="display:block; margin:0 auto;">
            ${arcs}
            ${arrows}
            ${rects}
          </svg>
        `;
      }

      function renderLrSweepChart(telemetry) {
        const rows = (telemetry && telemetry.lr_sweep_300steps) || [];
        if (!rows.length) return "<div style='color:#8b949e'>No LR sweep data found.</div>";
        const bestRow = rows.reduce((acc, r) => (r.best_val_ce != null && r.best_val_ce < (acc.best_val_ce || 99)) ? r : acc, rows[0]);

        const trs = rows.map(r => {
          const isBest = r.run_key === bestRow.run_key;
          const barPct = Math.max(8, Math.min(100, ((6.25 - Number(r.best_val_ce || 6.2)) / 0.40) * 100));
          return `
            <tr style="${isBest ? 'background:rgba(16,185,129,0.12); font-weight:700;' : ''}">
              <td class="mono" style="color:${isBest ? '#34d399' : '#e6edf3'}">${r.lr_mult} ${isBest ? '★' : ''}</td>
              <td class="mono" style="color:${isBest ? '#34d399' : '#93c5fd'}">${Number(r.best_val_ce).toFixed(4)}</td>
              <td class="mono">${Number(r.top1_acc || 0).toFixed(2)}%</td>
              <td class="mono">${Math.round(Number(r.tok_per_s || 0)).toLocaleString()}</td>
              <td style="width:38%">
                <div style="background:#090f1d; border-radius:4px; height:10px; overflow:hidden; border:1px solid #1e293b;">
                  <div style="width:${barPct.toFixed(1)}%; height:100%; background:${isBest ? '#10b981' : '#38bdf8'};"></div>
                </div>
              </td>
            </tr>
          `;
        }).join("");

        return `
          <table class="param-table">
            <thead>
              <tr>
                <th>LR Mult</th>
                <th>Best Val CE (300 st)</th>
                <th>Top-1 %</th>
                <th>Tok/s</th>
                <th>Relative Improvement</th>
              </tr>
            </thead>
            <tbody>${trs}</tbody>
          </table>
        `;
      }

      function renderComparisonRunsTable(telemetry) {
        const runs = (telemetry && telemetry.comparison_runs) || [];
        if (!runs.length) return "";
        const trs = runs.map(r => {
          const isTarget = r.run_key.indexOf("w44_compact_r2_d128c_L2_b32") === 0;
          return `
            <tr style="${isTarget ? 'background:rgba(56,189,248,0.10);' : ''}">
              <td class="mono" style="font-weight:700; color:${isTarget ? '#38bdf8' : '#e6edf3'}">${r.run_key}</td>
              <td class="mono">${Number(r.params_M || 0).toFixed(2)}M</td>
              <td class="mono">${r.current_step || 0}/${r.total_steps || 0} (${Number(r.live_tokens_M || 0).toFixed(1)}M)</td>
              <td class="mono" style="font-weight:700; color:#34d399">${r.best_val_ce != null ? Number(r.best_val_ce).toFixed(4) : '—'}</td>
              <td class="mono">${r.top1_acc != null ? Number(r.top1_acc).toFixed(2) + '%' : '—'}</td>
              <td class="mono">${r.avg_tok_per_s ? Math.round(r.avg_tok_per_s).toLocaleString() : '—'}</td>
            </tr>
          `;
        }).join("");
        return `
          <table class="param-table">
            <thead>
              <tr>
                <th>Run Key</th>
                <th>Params</th>
                <th>Steps (Tokens)</th>
                <th>Best Val CE</th>
                <th>Top-1</th>
                <th>Tok/s</th>
              </tr>
            </thead>
            <tbody>${trs}</tbody>
          </table>
        `;
      }

      function render() {
        const preset = getCurrentPreset();
        const activeHighway = preset.highways.find(h => h.id === state.highwayId) || preset.highways[0];
        const { stage: selStage, subBlock: selSub } = findSubBlock(preset, state.selectedSubBlockId);
        state.selectedSubBlockId = selSub.id;

        const presetButtons = Object.values(state.payload.presets).map(p => `
          <button class="btn-pill ${p.preset_key === state.presetKey ? 'active' : ''}" onclick="window.__setPreset('${p.preset_key}')">
            ${p.preset_key} (${(p.total_params / 1e6).toFixed(2)}M)
          </button>
        `).join("");

        const highwayButtons = preset.highways.map(hw => `
          <button class="btn-pill ${hw.id === state.highwayId ? 'active' : ''}"
                  style="${hw.id === state.highwayId ? 'border-color:' + hw.color + '; background:rgba(255,255,255,0.08);' : ''}"
                  onclick="window.__setHighway('${hw.id}')">
            ${hw.label}
          </button>
        `).join("");

        const stagesHtml = preset.stages.map((st, sIdx) => {
          const subBlocksHtml = st.sub_blocks.map(sb => {
            const isSelected = sb.id === selSub.id;
            const onHighway = activeHighway.node_ids.includes(sb.id);
            const catClass = "cat-" + (sb.category || "input");
            const pCount = sb.meta && sb.meta.params ? formatInt(sb.meta.params) + " params" : "0 params (op)";
            return `
              <div class="subblock-item ${isSelected ? 'selected' : ''} ${!onHighway ? 'dimmed' : ''}"
                   onclick="window.__selectSubBlock('${sb.id}')">
                <div class="subblock-top">
                  <div class="subblock-title">${sb.title}</div>
                  <div style="display:flex; gap:6px; align-items:center;">
                    <span class="badge-cat ${catClass}">${sb.category.replace('_', ' ')}</span>
                    <span class="mono" style="font-size:11px; color:#cbd5e1; background:rgba(255,255,255,0.06); padding:2px 6px; border-radius:4px;">${pCount}</span>
                  </div>
                </div>
                <div class="subblock-sub">${sb.subtitle}</div>
                <div class="shape-ribbon">
                  <span><strong>In:</strong> ${sb.in_shape}</span>
                  <span>➔</span>
                  <span><strong>Out:</strong> ${sb.out_shape}</span>
                </div>
              </div>
            `;
          }).join("");

          const connectorHtml = sIdx < preset.stages.length - 1 ? `
            <div class="wire-connector">
              <span>⬇ Tensor Wire: ${st.tensor_ribbon} ⬇</span>
            </div>
          ` : "";

          return `
            <div class="stage-card">
              <div class="stage-header">
                <div>
                  <div class="stage-title">${st.title}</div>
                  <div class="stage-sub">${st.subtitle}</div>
                </div>
                <div class="stage-param-badge">${formatInt(st.total_params)} params (${st.pct_of_model}%)</div>
              </div>
              <div class="subblocks-list">
                ${subBlocksHtml}
              </div>
            </div>
            ${connectorHtml}
          `;
        }).join("");

        const paramTensors = (selSub.meta && selSub.meta.param_tensors) || [];
        const paramTableHtml = paramTensors.length ? `
          <table class="param-table">
            <thead>
              <tr>
                <th>Parameter Tensor</th>
                <th>Shape</th>
                <th>Count</th>
                <th>Optimizer Routing &amp; Init</th>
              </tr>
            </thead>
            <tbody>
              ${paramTensors.map(pt => `
                <tr>
                  <td class="mono" style="color:#38bdf8; font-weight:600;">${pt.name}<div style="color:#8b949e; font-family:sans-serif; font-weight:400; margin-top:2px;">${pt.role}</div></td>
                  <td class="mono">${pt.shape}</td>
                  <td class="mono">${formatInt(pt.params)}</td>
                  <td>
                    <div style="color:#f472b6; font-weight:600;">${pt.optimizer_group}</div>
                    <div style="color:#9ca3af; font-size:11px; margin-top:2px;">Init: ${pt.init_rule}</div>
                  </td>
                </tr>
              `).join("")}
            </tbody>
          </table>
        ` : `<div style="color:#8b949e; font-size:12px; padding:6px 0;">Parameter-free quantum operation (or reuses tied <code class="mono">embed_re / embed_im</code> wavefunctions).</div>`;

        const edgeNotesHtml = (activeHighway.edge_notes || []).map(n => `<li style="margin-bottom:3px;">${n}</li>`).join("");

        document.getElementById("app").innerHTML = `
          <div class="top-header">
            <div>
              <div class="title-main">🧬 QuantumDLBlocks — Interactive Block Architecture Visualizer</div>
              <div class="subtitle-main">${preset.title} • ${preset.summary_badge}</div>
            </div>
            <div style="display:flex; gap:8px; align-items:center;">
              <a href="/" class="btn-pill" style="text-decoration:none;">⬅ Back to Main Telemetry Dashboard</a>
            </div>
          </div>

          <div class="controls-bar">
            <span class="ctrl-label">Model Preset:</span>
            ${presetButtons}
            <span class="ctrl-label" style="margin-left:10px;">Highlight Signal Highway:</span>
            ${highwayButtons}
          </div>

          <div class="highway-banner" style="border-left-color:${activeHighway.color}">
            <div>
              <div style="font-weight:800; color:${activeHighway.color}; font-size:13.5px;">${activeHighway.label}</div>
              <div style="color:#d1d5db; margin-top:2px;">${activeHighway.description}</div>
              <ul style="margin:6px 0 0 18px; padding:0; color:#9ca3af; font-size:12px;">${edgeNotesHtml}</ul>
            </div>
            <div class="mono" style="background:#080d18; padding:8px 12px; border-radius:6px; border:1px solid #22304c; font-size:11.5px; color:#38bdf8;">
              Total Params: <strong>${formatInt(preset.total_params)}</strong><br/>
              Hilbert Space: <strong>(C^16)^(⊗4) = C^65,536 × R=${preset.hyperparams.num_meanings_R}</strong>
            </div>
          </div>

          <div class="schematic-box">
            <div style="font-size:11.5px; font-weight:700; color:#8b949e; margin-bottom:6px; text-transform:uppercase; letter-spacing:0.5px;">
              End-to-End Macro Stage Pipeline &amp; Skip-Highway Topology (Click Any Stage or Sub-Block Below to Inspect)
            </div>
            ${renderSchematicSvg(preset, activeHighway)}
          </div>

          <div class="main-grid">
            <div>
              ${stagesHtml}
            </div>

            <div class="inspector-panel">
              <div style="display:flex; justify-content:space-between; align-items:flex-start; gap:8px;">
                <div>
                  <div style="font-size:11px; color:#38bdf8; font-weight:700; text-transform:uppercase;">Selected Sub-Block Inspector (${selStage.title.split("—")[0].trim()})</div>
                  <div style="font-size:16px; font-weight:800; color:#fff; margin-top:2px;">${selSub.title}</div>
                  <div style="font-size:12px; color:#9ca3af; margin-top:2px;">${selSub.subtitle}</div>
                </div>
                <span class="badge-cat cat-${selSub.category}">${selSub.category.replace('_', ' ')}</span>
              </div>

              <div class="sec-heading">1. Tensor Wire Shapes &amp; Quantum Manifold</div>
              <div style="background:#080d18; border:1px solid #22304c; border-radius:6px; padding:10px 12px; font-size:12px;">
                <div class="mono" style="color:#93c5fd;"><strong>Input Wire:</strong>  ${selSub.in_shape}</div>
                <div class="mono" style="color:#34d399; margin-top:4px;"><strong>Output Wire:</strong> ${selSub.out_shape}</div>
                <div style="color:#d8b4fe; margin-top:6px;"><strong>Hilbert Manifold:</strong> ${selSub.hilbert_space}</div>
              </div>

              <div class="sec-heading">2. Mathematical Equation / Operation</div>
              <div class="math-box">${selSub.meta.equation || ""}</div>

              <div class="sec-heading">3. Why This Block Was Strung Here (Wave 41–44 Provenance)</div>
              <div class="why-box">${selSub.meta.why_built || ""}</div>

              <div class="sec-heading">4. Fused GPU Kernel &amp; Optimizer Routing</div>
              <div style="background:#080d18; border:1px solid #22304c; border-radius:6px; padding:9px 12px; font-size:12px;">
                <div><strong style="color:#38bdf8;">Fused Kernel:</strong> <span class="mono">${selSub.meta.kernel || "Standard PyTorch"}</span></div>
                <div style="margin-top:4px;"><strong style="color:#f472b6;">Optimizer Group:</strong> <span class="mono">${selSub.meta.optimizer || "—"}</span></div>
                <div style="margin-top:4px;"><strong style="color:#9ca3af;">Source Location:</strong> <code class="mono" style="color:#a7f3d0;">${selSub.meta.source_ref || ""}</code></div>
              </div>

              <div class="sec-heading">5. Learnable Parameter Tensors in This Sub-Block (${formatInt(selSub.meta.params || 0)} params)</div>
              ${paramTableHtml}
            </div>
          </div>

          <div class="bottom-grid">
            <div class="stage-card">
              <div class="stage-header">
                <div>
                  <div class="stage-title">📈 Empirical Constant-LR Sweep on w44_compact_r2_d128c_L2_b32 (300 Steps, B=32, T=512)</div>
                  <div class="stage-sub">Identifies the 1.50x constant learning rate sweet spot (5.9102 Val CE in 300 steps at 20,975 tok/s)</div>
                </div>
              </div>
              ${renderLrSweepChart(state.payload.telemetry)}
            </div>

            <div class="stage-card">
              <div class="stage-header">
                <div>
                  <div class="stage-title">🏁 Convergence &amp; Architecture Comparison Runs</div>
                  <div class="stage-sub">Live / Completed runs from .cache/live_logs/ comparing compact R=2 L=2 against R=1, L=6, and Classical L=12</div>
                </div>
              </div>
              ${renderComparisonRunsTable(state.payload.telemetry)}
            </div>
          </div>
        `;
      }

      window.__setPreset = function(k) {
        state.presetKey = k;
        render();
      };
      window.__setHighway = function(hid) {
        state.highwayId = hid;
        render();
      };
      window.__selectSubBlock = function(sid) {
        state.selectedSubBlockId = sid;
        render();
      };
      window.__selectStageFirstBlock = function(sid) {
        state.selectedSubBlockId = sid;
        render();
      };

      render();

      // Fetch live updates from /api/architecture-graph if served via monitor_colab_runs.py
      fetch("/api/architecture-graph")
        .then(r => r.ok ? r.json() : null)
        .then(data => {
          if (data && data.presets) {
            state.payload = data;
            render();
          }
        })
        .catch(() => {});
    })();
  </script>
</body>
</html>
"""


def render_architecture_visualizer_html(payload: dict[str, Any] | None = None) -> str:
    """Render standalone HTML for the Block Architecture Visualizer (`<= 8` args)."""
    resolved = payload if payload is not None else build_architecture_visualizer_payload()
    payload_json = json.dumps(resolved)
    return HTML_TEMPLATE.replace("__PAYLOAD_JSON__", payload_json)


def main() -> None:
    """Parse CLI arguments and export the Block Architecture Visualizer HTML & JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preset",
        type=str,
        default=DEFAULT_PRESET_KEY,
        help="Default model preset key to highlight on load.",
    )
    parser.add_argument(
        "--html-out",
        type=str,
        default=DEFAULT_HTML_OUT,
        help="Output path for standalone interactive HTML visualizer.",
    )
    parser.add_argument(
        "--json-out",
        type=str,
        default=DEFAULT_JSON_OUT,
        help="Output path for block-graph JSON artifact.",
    )
    args = parser.parse_args()

    payload = build_architecture_visualizer_payload(live_logs_dir=ROOT / ".cache" / "live_logs")
    if args.preset in payload["presets"]:
        payload["default_preset"] = args.preset

    html_path = Path(args.html_out)
    if not html_path.is_absolute():
        html_path = ROOT / html_path
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(render_architecture_visualizer_html(payload), encoding="utf-8")

    json_path = Path(args.json_out)
    if not json_path.is_absolute():
        json_path = ROOT / json_path
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    preset_obj = payload["presets"][payload["default_preset"]]
    print(
        f"[export_model_block_visualizer] Exported preset={payload['default_preset']} "
        f"(total_params={preset_obj['total_params']:,}, stages={len(preset_obj['stages'])}) "
        f"-> html={html_path} json={json_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
