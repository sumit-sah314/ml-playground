import { useState } from "react";
import { predictNetwork } from "@/lib/neural-network.js";
import { C, useHiDpiCanvas, drawMarker } from "../logistic/LogisticCanvases.jsx";

const MONO = "'JetBrains Mono', monospace";
const PAD = { l: 30, r: 10, t: 10, b: 28 };
const N = 64;

export function mapView(cw, ch) {
  const iw = cw - PAD.l - PAD.r, ih = ch - PAD.t - PAD.b;
  return {
    iw, ih,
    sx: (x) => PAD.l + ((x + 1) / 2) * iw,
    sy: (y) => PAD.t + ((1 - y) / 2) * ih,
    ix: (px) => -1 + ((px - PAD.l) / iw) * 2,
    iy: (py) => 1 - ((py - PAD.t) / ih) * 2,
  };
}

function mix(t) {
  const a = [28, 122, 114], z = [236, 106, 76], p = [251, 248, 240];
  const k = Math.abs(t - 0.5) * 2;
  const end = t >= 0.5 ? z : a;
  return p.map((v, i) => Math.round(v + (end[i] - v) * k * 0.5));
}

export function DecisionMap({ network, points, version, probe, onPointerMove, onPointerLeave }) {
  const ref = useHiDpiCanvas((ctx, cw, ch) => {
    const v = mapView(cw, ch);
    ctx.fillStyle = C.paper; ctx.fillRect(0, 0, cw, ch);
    // sample grid at cell corners (N+1)^2
    const g = new Float64Array((N + 1) * (N + 1));
    for (let j = 0; j <= N; j++) for (let i = 0; i <= N; i++) {
      const p = predictNetwork(network, -1 + (2 * i) / N, 1 - (2 * j) / N);
      g[j * (N + 1) + i] = Number.isFinite(p) ? p : 0.5;
    }
    const cw2 = v.iw / N, chh = v.ih / N;
    for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) {
      const k = j * (N + 1) + i;
      const c = mix((g[k] + g[k + 1] + g[k + N + 1] + g[k + N + 2]) / 4);
      ctx.fillStyle = `rgb(${c[0]},${c[1]},${c[2]})`;
      ctx.fillRect(PAD.l + i * cw2, PAD.t + j * chh, cw2 + 0.6, chh + 0.6);
    }
    // grid + axes
    ctx.font = `10px ${MONO}`; ctx.fillStyle = C.axis; ctx.lineWidth = 1;
    for (let t = -1; t <= 1.001; t += 0.5) {
      ctx.strokeStyle = Math.abs(t) < 1e-9 ? "rgba(31,42,68,.35)" : "rgba(138,129,112,.2)";
      ctx.beginPath(); ctx.moveTo(v.sx(t), PAD.t); ctx.lineTo(v.sx(t), PAD.t + v.ih); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(PAD.l, v.sy(t)); ctx.lineTo(PAD.l + v.iw, v.sy(t)); ctx.stroke();
      ctx.textAlign = "center"; ctx.fillText(String(t), v.sx(t), ch - PAD.b + 13);
      ctx.textAlign = "right"; ctx.fillText(String(t), PAD.l - 4, v.sy(t) + 3);
    }
    ctx.fillStyle = C.ink; ctx.font = `600 11px ${MONO}`; ctx.textAlign = "center";
    ctx.fillText("x1", PAD.l + v.iw / 2, ch - 4);
    ctx.save(); ctx.translate(9, PAD.t + v.ih / 2); ctx.rotate(-Math.PI / 2); ctx.fillText("x2", 0, 0); ctx.restore();
    // marching squares p = 0.5 contour
    ctx.save(); ctx.beginPath(); ctx.rect(PAD.l, PAD.t, v.iw, v.ih); ctx.clip();
    ctx.strokeStyle = C.ink; ctx.lineWidth = 2.2; ctx.lineCap = "round"; ctx.beginPath();
    const px = (i) => PAD.l + i * cw2, py = (j) => PAD.t + j * chh;
    const lerp = (a, b) => (0.5 - a) / (b - a);
    for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) {
      const k = j * (N + 1) + i;
      const tl = g[k], tr = g[k + 1], bl = g[k + N + 1], br = g[k + N + 2];
      const e = [];
      if ((tl >= 0.5) !== (tr >= 0.5)) e.push([px(i + lerp(tl, tr)), py(j)]);
      if ((tr >= 0.5) !== (br >= 0.5)) e.push([px(i + 1), py(j + lerp(tr, br))]);
      if ((bl >= 0.5) !== (br >= 0.5)) e.push([px(i + lerp(bl, br)), py(j + 1)]);
      if ((tl >= 0.5) !== (bl >= 0.5)) e.push([px(i), py(j + lerp(tl, bl))]);
      for (let s = 0; s + 1 < e.length; s += 2) { ctx.moveTo(e[s][0], e[s][1]); ctx.lineTo(e[s + 1][0], e[s + 1][1]); }
    }
    ctx.stroke();
    points.forEach((p) => drawMarker(ctx, v.sx(p.x), v.sy(p.y), p.label, 4));
    if (probe) {
      const qx = v.sx(probe.x), qy = v.sy(probe.y);
      ctx.strokeStyle = C.gold; ctx.lineWidth = 2.5; ctx.beginPath(); ctx.arc(qx, qy, 9, 0, Math.PI * 2); ctx.stroke();
      ctx.strokeStyle = C.ink; ctx.lineWidth = 1.5; ctx.beginPath();
      ctx.moveTo(qx - 5, qy); ctx.lineTo(qx + 5, qy); ctx.moveTo(qx, qy - 5); ctx.lineTo(qx, qy + 5); ctx.stroke();
    }
    ctx.restore();
  }, [network, points, version, probe]);
  return (
    <canvas ref={ref} data-testid="canvas-decision-map" className="w-full h-full rounded-xl touch-none cursor-crosshair"
      aria-label="Decision probability map over x1, x2 in [-1, 1] with the p = 0.5 boundary drawn as a dark line"
      onPointerMove={onPointerMove} onPointerLeave={onPointerLeave} onPointerDown={onPointerMove} />
  );
}

/** Fixed edge scale: stroke px = 0.5 + 2.2·|w|, capped at 9 px (|w| ≥ ~3.9). */
export const edgeWidth = (w) => Math.min(9, 0.5 + 2.2 * Math.abs(w));

export function NetworkDiagram({ network, activations, version }) {
  const [sel, setSel] = useState(null);
  void version;
  const sizes = network.sizes;
  const W = 640, H = 340, top = 28, bot = 28;
  const col = (l) => 50 + (l * (W - 100)) / (sizes.length - 1);
  const maxN = Math.max(...sizes);
  const gap = Math.min(40, (H - top - bot) / Math.max(1, maxN - 1));
  const row = (l, i) => H / 2 + (i - (sizes[l] - 1) / 2) * gap;
  const r = Math.max(6, Math.min(13, gap * 0.38));
  const labels = sizes.map((_, l) => (l === 0 ? "input" : l === sizes.length - 1 ? "σ output" : `hidden ${l}`));
  const fill = (l, i) => {
    if (!activations) return C.paper;
    const a = activations[l]?.[i];
    if (!Number.isFinite(a)) return C.paper;
    let t = l === sizes.length - 1 ? a * 2 - 1 : Math.max(-1, Math.min(1, a));
    const end = t >= 0 ? [236, 106, 76] : [28, 122, 114];
    t = Math.abs(t);
    return `rgb(${[251, 248, 240].map((v, k) => Math.round(v + (end[k] - v) * t * 0.85)).join(",")})`;
  };
  const selW = sel ? network.layers[sel.l]?.weights[sel.j]?.[sel.i] : null;
  return (
    <div>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto" role="img" aria-label="Network diagram; edge thickness proportional to absolute weight" data-testid="svg-network">
        {network.layers.map((layer, l) => layer.weights.map((rowW, j) => rowW.map((w, i) => {
          const on = sel && sel.l === l && sel.i === i && sel.j === j;
          return (
            <line key={`${l}-${j}-${i}`} x1={col(l)} y1={row(l, i)} x2={col(l + 1)} y2={row(l + 1, j)}
              stroke={on ? C.gold : w >= 0 ? C.coral : C.teal} strokeOpacity={on ? 1 : 0.7} strokeWidth={Number.isFinite(w) ? edgeWidth(w) : 0.5}
              strokeLinecap="round" style={{ cursor: "pointer" }}
              onPointerEnter={() => setSel({ l, i, j })} onClick={() => setSel({ l, i, j })} data-testid={`edge-${l}-${j}-${i}`}>
              <title>{`w = ${w.toFixed(3)}`}</title>
            </line>
          );
        })))}
        {sizes.map((n, l) => Array.from({ length: n }, (_, i) => {
          const b = l > 0 ? network.layers[l - 1].biases[i] : null;
          return (
            <g key={`n${l}-${i}`}>
              <circle cx={col(l)} cy={row(l, i)} r={r} fill={fill(l, i)} stroke={C.ink} strokeWidth={b == null ? 1.5 : Math.min(4, 1 + Math.abs(b))} strokeDasharray={b != null && b < 0 ? "3 2" : undefined}>
                <title>{b == null ? `input x${i + 1}` : `bias b = ${b.toFixed(3)}`}</title>
              </circle>
              {l === 0 && <text x={col(l) - r - 6} y={row(l, i) + 4} textAnchor="end" fontSize="12" fontFamily={MONO} fill={C.ink}>x{i + 1}</text>}
            </g>
          );
        }))}
        {labels.map((t, l) => <text key={t} x={col(l)} y={H - 8} textAnchor="middle" fontSize="11" fontFamily={MONO} fill={C.axis}>{t}</text>)}
      </svg>
      <p className="num text-xs mt-2 min-h-[1.25rem]" data-testid="text-selected-weight">
        {sel && Number.isFinite(selW)
          ? <>layer {sel.l + 1}: {sel.l === 0 ? `x${sel.i + 1}` : `h${sel.l}.${sel.i + 1}`} → {sel.l + 1 === sizes.length - 1 ? "output" : `h${sel.l + 1}.${sel.j + 1}`} &nbsp; w = <b>{selW.toFixed(4)}</b></>
          : <span className="text-muted-foreground">hover or tap an edge to read its weight</span>}
      </p>
    </div>
  );
}

export function EpochLossChart({ history, version }) {
  const ref = useHiDpiCanvas((ctx, cw, ch) => {
    ctx.fillStyle = C.paper; ctx.fillRect(0, 0, cw, ch);
    const pl = 46, pr = 36, pt = 12, pb = 26;
    ctx.font = `10px ${MONO}`; ctx.fillStyle = C.axis;
    const pts = history.filter((h) => Number.isFinite(h.loss));
    if (pts.length < 2) { ctx.textAlign = "center"; ctx.fillText("Log-loss per epoch appears once training starts", cw / 2, ch / 2); return; }
    const hi = Math.max(...pts.map((h) => h.loss), 0.75) * 1.05;
    const e0 = pts[0].epoch, span = Math.max(1, pts[pts.length - 1].epoch - e0);
    const sx = (e) => pl + ((e - e0) / span) * (cw - pl - pr);
    const sy = (x) => ch - pb - (x / hi) * (ch - pt - pb);
    ctx.strokeStyle = C.grid; ctx.lineWidth = 1;
    [0, hi / 2, hi].forEach((t) => { ctx.beginPath(); ctx.moveTo(pl, sy(t)); ctx.lineTo(cw - pr, sy(t)); ctx.stroke(); ctx.textAlign = "right"; ctx.fillText(t.toFixed(2), pl - 6, sy(t) + 3); });
    ctx.fillStyle = C.coral; ctx.textAlign = "left";
    [0, 0.5, 1].forEach((t) => ctx.fillText(`${t * 100}%`, cw - pr + 4, ch - pb - t * (ch - pt - pb) + 3));
    ctx.strokeStyle = C.gold; ctx.setLineDash([5, 4]); ctx.beginPath(); ctx.moveTo(pl, sy(Math.LN2)); ctx.lineTo(cw - pr, sy(Math.LN2)); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = "#b07f0a"; ctx.textAlign = "left"; ctx.fillText("ln 2 (coin flip)", pl + 4, sy(Math.LN2) - 4);
    ctx.fillStyle = C.axis; ctx.fillText(`epoch ${e0}`, pl, ch - 6);
    ctx.textAlign = "right"; ctx.fillText(`epoch ${pts[pts.length - 1].epoch}`, cw - pr, ch - 6);
    ctx.strokeStyle = C.teal; ctx.lineWidth = 2; ctx.lineJoin = "round"; ctx.beginPath();
    pts.forEach((h, i) => (i ? ctx.lineTo(sx(h.epoch), sy(h.loss)) : ctx.moveTo(sx(h.epoch), sy(h.loss))));
    ctx.stroke();
    ctx.strokeStyle = C.coral; ctx.lineWidth = 1.5; ctx.globalAlpha = 0.8; ctx.beginPath();
    pts.forEach((h, i) => { const y = ch - pb - h.acc * (ch - pt - pb); i ? ctx.lineTo(sx(h.epoch), y) : ctx.moveTo(sx(h.epoch), y); });
    ctx.stroke(); ctx.globalAlpha = 1;
  }, [version, history.length]);
  return <canvas ref={ref} data-testid="canvas-loss" className="w-full h-full rounded-xl" aria-label="Training log-loss and accuracy versus epoch" />;
}
