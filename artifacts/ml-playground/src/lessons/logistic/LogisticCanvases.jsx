import { useEffect, useRef, useState } from "react";
import { sigmoid, logisticScore } from "@/lib/logistic.js";

export const C = {
  ink: "#1f2a44", paper: "#fbf8f0", grid: "#e7dfcc", axis: "#8a8170",
  teal: "#1c7a72", coral: "#ec6a4c", gold: "#e0a419", plum: "#7a5a9e",
};
export const LIM = 4;
const PAD = { l: 34, r: 12, t: 12, b: 32 };
const MONO = "'JetBrains Mono', monospace";

export function useHiDpiCanvas(draw, deps) {
  const ref = useRef(null);
  const drawRef = useRef(draw);
  drawRef.current = draw;
  const [size, setSize] = useState({ w: 0, h: 0 });
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver(() => { const r = el.getBoundingClientRect(); setSize({ w: r.width, h: r.height }); });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  useEffect(() => {
    const el = ref.current;
    if (!el || !size.w) return;
    const dpr = window.devicePixelRatio || 1;
    el.width = Math.round(size.w * dpr); el.height = Math.round(size.h * dpr);
    const ctx = el.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, size.w, size.h);
    drawRef.current(ctx, size.w, size.h);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [size, ...deps]);
  return ref;
}

export function planeView(cw, ch) {
  const iw = cw - PAD.l - PAD.r, ih = ch - PAD.t - PAD.b;
  const sx = (x) => PAD.l + ((x + LIM) / (2 * LIM)) * iw;
  const sy = (y) => PAD.t + ((LIM - y) / (2 * LIM)) * ih;
  const ix = (px) => -LIM + ((px - PAD.l) / iw) * 2 * LIM;
  const iy = (py) => LIM - ((py - PAD.t) / ih) * 2 * LIM;
  return { sx, sy, ix, iy, iw, ih };
}

function mix(t) {
  const a = [28, 122, 114], z = [236, 106, 76], p = [251, 248, 240];
  // t=P(B); near .5 fades to paper so certainty reads as saturation
  const k = Math.abs(t - 0.5) * 2;
  const end = t >= 0.5 ? z : a;
  return p.map((v, i) => Math.round(v + (end[i] - v) * k * 0.42));
}

export function drawMarker(ctx, x, y, label, r = 6, ring) {
  ctx.lineWidth = 2;
  if (label === 1) {
    ctx.beginPath(); ctx.rect(x - r, y - r, r * 2, r * 2);
    ctx.fillStyle = C.coral; ctx.fill(); ctx.strokeStyle = ring || C.paper; ctx.stroke();
  } else {
    ctx.beginPath(); ctx.arc(x, y, r + 0.5, 0, Math.PI * 2);
    ctx.fillStyle = C.teal; ctx.fill(); ctx.strokeStyle = ring || C.paper; ctx.stroke();
  }
}

function boundarySegment(p) {
  const { wx, wy, b } = p;
  const eps = 1e-9;
  if (!(Number.isFinite(wx) && Number.isFinite(wy) && Number.isFinite(b))) return null;
  if (Math.abs(wx) < eps && Math.abs(wy) < eps) return null;
  const big = LIM * 4;
  if (Math.abs(wy) >= Math.abs(wx)) return [[-big, -(wx * -big + b) / wy], [big, -(wx * big + b) / wy]];
  return [[-(wy * -big + b) / wx, -big], [-(wy * big + b) / wx, big]];
}

export function PlaneCanvas({ points, params, version, selected, probe, onPointerDown, onPointerMove, onPointerUp, onPointerLeave }) {
  const ref = useHiDpiCanvas((ctx, cw, ch) => {
    const v = planeView(cw, ch);
    ctx.fillStyle = C.paper; ctx.fillRect(0, 0, cw, ch);
    // probability shading P(B)
    const N = 56, cell = v.iw / N, cellH = v.ih / N;
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
      const x = -LIM + ((i + 0.5) / N) * 2 * LIM, y = LIM - ((j + 0.5) / N) * 2 * LIM;
      const pr = sigmoid(logisticScore({ x, y }, params));
      const c = mix(Number.isFinite(pr) ? pr : 0.5);
      ctx.fillStyle = `rgb(${c[0]},${c[1]},${c[2]})`;
      ctx.fillRect(PAD.l + i * cell, PAD.t + j * cellH, cell + 0.6, cellH + 0.6);
    }
    // grid
    ctx.font = `10px ${MONO}`; ctx.fillStyle = C.axis; ctx.lineWidth = 1;
    for (let t = -LIM; t <= LIM; t++) {
      ctx.strokeStyle = t === 0 ? "rgba(31,42,68,.45)" : "rgba(138,129,112,.22)";
      ctx.beginPath(); ctx.moveTo(v.sx(t), PAD.t); ctx.lineTo(v.sx(t), ch - PAD.b); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(PAD.l, v.sy(t)); ctx.lineTo(cw - PAD.r, v.sy(t)); ctx.stroke();
      if (t % 2 === 0) {
        ctx.textAlign = "center"; ctx.fillText(String(t), v.sx(t), ch - PAD.b + 14);
        ctx.textAlign = "right"; ctx.fillText(String(t), PAD.l - 6, v.sy(t) + 3);
      }
    }
    ctx.fillStyle = C.ink; ctx.font = `600 11px ${MONO}`; ctx.textAlign = "center";
    ctx.fillText("x1", PAD.l + v.iw / 2, ch - 6);
    ctx.save(); ctx.translate(10, PAD.t + v.ih / 2); ctx.rotate(-Math.PI / 2); ctx.fillText("x2", 0, 0); ctx.restore();

    ctx.save(); ctx.beginPath(); ctx.rect(PAD.l, PAD.t, v.iw, v.ih); ctx.clip();
    const seg = boundarySegment(params);
    if (seg) {
      ctx.strokeStyle = C.ink; ctx.lineWidth = 2.5; ctx.setLineDash([9, 6]);
      ctx.beginPath(); ctx.moveTo(v.sx(seg[0][0]), v.sy(seg[0][1])); ctx.lineTo(v.sx(seg[1][0]), v.sy(seg[1][1])); ctx.stroke();
      ctx.setLineDash([]);
      // normal arrow toward B side
      const n = Math.hypot(params.wx, params.wy);
      const fx = (-params.b * params.wx) / (n * n), fy = (-params.b * params.wy) / (n * n);
      if (Math.abs(fx) < LIM && Math.abs(fy) < LIM) {
        const ux = params.wx / n, uy = params.wy / n;
        const ax = v.sx(fx), ay = v.sy(fy), bx = v.sx(fx + ux * 0.8), by = v.sy(fy + uy * 0.8);
        ctx.strokeStyle = C.coral; ctx.lineWidth = 2.5;
        ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(bx, by); ctx.stroke();
        const ang = Math.atan2(by - ay, bx - ax);
        ctx.fillStyle = C.coral; ctx.beginPath(); ctx.moveTo(bx + Math.cos(ang) * 6, by + Math.sin(ang) * 6);
        ctx.lineTo(bx - Math.sin(ang) * 5, by + Math.cos(ang) * 5); ctx.lineTo(bx + Math.sin(ang) * 5, by - Math.cos(ang) * 5); ctx.closePath(); ctx.fill();
      }
    }
    points.forEach((p, i) => drawMarker(ctx, v.sx(p.x), v.sy(p.y), p.label, 6, i === selected ? C.gold : undefined));
    if (selected != null && points[selected]) {
      const p = points[selected];
      ctx.strokeStyle = C.gold; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(v.sx(p.x), v.sy(p.y), 11, 0, Math.PI * 2); ctx.stroke();
    }
    if (probe && selected == null) {
      const px = v.sx(probe.x), py = v.sy(probe.y);
      ctx.strokeStyle = C.ink; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(px - 8, py); ctx.lineTo(px + 8, py); ctx.moveTo(px, py - 8); ctx.lineTo(px, py + 8); ctx.stroke();
    }
    ctx.restore();
    if (!seg) {
      ctx.font = `600 11px ${MONO}`;
      const label = "no boundary: wx = wy = 0, every point gets the same P(B)";
      const tw = Math.min(ctx.measureText(label).width + 14, v.iw - 8);
      ctx.fillStyle = "rgba(31,42,68,.85)"; ctx.beginPath(); ctx.roundRect(PAD.l + 6, PAD.t + 6, tw, 22, 11); ctx.fill();
      ctx.fillStyle = C.paper; ctx.textAlign = "left"; ctx.fillText(label, PAD.l + 13, PAD.t + 21, tw - 14);
    }
  }, [points, params, version, selected, probe]);

  return (
    <canvas
      ref={ref} data-testid="canvas-plane"
      className="w-full h-full touch-none rounded-xl cursor-crosshair select-none"
      aria-label="Feature plane x1 by x2 from -4 to 4. Left click adds class A, right click adds class B."
      onContextMenu={(e) => e.preventDefault()}
      onPointerDown={onPointerDown} onPointerMove={onPointerMove} onPointerUp={onPointerUp}
      onPointerCancel={onPointerUp} onPointerLeave={onPointerLeave}
    />
  );
}

const ZMAX = 8;
export function SigmoidCanvas({ points, params, version, focus }) {
  const ref = useHiDpiCanvas((ctx, cw, ch) => {
    const pl = 38, pr = 12, pt = 14, pb = 30;
    const iw = cw - pl - pr, ih = ch - pt - pb;
    const sx = (z) => pl + ((z + ZMAX) / (2 * ZMAX)) * iw;
    const sy = (p) => pt + (1 - p) * ih;
    ctx.fillStyle = C.paper; ctx.fillRect(0, 0, cw, ch);
    ctx.fillStyle = "rgba(28,122,114,.07)"; ctx.fillRect(pl, pt, iw / 2, ih);
    ctx.fillStyle = "rgba(236,106,76,.08)"; ctx.fillRect(pl + iw / 2, pt, iw / 2, ih);
    ctx.font = `10px ${MONO}`; ctx.fillStyle = C.axis; ctx.strokeStyle = C.grid; ctx.lineWidth = 1;
    for (let z = -ZMAX; z <= ZMAX; z += 2) {
      ctx.beginPath(); ctx.moveTo(sx(z), pt); ctx.lineTo(sx(z), pt + ih); ctx.stroke();
      ctx.textAlign = "center"; ctx.fillText(String(z), sx(z), ch - pb + 14);
    }
    [0, 0.5, 1].forEach((p) => { ctx.beginPath(); ctx.moveTo(pl, sy(p)); ctx.lineTo(pl + iw, sy(p)); ctx.stroke(); ctx.textAlign = "right"; ctx.fillText(String(p), pl - 6, sy(p) + 3); });
    ctx.strokeStyle = C.gold; ctx.lineWidth = 1.5; ctx.setLineDash([5, 4]);
    ctx.beginPath(); ctx.moveTo(pl, sy(0.5)); ctx.lineTo(pl + iw, sy(0.5)); ctx.moveTo(sx(0), pt); ctx.lineTo(sx(0), pt + ih); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = C.ink; ctx.font = `600 11px ${MONO}`; ctx.textAlign = "center";
    ctx.fillText("raw score z", pl + iw / 2, ch - 4);
    ctx.save(); ctx.translate(10, pt + ih / 2); ctx.rotate(-Math.PI / 2); ctx.fillText("P(B)", 0, 0); ctx.restore();
    ctx.strokeStyle = C.ink; ctx.lineWidth = 2.5; ctx.beginPath();
    for (let i = 0; i <= 160; i++) { const z = -ZMAX + (i / 160) * 2 * ZMAX; const x = sx(z), y = sy(sigmoid(z)); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }
    ctx.stroke();
    const place = (z) => { const zc = Math.max(-ZMAX, Math.min(ZMAX, z)); return { x: sx(zc), y: sy(sigmoid(z)), off: Math.abs(z) > ZMAX }; };
    points.forEach((p) => {
      const z = logisticScore(p, params); if (!Number.isFinite(z)) return;
      const m = place(z);
      ctx.globalAlpha = 0.85; drawMarker(ctx, m.x, m.y + (p.label ? -0 : 0), p.label, 4.5); ctx.globalAlpha = 1;
      if (m.off) { ctx.fillStyle = C.axis; ctx.fillRect(m.x + (z > 0 ? -1 : -1), pt + ih + 2, 2, 5); }
    });
    if (focus) {
      const z = logisticScore(focus, params);
      if (Number.isFinite(z)) {
        const m = place(z);
        ctx.strokeStyle = C.gold; ctx.lineWidth = 1.5; ctx.setLineDash([3, 3]);
        ctx.beginPath(); ctx.moveTo(m.x, pt + ih); ctx.lineTo(m.x, m.y); ctx.lineTo(pl, m.y); ctx.stroke(); ctx.setLineDash([]);
        ctx.fillStyle = C.gold; ctx.beginPath(); ctx.moveTo(m.x, m.y - 9); ctx.lineTo(m.x + 9, m.y); ctx.lineTo(m.x, m.y + 9); ctx.lineTo(m.x - 9, m.y); ctx.closePath(); ctx.fill();
        ctx.strokeStyle = C.ink; ctx.lineWidth = 1.5; ctx.stroke();
      }
    }
  }, [points, params, version, focus]);
  return <canvas ref={ref} data-testid="canvas-sigmoid" className="w-full h-full rounded-xl" aria-label="Fixed sigmoid curve: raw score z on the horizontal axis, P(B) vertical. Markers show each point's current score." />;
}

export function LossChart({ history, version }) {
  const ref = useHiDpiCanvas((ctx, cw, ch) => {
    ctx.fillStyle = C.paper; ctx.fillRect(0, 0, cw, ch);
    const pl = 48, pr = 12, pt = 12, pb = 24;
    ctx.font = `10px ${MONO}`; ctx.fillStyle = C.axis;
    const L = history.map((h) => h.loss).filter(Number.isFinite);
    if (L.length < 2) { ctx.textAlign = "center"; ctx.fillText("Log-loss per step appears once training starts", cw / 2, ch / 2); return; }
    const hi = Math.max(...L, Math.LN2) * 1.05, lo = 0;
    const n = history.length - 1;
    const s0 = history[0].step;
    const span = Math.max(1, history[n].step - s0);
    const sx = (i) => pl + ((history[i].step - s0) / span) * (cw - pl - pr);
    const sy = (x) => ch - pb - ((x - lo) / (hi - lo)) * (ch - pt - pb);
    ctx.strokeStyle = C.grid; ctx.lineWidth = 1;
    [0, hi / 2, hi].forEach((t) => { ctx.beginPath(); ctx.moveTo(pl, sy(t)); ctx.lineTo(cw - pr, sy(t)); ctx.stroke(); ctx.textAlign = "right"; ctx.fillText(t.toFixed(2), pl - 6, sy(t) + 3); });
    ctx.strokeStyle = C.gold; ctx.setLineDash([5, 4]); ctx.lineWidth = 1.5;
    const base = history[0].loss * 0.8;
    ctx.beginPath(); ctx.moveTo(pl, sy(base)); ctx.lineTo(cw - pr, sy(base)); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = "#b07f0a"; ctx.textAlign = "left"; ctx.fillText("-20% target", pl + 4, sy(base) - 4);
    ctx.fillStyle = C.axis; ctx.fillText(`step ${s0}`, pl, ch - 6);
    ctx.textAlign = "right"; ctx.fillText(`step ${history[n].step}`, cw - pr, ch - 6);
    ctx.beginPath();
    ctx.moveTo(sx(0), sy(0));
    for (let i = 0; i <= n; i++) if (Number.isFinite(history[i].loss)) ctx.lineTo(sx(i), sy(history[i].loss));
    ctx.lineTo(sx(n), sy(0)); ctx.closePath(); ctx.fillStyle = "rgba(28,122,114,.1)"; ctx.fill();
    ctx.strokeStyle = C.teal; ctx.lineWidth = 2; ctx.lineJoin = "round"; ctx.beginPath();
    let started = false;
    for (let i = 0; i <= n; i++) { const l = history[i].loss; if (!Number.isFinite(l)) continue; started ? ctx.lineTo(sx(i), sy(l)) : ctx.moveTo(sx(i), sy(l)); started = true; }
    ctx.stroke();
  }, [version, history.length]);
  return <canvas ref={ref} data-testid="canvas-loss" className="w-full h-full rounded-xl" aria-label="Binary log-loss versus training step" />;
}
