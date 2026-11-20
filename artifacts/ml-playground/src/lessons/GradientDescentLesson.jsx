import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Play, Pause, RotateCcw, StepForward, Check, Circle, Crosshair } from "lucide-react";
import { meanSquaredError, gradientStep } from "@/lib/regression.js";
import { SURFACE_POINTS, surfaceGeometry, contourRing } from "@/lib/loss-surface.js";

const C = {
  ink: "#1f2a44", paper: "#fbf8f0", grid: "#e7dfcc", axis: "#8a8170",
  teal: "#1c7a72", coral: "#ec6a4c", gold: "#e0a419", plum: "#7a5a9e",
};
const MAX_STEPS = 600;
const CONVERGED_NORM = 1e-5;
const PAD = { l: 52, r: 16, t: 16, b: 40 };

const fmt = (v, d = 3) => (Number.isFinite(v) ? v.toFixed(d) : "—");
const fmtLoss = (v) => (!Number.isFinite(v) ? "∞" : Math.abs(v) >= 1e5 ? v.toExponential(2) : v.toFixed(4));
const fmtRate = (v) => (Number.isFinite(v) ? (v >= 0.01 ? v.toFixed(4) : v.toPrecision(3)) : "—");

function niceTicks(lo, hi, n = 6) {
  const raw = (hi - lo) / n;
  const p = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * p).find((s) => s >= raw) || raw;
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-12; v += step) out.push(Math.abs(v) < step * 1e-9 ? 0 : v);
  return out;
}

function useHiDpiCanvas(draw, deps) {
  const ref = useRef(null);
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
    draw(ctx, size.w, size.h);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [size, ...deps]);
  return ref;
}

function makeView(bounds, cw, ch) {
  const { w0, w1, b0, b1 } = bounds;
  const sx = (w) => PAD.l + ((w - w0) / (w1 - w0)) * (cw - PAD.l - PAD.r);
  const sy = (b) => ch - PAD.b - ((b - b0) / (b1 - b0)) * (ch - PAD.t - PAD.b);
  const iw = (px) => w0 + ((px - PAD.l) / (cw - PAD.l - PAD.r)) * (w1 - w0);
  const ib = (py) => b0 + ((ch - PAD.b - py) / (ch - PAD.t - PAD.b)) * (b1 - b0);
  const inside = (p) => p.w >= w0 && p.w <= w1 && p.b >= b0 && p.b <= b1;
  return { sx, sy, iw, ib, inside };
}

function lerpColor(t) {
  const a = [28, 122, 114], z = [236, 106, 76];
  const c = a.map((v, i) => Math.round(v + (z[i] - v) * t));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}

function ContourMap({ geometry, rings, bounds, path, start, onPickStart, version }) {
  const ref = useHiDpiCanvas((ctx, cw, ch) => {
    const v = makeView(bounds, cw, ch);
    ctx.fillStyle = C.paper; ctx.fillRect(0, 0, cw, ch);
    ctx.font = "11px 'JetBrains Mono', monospace";
    // grid + ticks
    ctx.lineWidth = 1; ctx.strokeStyle = C.grid; ctx.fillStyle = C.axis;
    niceTicks(bounds.w0, bounds.w1).forEach((t) => {
      ctx.beginPath(); ctx.moveTo(v.sx(t), PAD.t); ctx.lineTo(v.sx(t), ch - PAD.b); ctx.stroke();
      ctx.textAlign = "center"; ctx.fillText(+t.toPrecision(4) + "", v.sx(t), ch - PAD.b + 15);
    });
    niceTicks(bounds.b0, bounds.b1).forEach((t) => {
      ctx.beginPath(); ctx.moveTo(PAD.l, v.sy(t)); ctx.lineTo(cw - PAD.r, v.sy(t)); ctx.stroke();
      ctx.textAlign = "right"; ctx.fillText(+t.toPrecision(4) + "", PAD.l - 6, v.sy(t) + 4);
    });
    ctx.fillStyle = C.ink; ctx.font = "600 12px 'JetBrains Mono', monospace";
    ctx.textAlign = "center"; ctx.fillText("w (slope)", PAD.l + (cw - PAD.l - PAD.r) / 2, ch - 8);
    ctx.save(); ctx.translate(13, PAD.t + (ch - PAD.t - PAD.b) / 2); ctx.rotate(-Math.PI / 2); ctx.fillText("b (intercept)", 0, 0); ctx.restore();

    ctx.save();
    ctx.beginPath(); ctx.rect(PAD.l, PAD.t, cw - PAD.l - PAD.r, ch - PAD.t - PAD.b); ctx.clip();
    // contour fills from outside in
    rings.forEach((r, k) => {
      const t = k / Math.max(1, rings.length - 1);
      ctx.beginPath();
      r.pts.forEach((p, i) => (i ? ctx.lineTo(v.sx(p.w), v.sy(p.b)) : ctx.moveTo(v.sx(p.w), v.sy(p.b))));
      ctx.closePath();
      ctx.fillStyle = `rgba(28,122,114,${0.03 + t * 0.07})`; ctx.fill();
      ctx.strokeStyle = `rgba(31,42,68,${0.25 + t * 0.35})`; ctx.lineWidth = 1.2; ctx.stroke();
    });
    // contour labels
    ctx.font = "10px 'JetBrains Mono', monospace"; ctx.fillStyle = C.axis; ctx.textAlign = "left";
    rings.forEach((r, k) => {
      if (k % 2) return;
      const p = r.pts[Math.floor(r.pts.length * 0.12)];
      if (p && v.inside(p)) ctx.fillText(fmtLoss(r.loss).replace(/0+$/, "").replace(/\.$/, ""), v.sx(p.w) + 3, v.sy(p.b) - 3);
    });
    // path
    const visible = path.filter((p) => Number.isFinite(p.w) && Number.isFinite(p.b));
    const n = visible.length;
    ctx.lineCap = "round"; ctx.lineJoin = "round";
    for (let i = 1; i < n; i++) {
      const a = visible[i - 1], c = visible[i];
      const ax = v.sx(a.w), ay = v.sy(a.b), cx = v.sx(c.w), cy = v.sy(c.b);
      if (![ax, ay, cx, cy].every((q) => Math.abs(q) < 1e7)) break;
      ctx.strokeStyle = lerpColor(i / Math.max(1, n - 1)); ctx.lineWidth = 2.5;
      ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(cx, cy); ctx.stroke();
    }
    visible.forEach((p, i) => {
      if (i === 0 || i === n - 1 || !v.inside(p)) return;
      ctx.beginPath(); ctx.arc(v.sx(p.w), v.sy(p.b), 2.6, 0, Math.PI * 2);
      ctx.fillStyle = lerpColor(i / Math.max(1, n - 1)); ctx.fill();
    });
    // direction arrow on last segment
    if (n >= 2) {
      const a = visible[n - 2], c = visible[n - 1];
      if (v.inside(c)) {
        const ang = Math.atan2(v.sy(c.b) - v.sy(a.b), v.sx(c.w) - v.sx(a.w));
        const x = v.sx(c.w), y = v.sy(c.b);
        ctx.fillStyle = C.coral; ctx.beginPath();
        ctx.moveTo(x - Math.cos(ang) * 9, y - Math.sin(ang) * 9);
        ctx.lineTo(x - Math.cos(ang) * 20 - Math.sin(ang) * 6, y - Math.sin(ang) * 20 + Math.cos(ang) * 6);
        ctx.lineTo(x - Math.cos(ang) * 20 + Math.sin(ang) * 6, y - Math.sin(ang) * 20 - Math.cos(ang) * 6);
        ctx.closePath(); ctx.fill();
      }
    }
    // optimum
    const m = geometry.minimum;
    ctx.strokeStyle = C.gold; ctx.lineWidth = 3;
    const ox = v.sx(m.w), oy = v.sy(m.b);
    ctx.beginPath(); ctx.moveTo(ox - 7, oy - 7); ctx.lineTo(ox + 7, oy + 7); ctx.moveTo(ox + 7, oy - 7); ctx.lineTo(ox - 7, oy + 7); ctx.stroke();
    // start
    ctx.beginPath(); ctx.arc(v.sx(start.w), v.sy(start.b), 7, 0, Math.PI * 2);
    ctx.fillStyle = C.paper; ctx.fill(); ctx.strokeStyle = C.ink; ctx.lineWidth = 2.5; ctx.stroke();
    // current
    const cur = path[path.length - 1];
    if (cur && Number.isFinite(cur.w) && Number.isFinite(cur.b) && v.inside(cur) && path.length > 1) {
      ctx.beginPath(); ctx.arc(v.sx(cur.w), v.sy(cur.b), 7.5, 0, Math.PI * 2);
      ctx.fillStyle = C.coral; ctx.fill(); ctx.strokeStyle = C.paper; ctx.lineWidth = 2.5; ctx.stroke();
    }
    ctx.restore();
    // off-map indicator
    if (cur && path.length > 1 && !(Number.isFinite(cur.w) && Number.isFinite(cur.b) && v.inside(cur))) {
      const prev = path.length > 1 ? path[path.length - 2] : start;
      let dx = cur.w - (bounds.w0 + bounds.w1) / 2, dy = cur.b - (bounds.b0 + bounds.b1) / 2;
      if (!Number.isFinite(dx) || !Number.isFinite(dy)) { dx = cur.w - prev.w || 1; dy = cur.b - prev.b || 0; }
      const ux = dx / (bounds.w1 - bounds.w0), uy = dy / (bounds.b1 - bounds.b0);
      const k = 0.5 / Math.max(Math.abs(ux), Math.abs(uy), 1e-12);
      const ex = v.sx((bounds.w0 + bounds.w1) / 2 + dx * k * 0.92), ey = v.sy((bounds.b0 + bounds.b1) / 2 + dy * k * 0.92);
      const ang = Math.atan2(-uy, ux);
      ctx.save(); ctx.translate(ex, ey); ctx.rotate(ang);
      ctx.fillStyle = C.coral; ctx.beginPath(); ctx.moveTo(10, 0); ctx.lineTo(-8, -8); ctx.lineTo(-8, 8); ctx.closePath(); ctx.fill();
      ctx.restore();
      const label = "left the map";
      ctx.font = "600 11px 'JetBrains Mono', monospace";
      const tw = ctx.measureText(label).width + 12;
      const lx = Math.min(cw - PAD.r - tw, Math.max(PAD.l, ex - tw / 2)), ly = Math.min(ch - PAD.b - 22, Math.max(PAD.t, ey + (uy > 0 ? 12 : -32)));
      ctx.fillStyle = C.coral; ctx.beginPath(); ctx.roundRect(lx, ly, tw, 20, 10); ctx.fill();
      ctx.fillStyle = C.paper; ctx.textAlign = "left"; ctx.fillText(label, lx + 6, ly + 14);
    }
  }, [geometry, rings, bounds, start, version]);

  return (
    <canvas
      ref={ref} data-testid="canvas-contour"
      className="w-full h-full touch-none rounded-xl cursor-crosshair"
      aria-label="Contour map of mean squared error over slope w and intercept b. Click to choose a starting point."
      onPointerDown={(e) => {
        const r = e.currentTarget.getBoundingClientRect();
        const v = makeView(bounds, r.width, r.height);
        const p = { w: v.iw(e.clientX - r.left), b: v.ib(e.clientY - r.top) };
        if (v.inside(p)) onPickStart(p);
      }}
    />
  );
}

function LossMini({ path, minLoss, version }) {
  const ref = useHiDpiCanvas((ctx, cw, ch) => {
    ctx.fillStyle = C.paper; ctx.fillRect(0, 0, cw, ch);
    const pl = 54, pr = 10, pt = 10, pb = 22;
    ctx.font = "10px 'JetBrains Mono', monospace"; ctx.fillStyle = C.axis;
    const L = path.map((p) => p.loss);
    if (L.length < 2) { ctx.textAlign = "center"; ctx.fillText("Loss per step appears here", cw / 2, ch / 2); return; }
    const lg = (x) => Math.log10(Math.max(x, 1e-12));
    const vals = L.filter(Number.isFinite).map(lg); vals.push(lg(minLoss));
    let lo = Math.min(...vals), hi = Math.max(...vals);
    if (hi - lo < 0.3) { lo -= 0.15; hi += 0.15; }
    const n = L.length - 1;
    const sx = (i) => pl + (i / n) * (cw - pl - pr);
    const sy = (x) => ch - pb - ((lg(x) - lo) / (hi - lo)) * (ch - pt - pb);
    ctx.textAlign = "right";
    ctx.fillText(hi >= 4 ? `1e${hi.toFixed(0)}` : (10 ** hi).toPrecision(3), pl - 6, pt + 8);
    ctx.fillText((10 ** lo).toPrecision(3), pl - 6, ch - pb);
    ctx.textAlign = "left"; ctx.fillText("step 0", pl, ch - 6);
    ctx.textAlign = "right"; ctx.fillText(`step ${n}`, cw - pr, ch - 6);
    ctx.strokeStyle = C.gold; ctx.setLineDash([5, 4]); ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(pl, sy(minLoss)); ctx.lineTo(cw - pr, sy(minLoss)); ctx.stroke(); ctx.setLineDash([]);
    ctx.strokeStyle = C.teal; ctx.lineWidth = 2; ctx.lineJoin = "round"; ctx.beginPath();
    for (let i = 0; i <= n; i++) { if (!Number.isFinite(L[i])) break; i ? ctx.lineTo(sx(i), sy(L[i])) : ctx.moveTo(sx(i), sy(L[i])); }
    ctx.stroke();
  }, [version, minLoss]);
  return <canvas ref={ref} data-testid="canvas-loss-steps" className="w-full h-full rounded-xl" aria-label="Loss versus step, log scale" />;
}

const STATUS = {
  idle: { t: "Ready", c: "text-muted-foreground", d: "Pick a start and a learning rate, then press Run or Step." },
  running: { t: "Descending", c: "text-primary", d: "Each tick applies one real gradient step to both w and b at once." },
  paused: { t: "Paused", c: "text-muted-foreground", d: "Run or Step to continue from the current point." },
  converged: { t: "Converged", c: "text-primary", d: "The gradient is essentially zero. You are at the bottom of the bowl." },
  zigzag: { t: "Converged, with a zig-zag", c: "text-[#7a5a9e]", d: "The steps overshot the valley floor and bounced side to side, but each bounce was smaller than the last, so it still settled at the minimum." },
  diverged: { t: "Diverged, stopped for safety", c: "text-destructive", d: "Every step overshot by more than it corrected, so each bounce grew. The loss exploded and we stopped the run before the numbers became meaningless." },
  maxed: { t: "Step budget reached", c: "text-[#b07f0a]", d: "Not converged yet. Small rates are safe but slow. Press Resume for another 600 steps." },
};

export default function GradientDescentLesson() {
  const geometry = useMemo(() => surfaceGeometry(SURFACE_POINTS), []);
  const { minimum, stableRate, presets, eigenvectors } = geometry;
  const lossAt = useCallback((w, b) => meanSquaredError(SURFACE_POINTS, w, b), []);

  const { rings, bounds } = useMemo(() => {
    const span = Math.max(lossAt(0, 0) - minimum.loss, 1) * 1.6;
    const levels = Array.from({ length: 9 }, (_, k) => span * 0.5 ** k);
    const rs = levels.map((lv) => ({ loss: minimum.loss + lv, pts: contourRing(lv, geometry, 180) }));
    let w0 = Infinity, w1 = -Infinity, b0 = Infinity, b1 = -Infinity;
    rs[0].pts.forEach((p) => { w0 = Math.min(w0, p.w); w1 = Math.max(w1, p.w); b0 = Math.min(b0, p.b); b1 = Math.max(b1, p.b); });
    w0 = Math.min(w0, 0); w1 = Math.max(w1, 0); b0 = Math.min(b0, 0); b1 = Math.max(b1, 0);
    const pw = (w1 - w0) * 0.06, pb = (b1 - b0) * 0.06;
    return { rings: rs, bounds: { w0: w0 - pw, w1: w1 + pw, b0: b0 - pb, b1: b1 + pb } };
  }, [geometry, lossAt, minimum.loss]);

  const defaultStart = useMemo(() => ({ w: +(bounds.w0 + (bounds.w1 - bounds.w0) * 0.12).toPrecision(3), b: +(bounds.b0 + (bounds.b1 - bounds.b0) * 0.85).toPrecision(3) }), [bounds]);

  const [start, setStart] = useState(defaultStart);
  const [lr, setLr] = useState(presets.converge);
  const [rateText, setRateText] = useState(fmtRate(presets.converge));
  const [speed, setSpeed] = useState(24);
  const [status, setStatus] = useState("idle");
  const [version, setVersion] = useState(0);
  const [tried, setTried] = useState({ converge: false, oscillate: false, diverge: false });

  const pathRef = useRef([{ w: start.w, b: start.b, loss: lossAt(start.w, start.b) }]);
  const flipsRef = useRef(0);
  const pauseAtRef = useRef(MAX_STEPS);
  const rafRef = useRef(0);
  const cfg = useRef({ lr, speed });
  cfg.current = { lr, speed };

  const stopLoop = useCallback(() => { if (rafRef.current) cancelAnimationFrame(rafRef.current); rafRef.current = 0; }, []);
  useEffect(() => stopLoop, [stopLoop]);

  const resetPath = useCallback((s) => {
    stopLoop();
    pathRef.current = [{ w: s.w, b: s.b, loss: lossAt(s.w, s.b) }];
    flipsRef.current = 0;
    pauseAtRef.current = MAX_STEPS;
    setStatus("idle"); setVersion((v) => v + 1);
  }, [stopLoop, lossAt]);

  useEffect(() => {
    const prevT = document.title;
    const meta = document.querySelector('meta[name="description"]');
    const prevD = meta?.getAttribute("content");
    document.title = "Gradient Descent — Lesson 2 · ML Playground";
    meta?.setAttribute("content", "Watch gradient descent walk an MSE contour map. Change the learning rate and see it converge, zig-zag, or diverge.");
    return () => { document.title = prevT; if (meta && prevD != null) meta.setAttribute("content", prevD); };
  }, []);

  // One honest step. Returns outcome or null.
  const doStep = useCallback(() => {
    const path = pathRef.current;
    const cur = path[path.length - 1];
    const rate = cfg.current.lr;
    const r = gradientStep(SURFACE_POINTS, cur.w, cur.b, rate);
    const steps = path.length;
    const big = eigenvectors[0];
    const proj = (p) => (p.w - minimum.w) * big[0] + (p.b - minimum.b) * big[1];
    if (Number.isFinite(r.w) && Number.isFinite(r.b) && Math.sign(proj(r)) !== Math.sign(proj(cur)) && Math.abs(proj(cur)) > 1e-9) flipsRef.current++;
    path.push({ w: r.w, b: r.b, loss: r.loss });
    const startLoss = path[0].loss;
    if (!Number.isFinite(r.w) || !Number.isFinite(r.b) || !Number.isFinite(r.loss) || r.loss > Math.max(startLoss, 1) * 1e6) return "diverged";
    const g = gradientStep(SURFACE_POINTS, r.w, r.b, 0).gradientNorm;
    if (g < CONVERGED_NORM) return flipsRef.current >= 3 ? "zigzag" : "converged";
    if (steps >= pauseAtRef.current) return "maxed";
    return null;
  }, [eigenvectors, minimum.w, minimum.b]);

  const finish = useCallback((o) => {
    setStatus(o);
    if (o === "converged") setTried((t) => ({ ...t, converge: true }));
    if (o === "zigzag") setTried((t) => ({ ...t, oscillate: true }));
    if (o === "diverged") setTried((t) => ({ ...t, diverge: true }));
  }, []);

  const terminal = ["converged", "zigzag", "diverged"].includes(status);

  const run = () => {
    if (terminal) return;
    stopLoop();
    if (status === "maxed") pauseAtRef.current = pathRef.current.length - 1 + MAX_STEPS;
    setStatus("running");
    let acc = 0, last = performance.now();
    const tick = (now) => {
      acc += ((now - last) / 1000) * cfg.current.speed; last = now;
      let outcome = null;
      while (acc >= 1 && !outcome) { acc -= 1; outcome = doStep(); }
      setVersion((v) => v + 1);
      if (outcome) { rafRef.current = 0; finish(outcome); return; }
      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
  };
  const pause = () => { stopLoop(); setStatus("paused"); };
  const single = () => {
    if (terminal) return;
    stopLoop();
    if (status === "maxed") pauseAtRef.current = pathRef.current.length - 1 + MAX_STEPS;
    const o = doStep();
    setVersion((v) => v + 1);
    if (o) finish(o); else setStatus("paused");
  };

  const applyRate = (v) => {
    if (!Number.isFinite(v) || v <= 0) return;
    setLr(v); setRateText(fmtRate(v)); resetPath(start);
  };
  const pickStart = (p) => {
    const s = { w: +p.w.toPrecision(4), b: +p.b.toPrecision(4) };
    setStart(s); resetPath(s);
  };

  const path = pathRef.current;
  const cur = path[path.length - 1];
  const stepCount = path.length - 1;
  const ratio = lr / stableRate;
  const onMap = Number.isFinite(cur.w) && Number.isFinite(cur.b) && cur.w >= bounds.w0 && cur.w <= bounds.w1 && cur.b >= bounds.b0 && cur.b <= bounds.b1;
  const startLoss = path[0].loss;
  const progress = Number.isFinite(cur.loss) ? Math.max(0, Math.min(1, 1 - Math.log10(1 + (cur.loss - minimum.loss)) / Math.log10(1 + Math.max(startLoss - minimum.loss, 1e-9)))) : 0;
  const regime = ratio < 0.5 ? { t: "gentle", c: C.teal } : ratio < 1 ? { t: "zig-zag zone", c: C.plum } : ratio === 1 ? { t: "boundary: undamped oscillation", c: C.plum } : { t: "overshoot: diverges", c: C.coral };
  const s = STATUS[status];
  const allDone = tried.converge && tried.oscillate && tried.diverge;

  const LR_LO = Math.log10(stableRate * 0.02), LR_HI = Math.log10(stableRate * 1.25);
  const lrPos = (Math.log10(lr) - LR_LO) / (LR_HI - LR_LO);

  return (
    <div className="space-y-8">
      <header className="rise">
        <p className="num text-xs uppercase tracking-[0.2em] text-accent font-semibold">Lesson 02</p>
        <h1 className="text-4xl md:text-5xl font-extrabold tracking-tight mt-2">Gradient Descent</h1>
        <p className="mt-4 max-w-2xl text-lg leading-relaxed text-muted-foreground">
          Every possible line is a point on this map: across is the slope w, up is the intercept b. The rings connect lines with equal error, like a
          contour map of a valley. Gradient descent feels the slope where it stands and steps downhill. The learning rate decides how big each step is.
        </p>
        <details className="mt-3 text-xs text-muted-foreground">
          <summary className="cursor-pointer font-semibold">Which data defines this loss surface?</summary>
          <p className="mt-2 max-w-2xl leading-relaxed">
            Every MSE value uses the same nine practice points, independent of Lesson 1:
            {" "}{SURFACE_POINTS.map((p) => `(${p.x}, ${p.y})`).join(", ")}.
            The gold cross is their exact least-squares minimum.
          </p>
        </details>
      </header>

      <section className="grid lg:grid-cols-[1fr_330px] gap-6 rise" style={{ animationDelay: "80ms" }}>
        <div className="rounded-2xl border bg-card p-3 md:p-4 shadow-[0_1px_0_hsl(38_22%_84%),0_12px_32px_-18px_hsl(222_38%_17%/.35)]">
          <div className="flex flex-wrap items-center gap-x-4 gap-y-2 px-1 pb-3 text-xs font-medium">
            <span className="flex items-center gap-1.5"><span className="w-3 h-3 rounded-full border-[2.5px]" style={{ borderColor: C.ink, background: C.paper }} />start</span>
            <span className="flex items-center gap-1.5"><span className="w-3 h-3 rounded-full" style={{ background: C.coral }} />current</span>
            <span className="flex items-center gap-1.5"><span className="num font-bold" style={{ color: C.gold }}>×</span>exact minimum</span>
            <span className="flex items-center gap-1.5"><span className="w-6 h-1 rounded-full" style={{ background: `linear-gradient(90deg,${C.teal},${C.coral})` }} />path, early to late</span>
            <span className="flex items-center gap-1.5"><span className="w-4 h-3 rounded-full border" style={{ borderColor: C.ink }} />equal-MSE ring</span>
          </div>
          <div className="relative aspect-[4/3] w-full">
            <ContourMap geometry={geometry} rings={rings} bounds={bounds} path={path} start={start} onPickStart={pickStart} version={version} />
          </div>
          <p className="text-xs text-muted-foreground px-1 pt-3 flex items-center gap-1.5"><Crosshair size={13} />Click anywhere on the map to start from there. Each ring halves the extra error of the one outside it.</p>
          <div className="flex flex-wrap gap-2 px-1 pt-3">
            {status === "running"
              ? <Btn primary onClick={pause} testId="button-pause"><Pause size={14} />Pause</Btn>
              : <Btn primary onClick={run} disabled={terminal} testId="button-run"><Play size={14} />{stepCount ? "Resume" : "Run"}</Btn>}
            <Btn onClick={single} disabled={terminal} testId="button-step"><StepForward size={14} />One step</Btn>
            <Btn onClick={() => resetPath(start)} testId="button-reset"><RotateCcw size={14} />Reset path</Btn>
            <Btn onClick={() => { setStart(defaultStart); resetPath(defaultStart); }} testId="button-reset-start">Default start</Btn>
          </div>
        </div>

        <aside className="space-y-4">
          <div className="rounded-2xl bg-sidebar text-sidebar-foreground p-5">
            <p className="text-xs uppercase tracking-[0.18em] opacity-70">Mean squared error</p>
            <p className="num text-4xl font-semibold mt-2 text-white break-all" data-testid="text-mse">{fmtLoss(cur.loss)}</p>
            <p className="num text-xs mt-2 text-[#f3b39f]">minimum: {fmtLoss(minimum.loss)}</p>
            <div className="num text-xs grid grid-cols-2 gap-y-1 mt-4 opacity-90">
              <span className="opacity-60">step</span><span data-testid="text-step">{stepCount}</span>
              <span className="opacity-60">w</span><span data-testid="text-w">{Math.abs(cur.w) > 1e5 ? cur.w.toExponential(2) : fmt(cur.w, 4)}</span>
              <span className="opacity-60">b</span><span data-testid="text-b">{Math.abs(cur.b) > 1e5 ? cur.b.toExponential(2) : fmt(cur.b, 4)}</span>
              <span className="opacity-60">on map</span><span data-testid="text-onmap">{onMap ? "yes" : "no, off-screen"}</span>
            </div>
            <div className="mt-4 h-1.5 rounded-full bg-white/10 overflow-hidden">
              <div className="h-full rounded-full transition-[width] duration-200" style={{ width: `${progress * 100}%`, background: status === "diverged" ? C.coral : "#7fd1c7" }} />
            </div>
            <p className="text-[11px] opacity-60 mt-1.5">progress toward the minimum (log scale)</p>
          </div>
          <div className="rounded-xl bg-secondary/60 p-4">
            <p className={`font-bold ${s.c}`} data-testid="status-descent">{s.t}</p>
            <p className="text-xs text-muted-foreground leading-relaxed mt-1">{s.d}</p>
          </div>
          <div className="rounded-2xl border bg-card p-4 h-40"><LossMini path={path} minLoss={minimum.loss} version={version} /></div>
        </aside>
      </section>

      <section className="rounded-2xl border bg-card p-5 md:p-6 rise grid lg:grid-cols-[1.1fr_1fr] gap-8" style={{ animationDelay: "140ms" }}>
        <div className="space-y-5">
          <div>
            <h2 className="text-xl font-bold">Learning rate α</h2>
            <p className="text-sm text-muted-foreground mt-1">For this data, anything below <span className="num text-foreground">{fmtRate(stableRate)}</span> is stable. That limit is 2 divided by the steepest curvature of the bowl.</p>
          </div>
          <div className="grid grid-cols-3 gap-2">
            {[["converge", "Converge", C.teal], ["oscillate", "Oscillate", C.plum], ["diverge", "Diverge", C.coral]].map(([k, label, col]) => {
              const on = Math.abs(lr - presets[k]) < 1e-12;
              return (
                <button key={k} type="button" data-testid={`button-preset-${k}`} onClick={() => applyRate(presets[k])}
                  className="rounded-xl border-2 px-2 py-2.5 text-left transition-all active:scale-[0.97]"
                  style={{ borderColor: col, background: on ? col : "transparent", color: on ? C.paper : C.ink }}>
                  <span className="block text-sm font-bold">{label}</span>
                  <span className="num block text-[11px] opacity-80">α = {fmtRate(presets[k])}</span>
                </button>
              );
            })}
          </div>
          <label className="block">
            <div className="flex items-baseline justify-between mb-2">
              <span className="text-sm font-semibold">Fine-tune α</span>
              <span className="num text-xs font-semibold" style={{ color: regime.c }} data-testid="text-regime">{(ratio * 100).toFixed(1)}% of limit · {regime.t}</span>
            </div>
            <input type="range" className="rng" min={0} max={1} step={0.001} value={Math.min(1, Math.max(0, lrPos))} style={{ "--thumb": regime.c }}
              onChange={(e) => applyRate(Number((10 ** (LR_LO + Number(e.target.value) * (LR_HI - LR_LO))).toPrecision(4)))} data-testid="slider-learning-rate" />
          </label>
          <div className="flex items-center gap-2">
            <span className="text-sm font-semibold">Exact value</span>
            <input className="num flex-1 rounded-lg border bg-background px-3 py-1.5 text-sm" inputMode="decimal" value={rateText} data-testid="input-learning-rate"
              onChange={(e) => setRateText(e.target.value)}
              onBlur={() => { const v = Number(rateText); if (Number.isFinite(v) && v > 0) applyRate(v); else setRateText(fmtRate(lr)); }}
              onKeyDown={(e) => { if (e.key === "Enter") e.currentTarget.blur(); }} />
          </div>
          <label className="block">
            <div className="flex items-baseline justify-between mb-2"><span className="text-sm font-semibold">Animation speed</span><span className="num text-xs">{speed} steps/s</span></div>
            <input type="range" className="rng" min={1} max={30} step={1} value={speed} onChange={(e) => setSpeed(Number(e.target.value))} data-testid="slider-speed" />
          </label>
        </div>
        <div className="space-y-4">
          <h2 className="text-xl font-bold">Starting point</h2>
          <NumField label="start w" value={start.w} min={bounds.w0} max={bounds.w1} color={C.teal} testId="start-w" onChange={(v) => { const s2 = { ...start, w: v }; setStart(s2); resetPath(s2); }} />
          <NumField label="start b" value={start.b} min={bounds.b0} max={bounds.b1} color={C.coral} testId="start-b" onChange={(v) => { const s2 = { ...start, b: v }; setStart(s2); resetPath(s2); }} />
          <p className="text-xs text-muted-foreground leading-relaxed">Changing the start or the rate clears the path so every trail you see comes from one setting.</p>
        </div>
      </section>

      <section className="grid md:grid-cols-2 gap-6 rise" style={{ animationDelay: "200ms" }}>
        <div className="rounded-2xl border bg-card p-6">
          <h2 className="text-xl font-bold">The key equations</h2>
          <div className="mt-4 rounded-xl bg-secondary/70 px-4 py-5 num text-center text-[15px] leading-loose">
            MSE(w, b) = (1/n) Σ (w·x<sub>i</sub> + b − y<sub>i</sub>)²<br />
            ∂MSE/∂w = (2/n) Σ (ŷ<sub>i</sub> − y<sub>i</sub>)·x<sub>i</sub><br />
            ∂MSE/∂b = (2/n) Σ (ŷ<sub>i</sub> − y<sub>i</sub>)<br />
            <span style={{ color: C.teal }}>w</span> ← w − α·∂MSE/∂w &nbsp; <span style={{ color: C.coral }}>b</span> ← b − α·∂MSE/∂b
          </div>
          <ul className="mt-4 space-y-2 text-sm leading-relaxed text-muted-foreground">
            <li>Both partial derivatives are computed at the same point, then w and b update together.</li>
            <li>The gradient points straight uphill, across the rings. We step the opposite way.</li>
            <li><b className="text-foreground">A zig-zag is not failure.</b> If each bounce across the valley is shorter than the last, the path still lands on the minimum.</li>
            <li>Past the limit, each bounce is longer than the last. That is divergence.</li>
          </ul>
        </div>

        <div className={`rounded-2xl border p-6 transition-colors ${allDone ? "bg-primary text-primary-foreground border-primary" : "bg-card"}`} data-testid="card-challenge">
          <p className={`num text-xs uppercase tracking-[0.18em] font-semibold ${allDone ? "opacity-80" : "text-accent"}`}>Try this</p>
          <h2 className="text-xl font-bold mt-1">Three personalities of one rule</h2>
          <p className={`text-sm mt-2 leading-relaxed ${allDone ? "opacity-90" : "text-muted-foreground"}`}>
            Same equation, same data, different α. Run it to the end three times: once calmly, once with a zig-zag, and once off a cliff.
          </p>
          <ul className="mt-5 space-y-1.5 text-sm">
            <Check2 ok={tried.converge} inv={allDone}>Converge smoothly to the minimum</Check2>
            <Check2 ok={tried.oscillate} inv={allDone}>Zig-zag across the valley and still converge</Check2>
            <Check2 ok={tried.diverge} inv={allDone}>Push α past the limit until it diverges</Check2>
          </ul>
          {allDone && <p className="pop mt-4 font-semibold" data-testid="text-challenge-done">All three. You now know the one knob that most often breaks training, and why.</p>}
          {allDone && <Btn onClick={() => setTried({ converge: false, oscillate: false, diverge: false })} testId="button-challenge-reset">Start over</Btn>}
        </div>
      </section>
    </div>
  );
}

function NumField({ label, value, min, max, color, onChange, testId }) {
  return (
    <label className="block">
      <div className="flex items-baseline justify-between mb-2"><span className="text-sm font-semibold">{label}</span><span className="num text-sm" style={{ color }}>{fmt(value, 3)}</span></div>
      <input type="range" className="rng" min={min} max={max} step={(max - min) / 400} value={Math.min(max, Math.max(min, value))} style={{ "--thumb": color }}
        onChange={(e) => onChange(+Number(e.target.value).toPrecision(4))} data-testid={`slider-${testId}`} />
    </label>
  );
}

function Btn({ children, onClick, disabled, primary, testId }) {
  return (
    <button type="button" onClick={onClick} disabled={disabled} data-testid={testId}
      className={`mt-0 inline-flex items-center justify-center gap-1.5 rounded-full px-4 py-2 text-sm font-semibold transition-all active:scale-[0.97] disabled:opacity-40 disabled:pointer-events-none ${
        primary ? "bg-primary text-primary-foreground hover:brightness-110 shadow-[0_6px_16px_-8px_hsl(176_58%_30%)]" : "bg-secondary text-foreground hover:bg-[hsl(40_38%_84%)]"}`}>
      {children}
    </button>
  );
}

function Check2({ ok, inv, children }) {
  return (
    <li className={`flex items-center gap-2 ${ok ? "" : "opacity-70"}`}>
      {ok ? <Check size={15} className={inv ? "" : "text-primary"} /> : <Circle size={13} />}
      <span className={ok ? "line-through decoration-1" : ""}>{children}</span>
    </li>
  );
}
