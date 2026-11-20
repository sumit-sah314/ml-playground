import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Play, Square, RotateCcw, Sigma, Eye, EyeOff, Shuffle, Trash2, Check, Circle } from "lucide-react";
import { meanSquaredError, gradientStep, leastSquares, createDataset } from "@/lib/regression.js";

const C = {
  ink: "#1f2a44", paper: "#fbf8f0", grid: "#e7dfcc", axis: "#8a8170",
  line: "#1c7a72", residual: "#ec6a4c", exact: "#e0a419", point: "#1f2a44",
};
const X0 = 0, X1 = 10, Y0 = -1, Y1 = 11;
const MAX_STEPS = 20000;
const CONVERGED_NORM = 1e-4;
const DIVERGE_LOSS = 1e12;

const fmt = (v, d = 3) => (Number.isFinite(v) ? v.toFixed(d) : "—");
const fmtLoss = (v) => (!Number.isFinite(v) ? "—" : Math.abs(v) >= 1e5 ? v.toExponential(2) : v.toFixed(3));

function useCanvas(draw, deps) {
  const ref = useRef(null);
  const [size, setSize] = useState({ w: 0, h: 0 });
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver(() => {
      const r = el.getBoundingClientRect();
      setSize({ w: r.width, h: r.height });
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  useEffect(() => {
    const el = ref.current;
    if (!el || !size.w) return;
    const dpr = window.devicePixelRatio || 1;
    el.width = Math.round(size.w * dpr);
    el.height = Math.round(size.h * dpr);
    const ctx = el.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, size.w, size.h);
    draw(ctx, size.w, size.h);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [size, ...deps]);
  return { ref, size };
}

const PAD = { l: 36, r: 14, t: 14, b: 30 };
function mapper(w, h) {
  const sx = (x) => PAD.l + ((x - X0) / (X1 - X0)) * (w - PAD.l - PAD.r);
  const sy = (y) => h - PAD.b - ((y - Y0) / (Y1 - Y0)) * (h - PAD.t - PAD.b);
  const ix = (px) => X0 + ((px - PAD.l) / (w - PAD.l - PAD.r)) * (X1 - X0);
  const iy = (py) => Y0 + ((h - PAD.b - py) / (h - PAD.t - PAD.b)) * (Y1 - Y0);
  return { sx, sy, ix, iy };
}

function ScatterPlot({ points, w, b, exact, showResiduals, onAdd, onMove, onRemove, onDragStart }) {
  const [hover, setHover] = useState(-1);
  const dragRef = useRef(-1);
  const { ref, size } = useCanvas((ctx, cw, ch) => {
    const { sx, sy } = mapper(cw, ch);
    ctx.fillStyle = C.paper; ctx.fillRect(0, 0, cw, ch);
    ctx.lineWidth = 1; ctx.strokeStyle = C.grid;
    ctx.font = "11px 'JetBrains Mono', monospace"; ctx.fillStyle = C.axis;
    for (let i = 0; i <= 10; i++) {
      ctx.beginPath(); ctx.moveTo(sx(i), sy(Y0)); ctx.lineTo(sx(i), sy(Y1)); ctx.stroke();
      ctx.textAlign = "center"; ctx.fillText(String(i), sx(i), ch - PAD.b + 16);
    }
    for (let j = 0; j <= 10; j++) {
      ctx.beginPath(); ctx.moveTo(sx(X0), sy(j)); ctx.lineTo(sx(X1), sy(j)); ctx.stroke();
      ctx.textAlign = "right"; ctx.fillText(String(j), PAD.l - 8, sy(j) + 4);
    }
    ctx.strokeStyle = C.axis; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(sx(0), sy(Y0)); ctx.lineTo(sx(0), sy(Y1)); ctx.moveTo(sx(X0), sy(0)); ctx.lineTo(sx(X1), sy(0)); ctx.stroke();

    ctx.save();
    ctx.beginPath(); ctx.rect(PAD.l, PAD.t, cw - PAD.l - PAD.r, ch - PAD.t - PAD.b); ctx.clip();
    if (showResiduals && Number.isFinite(w) && Number.isFinite(b)) {
      ctx.strokeStyle = C.residual; ctx.lineWidth = 2; ctx.setLineDash([]);
      points.forEach((p) => {
        ctx.beginPath(); ctx.moveTo(sx(p.x), sy(p.y)); ctx.lineTo(sx(p.x), sy(w * p.x + b)); ctx.stroke();
      });
    }
    if (exact) {
      ctx.strokeStyle = C.exact; ctx.lineWidth = 2.5; ctx.setLineDash([8, 6]);
      ctx.beginPath(); ctx.moveTo(sx(X0), sy(exact.w * X0 + exact.b)); ctx.lineTo(sx(X1), sy(exact.w * X1 + exact.b)); ctx.stroke();
      ctx.setLineDash([]);
    }
    if (Number.isFinite(w) && Number.isFinite(b)) {
      ctx.strokeStyle = C.line; ctx.lineWidth = 3; ctx.lineCap = "round";
      ctx.beginPath(); ctx.moveTo(sx(X0), sy(w * X0 + b)); ctx.lineTo(sx(X1), sy(w * X1 + b)); ctx.stroke();
    }
    ctx.restore();
    points.forEach((p, i) => {
      const r = i === hover || i === dragRef.current ? 8 : 6;
      ctx.beginPath(); ctx.arc(sx(p.x), sy(p.y), r, 0, Math.PI * 2);
      ctx.fillStyle = i === hover ? C.residual : C.point; ctx.fill();
      ctx.lineWidth = 2.5; ctx.strokeStyle = C.paper; ctx.stroke();
    });
  }, [points, w, b, exact, showResiduals, hover]);

  const toData = (e) => {
    const r = ref.current.getBoundingClientRect();
    const px = e.clientX - r.left, py = e.clientY - r.top;
    const m = mapper(r.width, r.height);
    return { px, py, x: m.ix(px), y: m.iy(py), m };
  };
  const hit = (px, py, m) => {
    let best = -1, bd = 16 * 16;
    points.forEach((p, i) => {
      const d = (m.sx(p.x) - px) ** 2 + (m.sy(p.y) - py) ** 2;
      if (d < bd) { bd = d; best = i; }
    });
    return best;
  };
  const clamp = (v) => Math.min(10, Math.max(0, v));

  return (
    <canvas
      ref={ref}
      data-testid="canvas-scatter"
      className="w-full h-full touch-none rounded-xl"
      style={{ cursor: hover >= 0 ? "grab" : "crosshair" }}
      aria-label="Scatter plot. Click to add a point, drag a point to move it, double-click a point to remove it."
      onPointerDown={(e) => {
        if (!size.w) return;
        const { px, py, x, y, m } = toData(e);
        const i = hit(px, py, m);
        if (i >= 0) {
          dragRef.current = i; onDragStart();
          e.currentTarget.setPointerCapture(e.pointerId);
        } else if (x >= X0 && x <= X1 && y >= 0 && y <= 10) {
          onAdd({ x: clamp(x), y: clamp(y) });
        }
      }}
      onPointerMove={(e) => {
        const { px, py, x, y, m } = toData(e);
        if (dragRef.current >= 0) onMove(dragRef.current, { x: clamp(x), y: clamp(y) });
        else setHover(hit(px, py, m));
      }}
      onPointerUp={() => { dragRef.current = -1; }}
      onPointerCancel={() => { dragRef.current = -1; }}
      onPointerLeave={() => { if (dragRef.current < 0) setHover(-1); }}
      onDoubleClick={(e) => {
        const { px, py, m } = toData(e);
        const i = hit(px, py, m);
        if (i >= 0) { onRemove(i); setHover(-1); }
      }}
    />
  );
}

function LossChart({ history, exactLoss, version }) {
  const { ref } = useCanvas((ctx, cw, ch) => {
    ctx.fillStyle = C.paper; ctx.fillRect(0, 0, cw, ch);
    const pl = 52, pr = 12, pt = 12, pb = 24;
    const finite = history.filter(Number.isFinite);
    ctx.font = "11px 'JetBrains Mono', monospace"; ctx.fillStyle = C.axis;
    if (finite.length < 1) {
      ctx.textAlign = "center";
      ctx.fillText("Press Fit to watch the loss fall", cw / 2, ch / 2);
      return;
    }
    const vals = finite.map((v) => Math.log10(Math.max(v, 1e-6)));
    if (Number.isFinite(exactLoss)) vals.push(Math.log10(Math.max(exactLoss, 1e-6)));
    let lo = Math.min(...vals), hi = Math.max(...vals);
    if (hi - lo < 0.5) { lo -= 0.25; hi += 0.25; }
    const n = Math.max(history.length - 1, 1);
    const sx = (i) => pl + (i / n) * (cw - pl - pr);
    const sy = (v) => ch - pb - ((Math.log10(Math.max(v, 1e-6)) - lo) / (hi - lo)) * (ch - pt - pb);
    ctx.strokeStyle = C.grid; ctx.lineWidth = 1;
    for (let k = Math.ceil(lo); k <= Math.floor(hi); k++) {
      const y = ch - pb - ((k - lo) / (hi - lo)) * (ch - pt - pb);
      ctx.beginPath(); ctx.moveTo(pl, y); ctx.lineTo(cw - pr, y); ctx.stroke();
      ctx.textAlign = "right"; ctx.fillText(k >= 3 || k <= -2 ? `1e${k}` : String(10 ** k), pl - 6, y + 4);
    }
    ctx.textAlign = "left"; ctx.fillText("step 0", pl, ch - 6);
    ctx.textAlign = "right"; ctx.fillText(`step ${history.length - 1}`, cw - pr, ch - 6);
    if (Number.isFinite(exactLoss)) {
      ctx.strokeStyle = C.exact; ctx.setLineDash([6, 5]); ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(pl, sy(exactLoss)); ctx.lineTo(cw - pr, sy(exactLoss)); ctx.stroke(); ctx.setLineDash([]);
    }
    ctx.strokeStyle = C.line; ctx.lineWidth = 2.5; ctx.lineJoin = "round";
    ctx.beginPath();
    const stride = Math.max(1, Math.floor(history.length / 800));
    let started = false;
    for (let i = 0; i < history.length; i += stride) {
      const v = history[i];
      if (!Number.isFinite(v)) break;
      if (!started) { ctx.moveTo(sx(i), sy(v)); started = true; } else ctx.lineTo(sx(i), sy(v));
    }
    const last = history[history.length - 1];
    if (Number.isFinite(last)) ctx.lineTo(sx(history.length - 1), sy(last));
    ctx.stroke();
    if (Number.isFinite(last)) {
      ctx.fillStyle = C.line; ctx.beginPath(); ctx.arc(sx(history.length - 1), sy(last), 4, 0, Math.PI * 2); ctx.fill();
    }
  }, [version, exactLoss]);
  return <canvas ref={ref} data-testid="canvas-loss" className="w-full h-full rounded-xl" aria-label="Loss versus step chart" />;
}

function Slider({ label, value, min, max, step, onChange, color, display, testId }) {
  return (
    <label className="block">
      <div className="flex items-baseline justify-between mb-2">
        <span className="text-sm font-semibold">{label}</span>
        <span className="num text-sm" style={{ color }}>{display ?? fmt(value, 2)}</span>
      </div>
      <input
        type="range" className="rng" min={min} max={max} step={step} value={Number.isFinite(value) ? Math.min(max, Math.max(min, value)) : min}
        style={{ "--thumb": color }} onChange={(e) => onChange(Number(e.target.value))} data-testid={testId}
      />
    </label>
  );
}

const LR_MIN = Math.log10(0.0005), LR_MAX = Math.log10(0.1);

export default function LinearRegressionLesson() {
  const [points, setPoints] = useState(() => createDataset("linear", 15));
  const [w, setW] = useState(0.2);
  const [b, setB] = useState(2);
  const [showResiduals, setShowResiduals] = useState(true);
  const [showExact, setShowExact] = useState(false);
  const [lr, setLr] = useState(0.01);
  const [status, setStatus] = useState("idle"); // idle running converged maxed diverged stopped
  const [stepCount, setStepCount] = useState(0);
  const [gradNorm, setGradNorm] = useState(NaN);
  const [histVersion, setHistVersion] = useState(0);
  const [done, setDone] = useState({ added: false, dragged: false, fitted: false, diverged: false, manual: false });
  const historyRef = useRef([]);
  const rafRef = useRef(0);
  const stateRef = useRef({ points, w, b, lr });
  stateRef.current = { points, w, b, lr };

  const mse = useMemo(() => meanSquaredError(points, w, b), [points, w, b]);
  const exactFit = useMemo(() => leastSquares(points), [points]);
  const empty = points.length === 0;

  const stopLoop = useCallback(() => {
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    rafRef.current = 0;
  }, []);
  useEffect(() => stopLoop, [stopLoop]);

  const cancelFit = useCallback(() => {
    if (rafRef.current) { stopLoop(); setStatus("stopped"); }
  }, [stopLoop]);

  const clearHistory = () => { historyRef.current = []; setStepCount(0); setGradNorm(NaN); setHistVersion((v) => v + 1); };

  const startFit = () => {
    if (empty) return;
    stopLoop();
    const s = stateRef.current;
    let cw = s.w, cb = s.b;
    const pts = s.points, rate = s.lr;
    historyRef.current = [meanSquaredError(pts, cw, cb)];
    let steps = 0;
    setStatus("running");
    setDone((d) => ({ ...d, fitted: true }));
    const tick = () => {
      const perFrame = steps < 60 ? 1 : steps < 400 ? 6 : 40;
      let outcome = null, norm = NaN;
      for (let k = 0; k < perFrame; k++) {
        const r = gradientStep(pts, cw, cb, rate);
        steps++;
        norm = r.gradientNorm;
        if (!Number.isFinite(r.w) || !Number.isFinite(r.b) || !Number.isFinite(r.loss) || r.loss > DIVERGE_LOSS) {
          historyRef.current.push(Number.isFinite(r.loss) ? r.loss : Infinity);
          outcome = "diverged"; break;
        }
        cw = r.w; cb = r.b;
        historyRef.current.push(r.loss);
        if (r.gradientNorm < CONVERGED_NORM) { outcome = "converged"; break; }
        if (steps >= MAX_STEPS) { outcome = "maxed"; break; }
      }
      if (outcome !== "diverged") { setW(cw); setB(cb); }
      setStepCount(steps); setGradNorm(norm); setHistVersion((v) => v + 1);
      if (outcome) {
        rafRef.current = 0; setStatus(outcome);
        if (outcome === "diverged") setDone((d) => ({ ...d, diverged: true }));
        return;
      }
      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
  };

  const manualW = (v) => { cancelFit(); setW(v); setDone((d) => ({ ...d, manual: true })); };
  const manualB = (v) => { cancelFit(); setB(v); setDone((d) => ({ ...d, manual: true })); };

  const editData = (fn) => { cancelFit(); setPoints(fn); };
  const loadDataset = (kind) => {
    stopLoop(); setStatus("idle"); clearHistory();
    setPoints(createDataset(kind, 15)); setW(0.2); setB(2);
  };
  const resetLine = () => { stopLoop(); setStatus("idle"); clearHistory(); setW(0.2); setB(2); };

  // Challenge: manual MSE within 10% of the optimum (plus tiny absolute slack for near-perfect data)
  const goal = exactFit ? exactFit.loss * 1.1 + 0.05 : NaN;
  const gap = Number.isFinite(mse) && exactFit ? Math.max(0, mse - exactFit.loss) : NaN;
  const challengeMet = done.manual && status !== "running" && Number.isFinite(mse) && Number.isFinite(goal) && mse <= goal && points.length >= 2;
  const progress = Number.isFinite(gap) && exactFit ? Math.max(0, Math.min(1, 1 - Math.log10(1 + gap / (goal - exactFit.loss)) / 2)) : 0;

  const statusCopy = {
    idle: { t: "Ready", c: "text-muted-foreground", d: "Pick a learning rate and press Fit. Each step nudges w and b downhill on the loss surface." },
    running: { t: "Learning", c: "text-primary", d: "Each frame runs real gradient steps. Watch the loss chart fall." },
    converged: { t: "Converged", c: "text-primary", d: "The gradient is almost zero: the line can barely improve any further." },
    maxed: { t: `Stopped at ${MAX_STEPS.toLocaleString()} steps`, c: "text-[#b07f0a]", d: "Not fully converged. Small learning rates are safe but slow; the intercept in particular crawls. Press Fit again to continue from here." },
    diverged: { t: "Diverged", c: "text-destructive", d: "The learning rate is too large. Each step overshoots the valley and lands higher up the other side, so the loss explodes. Lower the rate and try again." },
    stopped: { t: "Paused", c: "text-muted-foreground", d: "Fitting was cancelled because you changed something. Press Fit to resume from the current line." },
  }[status];

  const lrPos = (Math.log10(lr) - LR_MIN) / (LR_MAX - LR_MIN);
  const lrHot = lr > 0.025;

  return (
    <div className="space-y-8">
      <header className="rise">
        <p className="num text-xs uppercase tracking-[0.2em] text-accent font-semibold">Lesson 01</p>
        <h1 className="text-4xl md:text-5xl font-extrabold tracking-tight mt-2">Linear Regression</h1>
        <p className="mt-4 max-w-2xl text-lg leading-relaxed text-muted-foreground">
          Draw one straight line that passes as close as possible to every dot. That is the whole idea.
          The machine "learns" by measuring how wrong the line is, then adjusting it to be a little less wrong.
        </p>
      </header>

      <section className="grid lg:grid-cols-[1fr_340px] gap-6 rise" style={{ animationDelay: "80ms" }}>
        <div className="rounded-2xl border bg-card p-3 md:p-4 shadow-[0_1px_0_hsl(38_22%_84%),0_12px_32px_-18px_hsl(222_38%_17%/.35)]">
          <div className="flex flex-wrap items-center justify-between gap-2 px-1 pb-3">
            <div className="flex items-center gap-4 text-xs font-medium">
              <Legend color={C.line} label="your line" />
              <Legend color={C.residual} label="residuals" />
              {showExact && exactFit && <Legend color={C.exact} label="exact fit" dashed />}
            </div>
            <span className="num text-xs text-muted-foreground" data-testid="text-point-count">{points.length} points</span>
          </div>
          <div className="relative aspect-[4/3] w-full">
            <ScatterPlot
              points={points} w={w} b={b} exact={showExact ? exactFit : null} showResiduals={showResiduals}
              onAdd={(p) => { editData((ps) => [...ps, p]); setDone((d) => ({ ...d, added: true })); }}
              onMove={(i, p) => editData((ps) => ps.map((q, j) => (j === i ? p : q)))}
              onRemove={(i) => editData((ps) => ps.filter((_, j) => j !== i))}
              onDragStart={() => { cancelFit(); setDone((d) => ({ ...d, dragged: true })); }}
            />
            {empty && (
              <div className="pointer-events-none absolute inset-0 grid place-items-center">
                <div className="rounded-xl bg-card/90 border px-5 py-4 text-center max-w-xs">
                  <p className="font-semibold">An empty classroom</p>
                  <p className="text-sm text-muted-foreground mt-1">Click anywhere on the grid to place your first point.</p>
                </div>
              </div>
            )}
          </div>
          <p className="text-xs text-muted-foreground px-1 pt-3">Click to add a point. Drag to move one. Double-click to remove it.</p>
          <div className="flex flex-wrap gap-2 px-1 pt-3">
            <Btn onClick={() => loadDataset("linear")} testId="button-dataset-linear"><Shuffle size={14} />Tidy line</Btn>
            <Btn onClick={() => loadDataset("noisy")} testId="button-dataset-noisy"><Shuffle size={14} />Noisy cloud</Btn>
            <Btn onClick={() => loadDataset("negative")} testId="button-dataset-negative"><Shuffle size={14} />Downhill</Btn>
            <Btn onClick={() => { stopLoop(); setStatus("idle"); clearHistory(); setPoints([]); }} testId="button-clear-points"><Trash2 size={14} />Clear</Btn>
            <Btn onClick={() => setShowResiduals((v) => !v)} testId="button-toggle-residuals" active={showResiduals}>
              {showResiduals ? <Eye size={14} /> : <EyeOff size={14} />}Residuals
            </Btn>
          </div>
        </div>

        <aside className="space-y-4">
          <div className="rounded-2xl bg-sidebar text-sidebar-foreground p-5">
            <p className="text-xs uppercase tracking-[0.18em] opacity-70">Mean squared error</p>
            <p key={Math.round((mse || 0) * 10)} className="num text-5xl font-semibold mt-2 text-white" data-testid="text-mse">{fmtLoss(mse)}</p>
            <p className="text-xs opacity-70 mt-2 leading-relaxed">
              {empty ? "Add points to measure error." : "Average of each red residual, squared. Lower is better."}
            </p>
            {exactFit && Number.isFinite(mse) && (
              <p className="num text-xs mt-3 text-[#f3b39f]" data-testid="text-gap">best possible: {fmtLoss(exactFit.loss)}</p>
            )}
          </div>
          <div className="rounded-2xl border bg-card p-5 space-y-5">
            <p className="num text-center text-lg">
              y = <span style={{ color: C.line }} className="font-semibold">{fmt(w, 2)}</span>x {b < 0 ? "−" : "+"}{" "}
              <span style={{ color: C.residual }} className="font-semibold">{fmt(Math.abs(b), 2)}</span>
            </p>
            <Slider label="Slope w" value={w} min={-3} max={3} step={0.01} onChange={manualW} color={C.line} testId="slider-slope" />
            <Slider label="Intercept b" value={b} min={-5} max={12} step={0.01} onChange={manualB} color={C.residual} testId="slider-intercept" />
            <Btn onClick={resetLine} testId="button-reset-line" full><RotateCcw size={14} />Reset line</Btn>
          </div>
        </aside>
      </section>

      <section className="grid md:grid-cols-2 gap-6 rise" style={{ animationDelay: "160ms" }}>
        <div className="rounded-2xl border bg-card p-6">
          <h2 className="text-xl font-bold">The key equation</h2>
          <div className="mt-4 rounded-xl bg-secondary/70 px-4 py-5 num text-center text-lg leading-loose">
            ŷ = <span style={{ color: C.line }}>w</span>·x + <span style={{ color: C.residual }}>b</span>
            <br />
            MSE = (1/n) Σ (y<sub>i</sub> − ŷ<sub>i</sub>)²
          </div>
          <ul className="mt-4 space-y-2 text-sm leading-relaxed text-muted-foreground">
            <li><b className="text-foreground">w (slope)</b> tilts the line: how much y rises when x grows by 1.</li>
            <li><b className="text-foreground">b (intercept)</b> lifts the line: where it crosses x = 0.</li>
            <li><b className="text-foreground">Residual</b> is the vertical gap between a dot and the line. Squaring makes every miss positive and punishes big misses more.</li>
          </ul>
        </div>

        <div className={`rounded-2xl border p-6 transition-colors ${challengeMet ? "bg-primary text-primary-foreground border-primary" : "bg-card"}`} data-testid="card-challenge">
          <p className={`num text-xs uppercase tracking-[0.18em] font-semibold ${challengeMet ? "opacity-80" : "text-accent"}`}>Try this</p>
          <h2 className="text-xl font-bold mt-1">Beat the eyeball test</h2>
          <p className={`text-sm mt-2 leading-relaxed ${challengeMet ? "opacity-90" : "text-muted-foreground"}`}>
            Using only the two sliders, get your MSE within 10% of the best possible line for this data.
            The target moves with the data, so it is always reachable.
          </p>
          <div className="mt-4">
            <div className="flex justify-between num text-xs mb-1.5">
              <span>you {fmtLoss(mse)}</span>
              <span>target ≤ {fmtLoss(goal)}</span>
            </div>
            <div className={`h-2.5 rounded-full overflow-hidden ${challengeMet ? "bg-white/25" : "bg-secondary"}`}>
              <div className="h-full rounded-full transition-[width] duration-300" style={{ width: `${(challengeMet ? 1 : progress) * 100}%`, background: challengeMet ? "#fbf8f0" : C.residual }} />
            </div>
            {challengeMet && <p className="pop mt-3 font-semibold" data-testid="text-challenge-done">Nailed it. Your hand-tuned line is nearly optimal.</p>}
            {!done.manual && !challengeMet && <p className="text-xs mt-3 text-muted-foreground">Move a slider to start.</p>}
          </div>
          <ul className="mt-5 space-y-1.5 text-sm">
            <Check2 ok={done.added} inv={challengeMet}>Add a point by clicking the grid</Check2>
            <Check2 ok={done.dragged} inv={challengeMet}>Drag a point and watch MSE react</Check2>
            <Check2 ok={done.fitted} inv={challengeMet}>Let gradient descent fit the line</Check2>
            <Check2 ok={done.diverged} inv={challengeMet}>Make it diverge with a learning rate that is too big</Check2>
          </ul>
        </div>
      </section>

      <section className="rounded-2xl border bg-card p-4 md:p-6 rise" style={{ animationDelay: "240ms" }}>
        <div className="flex flex-wrap items-end justify-between gap-4">
          <div>
            <h2 className="text-xl font-bold">Let the machine do it: gradient descent</h2>
            <p className="text-sm text-muted-foreground mt-1 max-w-xl">
              Measure which direction makes MSE smaller, take a small step that way, repeat. The learning rate is the step size.
            </p>
          </div>
          <div className="flex flex-wrap gap-2">
            {status === "running" ? (
              <Btn primary onClick={() => { stopLoop(); setStatus("stopped"); }} testId="button-stop-fit"><Square size={14} />Stop</Btn>
            ) : (
              <Btn primary onClick={startFit} disabled={empty} testId="button-fit"><Play size={14} />Fit with gradient descent</Btn>
            )}
            <Btn onClick={() => setShowExact((v) => !v)} disabled={empty} active={showExact} testId="button-show-exact"><Sigma size={14} />{showExact ? "Hide" : "Show"} exact solution</Btn>
          </div>
        </div>

        <div className="grid lg:grid-cols-[280px_1fr] gap-6 mt-6">
          <div className="space-y-4">
            <Slider
              label="Learning rate α" value={lrPos} min={0} max={1} step={0.001}
              onChange={(p) => { cancelFit(); setLr(Number((10 ** (LR_MIN + p * (LR_MAX - LR_MIN))).toPrecision(2))); }}
              color={lrHot ? C.residual : C.line} display={lr.toString()} testId="slider-learning-rate"
            />
            <p className={`text-xs leading-relaxed ${lrHot ? "text-destructive" : "text-muted-foreground"}`}>
              {lrHot
                ? "Careful: with x values up to 10, the slope's gradient is large. Rates above about 0.03 usually overshoot and explode."
                : "0.01 or smaller is stable here. Smaller is safer but slower."}
            </p>
            <div className="rounded-xl bg-secondary/60 p-4 space-y-1">
              <p className={`font-bold ${statusCopy.c}`} data-testid="status-fit">{statusCopy.t}</p>
              <p className="text-xs text-muted-foreground leading-relaxed">{statusCopy.d}</p>
              <div className="num text-xs grid grid-cols-2 gap-y-1 pt-2">
                <span className="text-muted-foreground">step</span><span data-testid="text-steps">{stepCount.toLocaleString()}</span>
                <span className="text-muted-foreground">|gradient|</span><span>{Number.isFinite(gradNorm) ? gradNorm.toExponential(2) : "—"}</span>
              </div>
            </div>
            <p className="num text-[11px] text-muted-foreground leading-relaxed">
              w ← w − α · (2/n) Σ (ŷ − y)·x<br />b ← b − α · (2/n) Σ (ŷ − y)
            </p>
          </div>
          <div className="h-56 md:h-64">
            <LossChart history={historyRef.current} exactLoss={showExact && exactFit ? exactFit.loss : NaN} version={histVersion} />
          </div>
        </div>

        {showExact && exactFit && (
          <div className="mt-6 grid md:grid-cols-[1fr_1.2fr] gap-6 pop" data-testid="panel-exact">
            <div className="rounded-xl border-2 border-dashed p-4" style={{ borderColor: C.exact }}>
              <p className="font-bold">The exact answer (least squares)</p>
              <p className="num text-sm mt-3 leading-loose">
                w = Σ(x − x̄)(y − ȳ) / Σ(x − x̄)²<br />b = ȳ − w·x̄
              </p>
              <p className="text-xs text-muted-foreground mt-2">For a straight line, calculus gives a closed-form answer. Gradient descent matters because most models have no such shortcut.</p>
              {!exactFit.unique && (
                <p className="text-xs mt-3 text-destructive leading-relaxed" data-testid="text-not-unique">
                  No unique answer here: with {points.length === 1 ? "one point" : "every point at the same x"}, infinitely many lines fit equally well. The dashed line is just one representative, flat at the average y.
                </p>
              )}
            </div>
            <table className="w-full num text-sm self-start">
              <thead><tr className="text-xs text-muted-foreground text-left"><th className="py-2 font-normal"></th><th className="font-normal">slope w</th><th className="font-normal">intercept b</th><th className="font-normal">MSE</th></tr></thead>
              <tbody>
                <tr className="border-t"><td className="py-2.5" style={{ color: C.line }}>current line</td><td>{fmt(w)}</td><td>{fmt(b)}</td><td>{fmtLoss(mse)}</td></tr>
                <tr className="border-t"><td className="py-2.5 text-[#b07f0a]">exact</td><td>{fmt(exactFit.w)}</td><td>{fmt(exactFit.b)}</td><td>{fmtLoss(exactFit.loss)}</td></tr>
                <tr className="border-t text-muted-foreground"><td className="py-2.5">difference</td><td>{fmt(w - exactFit.w)}</td><td>{fmt(b - exactFit.b)}</td><td>{fmtLoss(mse - exactFit.loss)}</td></tr>
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  );
}

function Legend({ color, label, dashed }) {
  return (
    <span className="flex items-center gap-1.5">
      <span className="inline-block w-5 h-0 border-t-[3px]" style={{ borderColor: color, borderStyle: dashed ? "dashed" : "solid" }} />
      {label}
    </span>
  );
}

function Btn({ children, onClick, disabled, primary, active, full, testId }) {
  return (
    <button
      type="button" onClick={onClick} disabled={disabled} data-testid={testId}
      className={`inline-flex items-center justify-center gap-1.5 rounded-full px-4 py-2 text-sm font-semibold transition-all active:scale-[0.97] disabled:opacity-40 disabled:pointer-events-none ${full ? "w-full" : ""} ${
        primary ? "bg-primary text-primary-foreground hover:brightness-110 shadow-[0_6px_16px_-8px_hsl(176_58%_30%)]"
          : active ? "bg-foreground text-background" : "bg-secondary hover:bg-[hsl(40_38%_84%)]"
      }`}
    >
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
