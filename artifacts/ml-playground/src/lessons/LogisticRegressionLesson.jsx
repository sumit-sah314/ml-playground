import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Play, Pause, RotateCcw, StepForward, Check, Circle, Trash2, Shuffle, MousePointer2, Eraser } from "lucide-react";
import {
  logisticLoss, logisticStep, logisticGradient, classificationAccuracy, makeClassificationDataset, logisticScore, sigmoid,
} from "@/lib/logistic.js";
import { C, LIM, PlaneCanvas, SigmoidCanvas, LossChart, planeView } from "./logistic/LogisticCanvases.jsx";

const BUDGET = 1500;
const FLAT_NORM = 1e-4;
const ZERO = { wx: 0, wy: 0, b: 0 };
const fmt = (v, d = 3) => (!Number.isFinite(v) ? "—" : Math.abs(v) >= 1e4 ? v.toExponential(2) : v.toFixed(d));
const fmtRate = (v) => (v >= 0.01 ? v.toFixed(3) : v.toPrecision(2));
const pct = (v) => (Number.isFinite(v) ? `${(v * 100).toFixed(1)}%` : "—");
const LR_LO = Math.log10(0.001), LR_HI = 0;

const STATUS = {
  idle: { t: "Ready to train", c: "text-muted-foreground", d: "Press Train to run gradient descent, or One step to watch a single update." },
  running: { t: "Training", c: "text-primary", d: "Each tick computes the gradient at the current weights, then moves wx, wy and b together." },
  paused: { t: "Paused", c: "text-muted-foreground", d: "Resume or step to continue from these exact weights." },
  budget: { t: "Step budget reached", c: "text-[#b07f0a]", d: "Stopped after this run's step allowance. This is not a claim of convergence. Press Resume for another batch." },
  budgetSep: { t: "Step budget reached, still improving", c: "text-[#b07f0a]", d: "Every point is classified correctly, so there is no finite best answer: growing the weights keeps sharpening the sigmoid and shaving the loss toward zero forever. We stopped at the budget, not at a minimum." },
  flat: { t: "Gradient is nearly flat", c: "text-primary", d: "The gradient norm fell below 0.0001, so further changes are very small. This is a numerical stopping rule, not proof of a finite minimum: separable data can keep improving as weights grow." },
  diverged: { t: "Stopped: numbers left safe range", c: "text-destructive", d: "A number became non-finite or a weight exceeded the safety limit. Lower the learning rate and reset the model." },
};

export default function LogisticRegressionLesson() {
  const [points, setPoints] = useState(() => makeClassificationDataset("separable"));
  const [params, setParams] = useState(ZERO);
  const [lr, setLr] = useState(0.05);
  const [speed, setSpeed] = useState(30);
  const [status, setStatus] = useState("idle");
  const [version, setVersion] = useState(0);
  const [tool, setTool] = useState(0);
  const [hover, setHover] = useState(null);
  const [probe, setProbe] = useState(null);
  const [dragIdx, setDragIdx] = useState(null);
  const [placed, setPlaced] = useState({ a: false, b: false });

  const pointsRef = useRef(points);
  const paramsRef = useRef(params);
  const stepRef = useRef(0);
  const historyRef = useRef([]);
  const pauseAtRef = useRef(BUDGET);
  const rafRef = useRef(0);
  const cfg = useRef({ lr, speed });
  cfg.current = { lr, speed };
  const dragRef = useRef(null);

  const counts = useMemo(() => ({ a: points.filter((p) => p.label === 0).length, b: points.filter((p) => p.label === 1).length }), [points]);
  const canTrain = counts.a > 0 && counts.b > 0;

  const stopLoop = useCallback(() => { if (rafRef.current) cancelAnimationFrame(rafRef.current); rafRef.current = 0; }, []);
  useEffect(() => stopLoop, [stopLoop]);

  useEffect(() => {
    const prevT = document.title;
    const meta = document.querySelector('meta[name="description"]');
    const prevD = meta?.getAttribute("content");
    document.title = "Logistic Regression — Lesson 3 · ML Playground";
    meta?.setAttribute("content", "Place two classes of points, then train logistic regression with gradient descent. Watch the decision boundary move and log-loss fall.");
    return () => { document.title = prevT; if (meta && prevD != null) meta.setAttribute("content", prevD); };
  }, []);

  const snapshot = (pts, prm) => {
    const both = pts.some((p) => p.label === 0) && pts.some((p) => p.label === 1);
    return both ? [{ step: stepRef.current, loss: logisticLoss(pts, prm), acc: classificationAccuracy(pts, prm) }] : [];
  };

  const restartHistory = useCallback((pts, prm) => {
    stopLoop();
    historyRef.current = snapshot(pts, prm);
    pauseAtRef.current = stepRef.current + BUDGET;
    setStatus("idle");
    setVersion((v) => v + 1);
  }, [stopLoop]);

  const commitPoints = useCallback((next) => {
    pointsRef.current = next;
    setPoints(next);
    restartHistory(next, paramsRef.current);
  }, [restartHistory]);

  const resetModel = () => {
    paramsRef.current = ZERO; setParams(ZERO); stepRef.current = 0;
    restartHistory(pointsRef.current, ZERO);
  };

  // init history for the starting dataset
  useEffect(() => { historyRef.current = snapshot(pointsRef.current, paramsRef.current); setVersion((v) => v + 1); }, []);

  const doStep = useCallback(() => {
    const pts = pointsRef.current;
    const r = logisticStep(pts, paramsRef.current, cfg.current.lr);
    const next = { wx: r.wx, wy: r.wy, b: r.b };
    stepRef.current += 1;
    const ok = [r.wx, r.wy, r.b, r.loss].every(Number.isFinite) && Math.max(Math.abs(r.wx), Math.abs(r.wy), Math.abs(r.b)) < 1e8;
    if (!ok) return "diverged";
    paramsRef.current = next;
    historyRef.current.push({ step: stepRef.current, loss: r.loss, acc: classificationAccuracy(pts, next) });
    if (historyRef.current.length > 4000) historyRef.current = historyRef.current.filter((_, i) => i === 0 || i % 2 === 1);
    const g = logisticGradient(pts, next).norm;
    if (g < FLAT_NORM) return "flat";
    if (stepRef.current >= pauseAtRef.current) return pts.every((point) => (point.label === 1 ? 1 : -1) * logisticScore(point, next) > 0) ? "budgetSep" : "budget";
    return null;
  }, []);

  const sync = () => { setParams(paramsRef.current); setVersion((v) => v + 1); };
  const extendIfStopped = () => {
    if (status === "budget" || status === "budgetSep" || status === "flat") pauseAtRef.current = stepRef.current + BUDGET;
  };
  const blocked = !canTrain || status === "diverged";

  const train = () => {
    if (blocked) return;
    stopLoop(); extendIfStopped();
    setStatus("running");
    let acc = 0, last = performance.now();
    const tick = (now) => {
      acc += ((now - last) / 1000) * cfg.current.speed; last = now;
      acc = Math.min(acc, 400);
      let out = null;
      while (acc >= 1 && !out) { acc -= 1; out = doStep(); }
      sync();
      if (out) { rafRef.current = 0; setStatus(out); return; }
      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
  };
  const pause = () => { stopLoop(); setStatus("paused"); };
  const single = () => {
    if (blocked) return;
    stopLoop(); extendIfStopped();
    const out = doStep(); sync();
    setStatus(out || "paused");
  };

  // ---- pointer interaction on plane
  const locate = (e) => {
    const r = e.currentTarget.getBoundingClientRect();
    const v = planeView(r.width, r.height);
    const px = e.clientX - r.left, py = e.clientY - r.top;
    const x = v.ix(px), y = v.iy(py);
    let hit = null, best = 13;
    pointsRef.current.forEach((p, i) => { const d = Math.hypot(v.sx(p.x) - px, v.sy(p.y) - py); if (d < best) { best = d; hit = i; } });
    return { x, y, hit, inside: Math.abs(x) <= LIM && Math.abs(y) <= LIM };
  };
  const round = (v) => Math.round(Math.max(-LIM, Math.min(LIM, v)) * 100) / 100;
  const addPoint = (x, y, label) => {
    commitPoints([...pointsRef.current, { x: round(x), y: round(y), label }]);
    setPlaced((s) => (label ? { ...s, b: true } : { ...s, a: true }));
  };
  const removePoint = (i) => { commitPoints(pointsRef.current.filter((_, j) => j !== i)); setHover(null); };

  const onDown = (e) => {
    const L = locate(e);
    if (!L.inside) return;
    const mouse = e.pointerType === "mouse";
    if (mouse) {
      if (e.button === 2) { e.preventDefault(); addPoint(L.x, L.y, 1); return; }
      if (e.button !== 0) return;
      if (e.altKey) { e.preventDefault(); if (L.hit != null) removePoint(L.hit); return; }
      if (L.hit != null) { dragRef.current = L.hit; setDragIdx(L.hit); e.currentTarget.setPointerCapture(e.pointerId); return; }
      addPoint(L.x, L.y, 0); return;
    }
    if (tool === 2) { if (L.hit != null) removePoint(L.hit); return; }
    if (L.hit != null) { dragRef.current = L.hit; setDragIdx(L.hit); e.currentTarget.setPointerCapture(e.pointerId); return; }
    addPoint(L.x, L.y, tool);
  };
  const onMove = (e) => {
    const L = locate(e);
    if (dragRef.current != null) {
      const i = dragRef.current;
      commitPoints(pointsRef.current.map((p, j) => (j === i ? { ...p, x: round(L.x), y: round(L.y) } : p)));
      return;
    }
    setHover(L.hit);
    setProbe(L.inside ? { x: L.x, y: L.y } : null);
  };
  const onUp = () => { dragRef.current = null; setDragIdx(null); };
  const onLeave = () => { if (dragRef.current == null) { setHover(null); setProbe(null); } };

  const loadSample = (kind) => { commitPoints(makeClassificationDataset(kind)); setPlaced({ a: false, b: false }); };
  const clearAll = () => { commitPoints([]); setPlaced({ a: false, b: false }); };

  // ---- derived display
  const history = historyRef.current;
  const loss = logisticLoss(points, params);
  const accuracy = classificationAccuracy(points, params);
  const grad = canTrain ? logisticGradient(points, params) : null;
  const baseline = history[0]?.loss;
  const selected = dragIdx ?? hover;
  const focus = selected != null && points[selected] ? points[selected] : probe;
  const focusZ = focus ? logisticScore(focus, params) : NaN;
  const focusP = Number.isFinite(focusZ) ? sigmoid(focusZ) : NaN;
  const drop = Number.isFinite(baseline) && baseline > 0 && Number.isFinite(loss) ? 1 - loss / baseline : 0;
  const goalDone = placed.a && placed.b && canTrain && history.length > 1 && drop >= 0.2;
  const s = STATUS[status];
  const lrPos = (Math.log10(lr) - LR_LO) / (LR_HI - LR_LO);

  const setRate = (v) => setLr(Math.min(1, Math.max(0.001, Number(v.toPrecision(3)))));

  let missing = null;
  if (points.length === 0) missing = "The plane is empty. Left click to drop class A, right click for class B, or load a sample.";
  else if (!counts.a) missing = "Only class B so far. Logistic regression needs both classes to learn where one ends. Left click (or pick A) to add class A.";
  else if (!counts.b) missing = "Only class A so far. Add at least one class B point with a right click (or pick B) before training.";

  return (
    <div className="space-y-8">
      <header className="rise">
        <p className="num text-xs uppercase tracking-[0.2em] text-accent font-semibold">Lesson 03</p>
        <h1 className="text-4xl md:text-5xl font-extrabold tracking-tight mt-2">Logistic Regression</h1>
        <p className="mt-4 max-w-2xl text-lg leading-relaxed text-muted-foreground">
          Now the answer is a yes or no: is this point class A or class B? The model scores every spot on the plane with a straight-line formula,
          squeezes that score into a probability with the sigmoid, and draws its decision boundary where the probability is exactly one half.
        </p>
      </header>

      <section className="grid lg:grid-cols-[1fr_330px] gap-6 rise" style={{ animationDelay: "80ms" }}>
        <div className="rounded-2xl border bg-card p-3 md:p-4 shadow-[0_1px_0_hsl(38_22%_84%),0_12px_32px_-18px_hsl(222_38%_17%/.35)]">
          <div className="flex flex-wrap items-center gap-x-4 gap-y-2 px-1 pb-3 text-xs font-medium">
            <span className="flex items-center gap-1.5"><span className="w-3 h-3 rounded-full border-2" style={{ background: C.teal, borderColor: C.paper }} />A, label 0 (circle)</span>
            <span className="flex items-center gap-1.5"><span className="w-3 h-3" style={{ background: C.coral }} />B, label 1 (square)</span>
            <span className="flex items-center gap-1.5"><span className="w-6 border-t-2 border-dashed" style={{ borderColor: C.ink }} />boundary, P(B) = 0.5</span>
            <span className="flex items-center gap-1.5"><span className="w-8 h-3 rounded-sm" style={{ background: "linear-gradient(90deg,#9bc7c1,#fbf8f0,#f3b9a8)" }} />shading = P(B)</span>
            <span className="flex items-center gap-1.5"><span style={{ color: C.coral }} className="font-bold">→</span>points to B side</span>
          </div>
          <div className="relative aspect-square w-full max-h-[620px]">
            <PlaneCanvas points={points} params={params} version={version} selected={selected} probe={probe}
              onPointerDown={onDown} onPointerMove={onMove} onPointerUp={onUp} onPointerLeave={onLeave} />
            {missing && (
              <div className="pointer-events-none absolute inset-x-6 top-1/2 -translate-y-1/2 mx-auto max-w-sm rounded-2xl bg-sidebar/90 text-sidebar-foreground p-4 text-sm leading-relaxed pop" data-testid="text-missing-class">
                {missing}
              </div>
            )}
          </div>
          <div className="flex flex-wrap items-center justify-between gap-3 px-1 pt-3">
            <p className="text-xs text-muted-foreground flex items-center gap-1.5"><MousePointer2 size={13} />Mouse: left click A, right click B. Drag to move; Alt + click a point to delete.</p>
            <div className="flex items-center gap-1 rounded-full bg-secondary p-1 text-xs font-semibold" role="radiogroup" aria-label="Class for touch and pen">
              <span className="px-2 opacity-60">Touch / pen</span>
              {[[0, "A"], [1, "B"], [2, "Erase"]].map(([k, l]) => (
                <button key={k} type="button" role="radio" aria-checked={tool === k} onClick={() => setTool(k)} data-testid={`button-tool-${l.toLowerCase()}`}
                  className={`rounded-full px-3 py-1 transition-colors flex items-center gap-1 ${tool === k ? "text-white" : "hover:bg-background"}`}
                  style={tool === k ? { background: k === 0 ? C.teal : k === 1 ? C.coral : C.ink } : undefined}>
                  {k === 2 && <Eraser size={12} />}{l}
                </button>
              ))}
            </div>
          </div>
          <div className="flex flex-wrap items-center gap-2 px-1 pt-3">
            {status === "running"
              ? <Btn primary onClick={pause} testId="button-pause"><Pause size={14} />Pause</Btn>
              : <Btn primary onClick={train} disabled={blocked} testId="button-train"><Play size={14} />{status === "idle" ? "Train" : "Resume"}</Btn>}
            <Btn onClick={single} disabled={blocked || status === "running"} testId="button-step"><StepForward size={14} />One step</Btn>
            <Btn onClick={resetModel} testId="button-reset-model"><RotateCcw size={14} />Reset model</Btn>
            <span className="w-px h-6 bg-border mx-1" />
            <Btn onClick={() => loadSample("separable")} testId="button-sample-separable"><Shuffle size={14} />Separable</Btn>
            <Btn onClick={() => loadSample("overlap")} testId="button-sample-overlap"><Shuffle size={14} />Overlapping</Btn>
            <Btn onClick={clearAll} disabled={!points.length} testId="button-clear"><Trash2 size={14} />Clear</Btn>
            <span className="num text-xs ml-auto" data-testid="text-counts">
              <span style={{ color: C.teal }} className="font-semibold">A {counts.a}</span> · <span style={{ color: C.coral }} className="font-semibold">B {counts.b}</span>
            </span>
          </div>
        </div>

        <aside className="space-y-4">
          <div className="rounded-2xl bg-sidebar text-sidebar-foreground p-5">
            <p className="text-xs uppercase tracking-[0.18em] opacity-70">Binary log-loss</p>
            <p className="num text-4xl font-semibold mt-2 text-white break-all" data-testid="text-loss">{fmt(loss, 4)}</p>
            <p className="num text-xs mt-2 text-[#f3b39f]">
              {Number.isFinite(baseline) ? `start of this run ${fmt(baseline, 4)} · ${drop >= 0 ? "down" : "up"} ${(Math.abs(drop) * 100).toFixed(1)}%` : "needs both classes"}
            </p>
            <div className="num text-xs grid grid-cols-2 gap-y-1 mt-4 opacity-90">
              <span className="opacity-60">step</span><span data-testid="text-step">{stepRef.current}</span>
              <span className="opacity-60">accuracy</span><span data-testid="text-accuracy">{pct(accuracy)}</span>
              <span className="opacity-60">wx</span><span data-testid="text-wx">{fmt(params.wx, 4)}</span>
              <span className="opacity-60">wy</span><span data-testid="text-wy">{fmt(params.wy, 4)}</span>
              <span className="opacity-60">b</span><span data-testid="text-b">{fmt(params.b, 4)}</span>
              <span className="opacity-60">|gradient|</span><span data-testid="text-grad">{grad ? fmt(grad.norm, 5) : "—"}</span>
            </div>
          </div>
          <div className="rounded-xl bg-secondary/60 p-4">
            <p className={`font-bold ${canTrain ? s.c : "text-muted-foreground"}`} data-testid="status-training">{canTrain ? s.t : "Training locked"}</p>
            <p className="text-xs text-muted-foreground leading-relaxed mt-1">{canTrain ? s.d : "Add at least one point of each class. Changing data always clears the loss history, so the chart never mixes datasets."}</p>
          </div>
          <div className="rounded-2xl border bg-card p-4 space-y-4">
            <label className="block">
              <div className="flex items-baseline justify-between mb-2"><span className="text-sm font-semibold">Learning rate α</span><span className="num text-xs font-semibold" data-testid="text-rate">{fmtRate(lr)}</span></div>
              <input type="range" className="rng" min={0} max={1} step={0.001} value={lrPos} data-testid="slider-learning-rate"
                onChange={(e) => setRate(10 ** (LR_LO + Number(e.target.value) * (LR_HI - LR_LO)))} />
              <div className="flex gap-1.5 mt-2">
                {[0.01, 0.05, 0.2, 1].map((v) => (
                  <button key={v} type="button" onClick={() => setRate(v)} data-testid={`button-rate-${v}`}
                    className={`num flex-1 rounded-lg border px-1 py-1 text-[11px] transition-colors ${Math.abs(lr - v) < 1e-9 ? "bg-primary text-primary-foreground border-primary" : "hover:bg-secondary"}`}>{v}</button>
                ))}
              </div>
            </label>
            <label className="block">
              <div className="flex items-baseline justify-between mb-2"><span className="text-sm font-semibold">Animation speed</span><span className="num text-xs">{speed} steps/s</span></div>
              <input type="range" className="rng" min={2} max={240} step={2} value={speed} onChange={(e) => setSpeed(Number(e.target.value))} data-testid="slider-speed" />
            </label>
            <p className="text-[11px] text-muted-foreground leading-relaxed">Each run allows {BUDGET} steps, then pauses. Resume grants another {BUDGET}.</p>
          </div>
        </aside>
      </section>

      <section className="grid md:grid-cols-2 gap-6 rise" style={{ animationDelay: "140ms" }}>
        <div className="rounded-2xl border bg-card p-4 md:p-5">
          <div className="flex items-baseline justify-between px-1">
            <h2 className="text-lg font-bold">The sigmoid stays put</h2>
            <span className="num text-xs text-muted-foreground" data-testid="text-focus">
              {focus ? `z = ${fmt(focusZ, 2)} → P(B) = ${fmt(focusP, 3)}` : "hover a point or the plane"}
            </span>
          </div>
          <div className="h-60 mt-3"><SigmoidCanvas points={points} params={params} version={version} focus={focus} /></div>
          <p className="text-xs text-muted-foreground leading-relaxed mt-3 px-1">
            The curve σ(z) = 1/(1+e<sup>−z</sup>) never changes. Training moves the <b className="text-foreground">points' scores</b> along it: As slide left toward 0, Bs slide right toward 1.
            The gold dashes mark the threshold z = 0, p = 0.5. Scores past ±8 are pinned to the edge.
          </p>
        </div>
        <div className="rounded-2xl border bg-card p-4 md:p-5">
          <div className="flex items-baseline justify-between px-1">
            <h2 className="text-lg font-bold">Log-loss going down</h2>
            <span className="num text-xs text-muted-foreground">{history.length > 1 ? `${history[history.length - 1].step - history[0].step} steps this dataset` : "no steps yet"}</span>
          </div>
          <div className="h-60 mt-3"><LossChart history={history} version={version} /></div>
          <p className="text-xs text-muted-foreground leading-relaxed mt-3 px-1">
            Real mean binary cross-entropy after every step, no smoothing. Starting from all-zero weights every point gets P(B) = 0.5, so the loss begins at ln 2 ≈ 0.693.
            {history.length > 1 && history[history.length - 1].step - history[0].step > history.length - 1 && " Older chart samples are thinned on long runs; the step axis preserves their actual spacing."}
          </p>
        </div>
      </section>

      <section className="grid md:grid-cols-2 gap-6 rise" style={{ animationDelay: "200ms" }}>
        <div className="rounded-2xl border bg-card p-6">
          <h2 className="text-xl font-bold">The key equations</h2>
          <div className="mt-4 rounded-xl bg-secondary/70 px-4 py-5 num text-center text-[14px] leading-loose">
            z = w<sub>x</sub>·x1 + w<sub>y</sub>·x2 + b<br />
            p = P(B) = σ(z) = 1 / (1 + e<sup>−z</sup>)<br />
            boundary: w<sub>x</sub>·x1 + w<sub>y</sub>·x2 + b = 0<br />
            L = −(1/n) Σ [ y·ln p + (1−y)·ln(1−p) ]<br />
            ∂L/∂w<sub>x</sub> = (1/n) Σ (p<sub>i</sub> − y<sub>i</sub>)·x1<sub>i</sub> &nbsp; ∂L/∂b = (1/n) Σ (p<sub>i</sub> − y<sub>i</sub>)
          </div>
          <ul className="mt-4 space-y-2 text-sm leading-relaxed text-muted-foreground">
            <li>Each point pulls on the weights in proportion to its error p − y. Confident, correct points barely pull at all.</li>
            <li>Log-loss punishes confident mistakes hard: predicting 0.01 for a true B costs ln 100 ≈ 4.6.</li>
            <li>If wy is 0 the boundary is vertical; if wx is 0 it is horizontal; if both are 0 there is no boundary at all.</li>
            <li><b className="text-foreground">Separable data has no finite winner.</b> Once every point is on the right side, making the weights bigger always lowers the loss a little more, so training never truly finishes.</li>
          </ul>
        </div>

        <div className={`rounded-2xl border p-6 transition-colors ${goalDone ? "bg-primary text-primary-foreground border-primary" : "bg-card"}`} data-testid="card-challenge">
          <p className={`num text-xs uppercase tracking-[0.18em] font-semibold ${goalDone ? "opacity-80" : "text-accent"}`}>Try this</p>
          <h2 className="text-xl font-bold mt-1">Your own data, 20% less wrong</h2>
          <p className={`text-sm mt-2 leading-relaxed ${goalDone ? "opacity-90" : "text-muted-foreground"}`}>
            Place at least one A and one B yourself (add them to a sample or start from Clear). Then train until the log-loss is at least 20% below
            where this dataset started.
            {" "}Try two separate clusters. If you already trained the data, Reset model gives you a fresh baseline; completely contradictory labels may not allow a 20% improvement.
          </p>
          <ul className="mt-5 space-y-1.5 text-sm">
            <Check2 ok={placed.a} inv={goalDone}>Place a class A point (left click)</Check2>
            <Check2 ok={placed.b} inv={goalDone}>Place a class B point (right click)</Check2>
            <Check2 ok={goalDone} inv={goalDone}>
              Cut the loss by 20% <span className="num text-xs opacity-80">(now {Number.isFinite(drop) && history.length > 1 ? `${(drop * 100).toFixed(1)}%` : "0.0%"})</span>
            </Check2>
          </ul>
          <div className={`mt-4 h-1.5 rounded-full overflow-hidden ${goalDone ? "bg-white/20" : "bg-secondary"}`}>
            <div className="h-full rounded-full transition-[width] duration-200" style={{ width: `${Math.max(0, Math.min(1, drop / 0.2)) * 100}%`, background: goalDone ? C.paper : C.teal }} />
          </div>
          {goalDone && <p className="pop mt-4 font-semibold" data-testid="text-challenge-done">Done. Now drag one point across the boundary and watch the loss history restart.</p>}
        </div>
      </section>
    </div>
  );
}

function Btn({ children, onClick, disabled, primary, testId }) {
  return (
    <button type="button" onClick={onClick} disabled={disabled} data-testid={testId}
      className={`inline-flex items-center justify-center gap-1.5 rounded-full px-4 py-2 text-sm font-semibold transition-all active:scale-[0.97] disabled:opacity-40 disabled:pointer-events-none ${
        primary ? "bg-primary text-primary-foreground hover:brightness-110 shadow-[0_6px_16px_-8px_hsl(176_58%_30%)]" : "bg-secondary text-foreground hover:bg-[hsl(40_38%_84%)]"}`}>
      {children}
    </button>
  );
}

function Check2({ ok, inv, children }) {
  return (
    <li className={`flex items-center gap-2 ${ok ? "" : "opacity-70"}`}>
      {ok ? <Check size={15} className={inv ? "" : "text-primary"} /> : <Circle size={13} />}
      <span>{children}</span>
    </li>
  );
}
