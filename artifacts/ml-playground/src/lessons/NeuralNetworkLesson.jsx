import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Play, Pause, RotateCcw, StepForward, Shuffle, Plus, Minus, Check, Circle } from "lucide-react";
import { createNetwork, forwardNetwork, networkMetrics, trainEpoch, makeNeuralDataset } from "@/lib/neural-network.js";
import { C } from "./logistic/LogisticCanvases.jsx";
import { DecisionMap, NetworkDiagram, EpochLossChart, mapView } from "./neural/NeuralCanvases.jsx";

const BUDGET = 3000;
const FRAME_MS = 12;
const SAFE = 1e6;
const KINDS = [["circles", "Circles"], ["xor", "XOR"], ["spirals", "Spirals"]];
const fmt = (v, d = 4) => (!Number.isFinite(v) ? "—" : Math.abs(v) >= 1e4 ? v.toExponential(2) : v.toFixed(d));
const pct = (v) => (Number.isFinite(v) ? `${(v * 100).toFixed(1)}%` : "—");
const LR_LO = Math.log10(0.003), LR_HI = Math.log10(3);

const STATUS = {
  idle: { t: "Fresh random weights", c: "text-muted-foreground", d: "Press Train, or One epoch to watch a single full-batch gradient step." },
  running: { t: "Training", c: "text-primary", d: "Each epoch: forward pass on every point, backprop the loss, update every weight at once." },
  paused: { t: "Paused", c: "text-muted-foreground", d: "Resume continues from these exact weights." },
  budget: { t: "Epoch budget reached", c: "text-[#b07f0a]", d: `Paused after ${BUDGET} epochs in this run. Not a claim of convergence: the loss may still be falling. Resume for another ${BUDGET}.` },
  diverged: { t: "Stopped: numbers left safe range", c: "text-destructive", d: "A loss or weight became non-finite or huge. Lower the learning rate and reset the weights." },
};

const flatWeights = (net) => net.layers.flatMap((l) => [...l.weights.flat(), ...l.biases]);

export default function NeuralNetworkLesson() {
  const [kind, setKind] = useState("circles");
  const [dataSeed, setDataSeed] = useState(42);
  const [hidden, setHidden] = useState([8, 8]);
  const [activation, setActivation] = useState("tanh");
  const [netSeed, setNetSeed] = useState(42);
  const [lr, setLr] = useState(0.1);
  const [speed, setSpeed] = useState(20);
  const [status, setStatus] = useState("idle");
  const [version, setVersion] = useState(0);
  const [probe, setProbe] = useState(null);
  const [notice, setNotice] = useState(null);
  const [tried, setTried] = useState({ tanh: 0, relu: 0 });

  const points = useMemo(() => makeNeuralDataset(kind, dataSeed), [kind, dataSeed]);
  const netRef = useRef(null);
  const epochRef = useRef(0);
  const historyRef = useRef([]);
  const pauseAtRef = useRef(BUDGET);
  const rafRef = useRef(0);
  const pointsRef = useRef(points);
  pointsRef.current = points;
  const cfg = useRef({ lr, speed });
  cfg.current = { lr, speed };

  const stopLoop = useCallback(() => { if (rafRef.current) cancelAnimationFrame(rafRef.current); rafRef.current = 0; }, []);
  useEffect(() => stopLoop, [stopLoop]);

  useEffect(() => {
    const prevT = document.title;
    const meta = document.querySelector('meta[name="description"]');
    const prevD = meta?.getAttribute("content");
    document.title = "Neural Networks — Lesson 4 · ML Playground";
    meta?.setAttribute("content", "Train a small neural network from scratch with backpropagation in your browser. Watch hidden units bend a decision boundary around circles, XOR and spirals.");
    return () => { document.title = prevT; if (meta && prevD != null) meta.setAttribute("content", prevD); };
  }, []);

  // (Re)build network whenever architecture/data/seed changes: cancels any running loop.
  const hiddenKey = hidden.join(",");
  if (!netRef.current || netRef.current.__key !== `${hiddenKey}|${activation}|${netSeed}|${kind}|${dataSeed}`) {
    stopLoop();
    const net = createNetwork(hidden, activation, netSeed);
    net.__key = `${hiddenKey}|${activation}|${netSeed}|${kind}|${dataSeed}`;
    netRef.current = net;
    epochRef.current = 0;
    const m = networkMetrics(net, points);
    historyRef.current = [{ epoch: 0, loss: m.loss, acc: m.accuracy }];
    pauseAtRef.current = BUDGET;
  }
  const net = netRef.current;

  const changed = (msg) => { stopLoop(); setStatus("idle"); setNotice(msg); setVersion((v) => v + 1); };
  const pickKind = (k) => { if (k === kind) return; setKind(k); changed("Dataset changed: weights re-randomised and loss history cleared."); };
  const regen = () => { setDataSeed((s) => s + 1); changed("New sample drawn: weights re-randomised and loss history cleared."); };
  const setLayers = (h) => { setHidden(h); changed("Architecture changed: a brand-new network with fresh random weights, history cleared."); };
  const pickAct = (a) => { if (a === activation) return; setActivation(a); changed(`Switched to ${a === "relu" ? "ReLU" : "tanh"}: fresh weights, history cleared.`); };
  const resetWeights = () => { setNetSeed((s) => s + 1); changed("Weights reset to a new random draw. History cleared."); };

  const doEpoch = useCallback(() => {
    const n = netRef.current;
    const r = trainEpoch(n, pointsRef.current, cfg.current.lr);
    epochRef.current += 1;
    const big = flatWeights(n).some((w) => !Number.isFinite(w) || Math.abs(w) > SAFE);
    if (!Number.isFinite(r.loss) || big) return "diverged";
    const h = historyRef.current;
    h.push({ epoch: epochRef.current, loss: r.loss, acc: r.accuracy });
    if (h.length > 3000) historyRef.current = h.filter((_, i) => i === 0 || i % 2 === 1 || i === h.length - 1);
    if (epochRef.current >= pauseAtRef.current) return "budget";
    return null;
  }, []);

  const blocked = status === "diverged";
  const extend = () => { if (status === "budget") pauseAtRef.current = epochRef.current + BUDGET; };

  const train = () => {
    if (blocked) return;
    stopLoop(); extend(); setNotice(null);
    setTried((t) => ({ ...t, [activation]: t[activation] + 1 }));
    setStatus("running");
    const tick = () => {
      const start = performance.now();
      let out = null, done = 0;
      const want = cfg.current.speed;
      while (done < want && !out && performance.now() - start < FRAME_MS) { out = doEpoch(); done++; }
      setVersion((v) => v + 1);
      if (out) { rafRef.current = 0; setStatus(out); return; }
      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
  };
  const pause = () => { stopLoop(); setStatus("paused"); };
  const single = () => {
    if (blocked) return;
    stopLoop(); extend(); setNotice(null);
    const out = doEpoch(); setVersion((v) => v + 1); setStatus(out || "paused");
  };

  const onMapMove = (e) => {
    const r = e.currentTarget.getBoundingClientRect();
    const v = mapView(r.width, r.height);
    const x = v.ix(e.clientX - r.left), y = v.iy(e.clientY - r.top);
    setProbe(Math.abs(x) <= 1 && Math.abs(y) <= 1 ? { x, y } : null);
  };

  // derived
  const history = historyRef.current;
  const last = history[history.length - 1];
  const params = net.layers.reduce((s, l) => s + l.weights.length * l.weights[0].length + l.biases.length, 0);
  const fwd = probe ? forwardNetwork(net, probe.x, probe.y) : null;
  const s = STATUS[status];
  const lrPos = (Math.log10(lr) - LR_LO) / (LR_HI - LR_LO);
  const setRate = (v) => setLr(Math.min(3, Math.max(0.003, Number(v.toPrecision(3)))));
  const acc = last?.acc ?? NaN;
  const circleGoal = kind === "circles" && acc >= 0.9 && epochRef.current > 0;
  const [best, setBest] = useState({ tanh: 0, relu: 0 });
  useEffect(() => {
    if (kind === "circles" && epochRef.current > 0 && Number.isFinite(acc) && acc > best[activation]) setBest((b) => ({ ...b, [activation]: acc }));
  }, [acc, kind, activation, best]);
  const bothHit = best.tanh >= 0.9 && best.relu >= 0.9;

  return (
    <div className="space-y-8">
      <header className="rise">
        <p className="num text-xs uppercase tracking-[0.2em] text-accent font-semibold">Lesson 04</p>
        <h1 className="text-4xl md:text-5xl font-extrabold tracking-tight mt-2">Neural Networks</h1>
        <p className="mt-4 max-w-2xl text-lg leading-relaxed text-muted-foreground">
          Logistic regression could only draw one straight line. Stack a few layers of tiny formulas, each a weighted sum followed by a bend,
          and the network can learn rings, checkerboards and spirals. Everything below is trained from random weights, right here, by backpropagation.
        </p>
      </header>

      {notice && <div className="rounded-xl bg-sidebar text-sidebar-foreground px-4 py-3 text-sm pop" data-testid="text-reset-notice">{notice}</div>}

      <section className="grid lg:grid-cols-[1fr_330px] gap-6 rise" style={{ animationDelay: "80ms" }}>
        <div className="rounded-2xl border bg-card p-3 md:p-4 shadow-[0_1px_0_hsl(38_22%_84%),0_12px_32px_-18px_hsl(222_38%_17%/.35)]">
          <div className="flex flex-wrap items-center gap-x-4 gap-y-2 px-1 pb-3 text-xs font-medium">
            <span className="flex items-center gap-1.5"><span className="w-3 h-3 rounded-full" style={{ background: C.teal }} />A, label 0 (circle)</span>
            <span className="flex items-center gap-1.5"><span className="w-3 h-3" style={{ background: C.coral }} />B, label 1 (square)</span>
            <span className="flex items-center gap-1.5"><span className="w-6 border-t-2" style={{ borderColor: C.ink }} />boundary, P(B) = 0.5</span>
            <span className="flex items-center gap-1.5"><span className="num opacity-60">0</span><span className="w-16 h-3 rounded-sm" style={{ background: "linear-gradient(90deg,#93c1bb,#fbf8f0,#f4b2a1)" }} /><span className="num opacity-60">1</span>shading = P(B)</span>
          </div>
          <div className="relative aspect-square w-full max-h-[600px]">
            <DecisionMap network={net} points={points} version={version} probe={probe} onPointerMove={onMapMove} onPointerLeave={() => setProbe(null)} />
          </div>
          <p className="num text-xs text-muted-foreground px-1 pt-2" data-testid="text-probe">
            {fwd ? `probe (${probe.x.toFixed(2)}, ${probe.y.toFixed(2)}) → score ${fmt(fwd.score, 3)} → P(B) ${fmt(fwd.probability, 3)}` : "Hover the map to probe a point: the diagram below lights up with its activations."}
          </p>
          <div className="flex flex-wrap items-center gap-2 px-1 pt-3">
            {status === "running"
              ? <Btn primary onClick={pause} testId="button-pause"><Pause size={14} />Pause</Btn>
              : <Btn primary onClick={train} disabled={blocked} testId="button-train"><Play size={14} />{status === "idle" ? "Train" : "Resume"}</Btn>}
            <Btn onClick={single} disabled={blocked || status === "running"} testId="button-step"><StepForward size={14} />One epoch</Btn>
            <Btn onClick={resetWeights} testId="button-reset-weights"><RotateCcw size={14} />Reset weights</Btn>
            <span className="w-px h-6 bg-border mx-1" />
            <div className="flex items-center gap-1 rounded-full bg-secondary p-1 text-xs font-semibold" role="radiogroup" aria-label="Dataset">
              {KINDS.map(([k, l]) => (
                <button key={k} type="button" role="radio" aria-checked={kind === k} onClick={() => pickKind(k)} data-testid={`button-dataset-${k}`}
                  className={`rounded-full px-3 py-1 transition-colors ${kind === k ? "bg-sidebar text-white" : "hover:bg-background"}`}>{l}</button>
              ))}
            </div>
            <Btn onClick={regen} testId="button-regenerate"><Shuffle size={14} />Regenerate</Btn>
          </div>
        </div>

        <aside className="space-y-4">
          <div className="rounded-2xl bg-sidebar text-sidebar-foreground p-5">
            <p className="text-xs uppercase tracking-[0.18em] opacity-70">Binary log-loss</p>
            <p className="num text-4xl font-semibold mt-2 text-white break-all" data-testid="text-loss">{fmt(last?.loss)}</p>
            <div className="num text-xs grid grid-cols-2 gap-y-1 mt-4 opacity-90">
              <span className="opacity-60">epoch</span><span data-testid="text-epoch">{epochRef.current}</span>
              <span className="opacity-60">training acc.</span><span data-testid="text-accuracy">{pct(acc)}</span>
              <span className="opacity-60">parameters</span><span data-testid="text-params">{params}</span>
              <span className="opacity-60">shape</span><span>{net.sizes.join("→")}</span>
            </div>
          </div>
          <div className="rounded-xl bg-secondary/60 p-4">
            <p className={`font-bold ${s.c}`} data-testid="status-training">{s.t}</p>
            <p className="text-xs text-muted-foreground leading-relaxed mt-1">{s.d}</p>
          </div>
          <div className="rounded-2xl border bg-card p-4 space-y-4">
            <div>
              <div className="flex items-baseline justify-between mb-2">
                <span className="text-sm font-semibold">Hidden layers</span>
                <span className="flex gap-1">
                  <IconBtn onClick={() => setLayers(hidden.slice(0, -1))} disabled={hidden.length <= 1} label="Remove layer" testId="button-remove-layer"><Minus size={12} /></IconBtn>
                  <IconBtn onClick={() => setLayers([...hidden, 4])} disabled={hidden.length >= 3} label="Add layer" testId="button-add-layer"><Plus size={12} /></IconBtn>
                </span>
              </div>
              {hidden.map((n, i) => (
                <div key={i} className="flex items-center gap-2 py-1">
                  <span className="num text-xs w-16 opacity-70">layer {i + 1}</span>
                  <IconBtn onClick={() => setLayers(hidden.map((v, j) => (j === i ? v - 1 : v)))} disabled={n <= 2} label={`Fewer neurons in layer ${i + 1}`} testId={`button-neurons-minus-${i}`}><Minus size={12} /></IconBtn>
                  <span className="num text-sm font-semibold w-6 text-center" data-testid={`text-neurons-${i}`}>{n}</span>
                  <IconBtn onClick={() => setLayers(hidden.map((v, j) => (j === i ? v + 1 : v)))} disabled={n >= 12} label={`More neurons in layer ${i + 1}`} testId={`button-neurons-plus-${i}`}><Plus size={12} /></IconBtn>
                  <span className="text-[11px] text-muted-foreground">neurons</span>
                </div>
              ))}
            </div>
            <div>
              <span className="text-sm font-semibold">Activation</span>
              <div className="flex gap-1.5 mt-2">
                {[["tanh", "tanh"], ["relu", "ReLU"]].map(([a, l]) => (
                  <button key={a} type="button" onClick={() => pickAct(a)} data-testid={`button-activation-${a}`}
                    className={`num flex-1 rounded-lg border px-2 py-1.5 text-xs transition-colors ${activation === a ? "bg-primary text-primary-foreground border-primary" : "hover:bg-secondary"}`}>{l}</button>
                ))}
              </div>
            </div>
            <label className="block">
              <div className="flex items-baseline justify-between mb-2"><span className="text-sm font-semibold">Learning rate α</span><span className="num text-xs font-semibold" data-testid="text-rate">{lr}</span></div>
              <input type="range" className="rng" min={0} max={1} step={0.001} value={lrPos} data-testid="slider-learning-rate"
                onChange={(e) => setRate(10 ** (LR_LO + Number(e.target.value) * (LR_HI - LR_LO)))} />
              <div className="flex gap-1.5 mt-2">
                {[0.03, 0.1, 0.3, 1].map((v) => (
                  <button key={v} type="button" onClick={() => setRate(v)} data-testid={`button-rate-${v}`}
                    className={`num flex-1 rounded-lg border px-1 py-1 text-[11px] transition-colors ${Math.abs(lr - v) < 1e-9 ? "bg-primary text-primary-foreground border-primary" : "hover:bg-secondary"}`}>{v}</button>
                ))}
              </div>
            </label>
            <label className="block">
              <div className="flex items-baseline justify-between mb-2"><span className="text-sm font-semibold">Speed</span><span className="num text-xs">{speed} epochs/frame</span></div>
              <input type="range" className="rng" min={1} max={100} step={1} value={speed} onChange={(e) => setSpeed(Number(e.target.value))} data-testid="slider-speed" />
            </label>
            <p className="text-[11px] text-muted-foreground leading-relaxed">Each frame trains up to that many epochs but stops after ~{FRAME_MS} ms so the page stays responsive. Changing data, layers or activation starts a new network. The learning rate and speed apply immediately without resetting.</p>
          </div>
        </aside>
      </section>

      <section className="grid md:grid-cols-[1.4fr_1fr] gap-6 rise" style={{ animationDelay: "140ms" }}>
        <div className="rounded-2xl border bg-card p-4 md:p-5">
          <div className="flex items-baseline justify-between px-1">
            <h2 className="text-lg font-bold">Inside the network</h2>
            <span className="num text-xs text-muted-foreground">{fwd ? "showing probe activations" : "no probe"}</span>
          </div>
          <div className="mt-3"><NetworkDiagram network={net} activations={fwd?.activations} version={version} /></div>
          <div className="flex flex-wrap gap-x-4 gap-y-1.5 text-[11px] text-muted-foreground px-1 mt-2">
            <span className="flex items-center gap-1.5"><span className="w-6 h-1 rounded" style={{ background: C.coral }} />positive weight</span>
            <span className="flex items-center gap-1.5"><span className="w-6 h-1 rounded" style={{ background: C.teal }} />negative weight</span>
            <span>Fixed scale: width = 0.5 + 2.2·|w| px, capped at 9 px.</span>
            <span>Node ring thickness grows with |bias|; dashed ring = negative bias (hover a node for its value).</span>
            <span>Node fill (when probing): coral = positive activation, teal = negative, paper = near 0.</span>
          </div>
        </div>
        <div className="rounded-2xl border bg-card p-4 md:p-5">
          <div className="flex items-baseline justify-between px-1">
            <h2 className="text-lg font-bold">Loss by epoch</h2>
            <span className="num text-xs text-muted-foreground">{epochRef.current} epochs</span>
          </div>
          <div className="h-64 mt-3"><EpochLossChart history={history} version={version} /></div>
          <p className="text-xs text-muted-foreground leading-relaxed mt-3 px-1">
            <span style={{ color: C.teal }} className="font-semibold">Teal</span>: mean log-loss after each actual epoch (x-axis is epochs, not frames).
            {" "}<span style={{ color: C.coral }} className="font-semibold">Coral</span>: training accuracy, using the right-hand percentage axis. Long runs retain a sample of earlier epochs; their actual spacing is preserved.
          </p>
        </div>
      </section>

      <section className="grid md:grid-cols-2 gap-6 rise" style={{ animationDelay: "200ms" }}>
        <div className="rounded-2xl border bg-card p-6">
          <h2 className="text-xl font-bold">The key equations</h2>
          <div className="mt-4 rounded-xl bg-secondary/70 px-4 py-5 num text-center text-[13px] leading-loose">
            z<sup>(l)</sup> = W<sup>(l)</sup>·a<sup>(l−1)</sup> + b<sup>(l)</sup> &nbsp; a<sup>(0)</sup> = (x1, x2)<br />
            a<sup>(l)</sup> = g(z<sup>(l)</sup>), g = tanh or ReLU = max(0, z)<br />
            p = σ(z<sup>(out)</sup>) &nbsp; L = −(1/n) Σ [y ln p + (1−y) ln(1−p)]<br />
            δ<sup>(out)</sup> = p − y &nbsp; δ<sup>(l)</sup> = (W<sup>(l+1)</sup>)<sup>T</sup>δ<sup>(l+1)</sup> ⊙ g′(z<sup>(l)</sup>)<br />
            ∂L/∂W<sup>(l)</sup> = mean δ<sup>(l)</sup>·(a<sup>(l−1)</sup>)<sup>T</sup> &nbsp; W ← W − α·∂L/∂W
          </div>
          <ul className="mt-4 space-y-2 text-sm leading-relaxed text-muted-foreground">
            <li><b className="text-foreground">Forward:</b> every neuron takes a weighted sum of the layer before, adds its bias, and bends it with the activation.</li>
            <li><b className="text-foreground">Backward:</b> the chain rule passes each point's error p − y back through the same edges, scaled by how steep each activation was.</li>
            <li><b className="text-foreground">Update:</b> one plain gradient-descent step on all weights at once, using the whole dataset. No momentum, no Adam, no tricks.</li>
            <li>Without the bend g, stacking layers collapses into one linear formula: still a single straight line.</li>
          </ul>
        </div>

        <div className="rounded-2xl border bg-card p-6">
          <h2 className="text-xl font-bold">Why logistic regression can't do circles</h2>
          <p className="text-sm leading-relaxed text-muted-foreground mt-3">
            On raw inputs x1, x2, logistic regression's boundary is w<sub>1</sub>x1 + w<sub>2</sub>x2 + b = 0: one straight line. A line splits the plane into two half-planes,
            and no half-plane can contain the whole inner ring while excluding the whole outer ring. A straight line cannot classify both rings perfectly.
          </p>
          <p className="text-sm leading-relaxed text-muted-foreground mt-3">
            Each hidden neuron bends a weighted sum with tanh or ReLU. Later layers combine those nonlinear features, so several units
            can enclose a region from different sides. Together they can learn the nonlinear boundary you see on the map.
          </p>
          <p className="text-sm leading-relaxed text-muted-foreground mt-3">
            The distinction is not absolute: if you hand logistic regression an engineered feature like r² = x1² + x2², the circles become separable by a single threshold.
            A network's value is that it can learn useful features like that by itself.
          </p>
        </div>
      </section>

      <section className={`rounded-2xl border p-6 transition-colors rise ${bothHit ? "bg-primary text-primary-foreground border-primary" : "bg-card"}`} data-testid="card-challenge">
        <p className={`num text-xs uppercase tracking-[0.18em] font-semibold ${bothHit ? "opacity-80" : "text-accent"}`}>Try this</p>
        <h2 className="text-xl font-bold mt-1">Ring the inner circle, twice</h2>
        <p className={`text-sm mt-2 leading-relaxed max-w-2xl ${bothHit ? "opacity-90" : "text-muted-foreground"}`}>
          On Circles, reach 90% training accuracy with tanh, then switch to ReLU and do it again. Compare the boundary shapes: tanh tends to round corners, ReLU builds them from straight pieces.
          This is accuracy on the training points, not on unseen data.
        </p>
        <ul className="mt-4 space-y-1.5 text-sm">
          <Check2 ok={best.tanh >= 0.9} inv={bothHit}>tanh on Circles ≥ 90% <span className="num text-xs opacity-80">(best {pct(best.tanh)}{tried.tanh ? "" : ", not trained yet"})</span></Check2>
          <Check2 ok={best.relu >= 0.9} inv={bothHit}>ReLU on Circles ≥ 90% <span className="num text-xs opacity-80">(best {pct(best.relu)}{tried.relu ? "" : ", not trained yet"})</span></Check2>
        </ul>
        {circleGoal && !bothHit && <p className="pop mt-3 text-sm font-semibold text-primary">Nice: {activation === "tanh" ? "tanh" : "ReLU"} cleared 90%. Now try the other activation.</p>}
        {bothHit && <p className="pop mt-4 font-semibold" data-testid="text-challenge-done">Done. Next, try Spirals: how few neurons can still wind around both arms?</p>}
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

function IconBtn({ children, onClick, disabled, label, testId }) {
  return (
    <button type="button" onClick={onClick} disabled={disabled} aria-label={label} data-testid={testId}
      className="w-6 h-6 grid place-items-center rounded-md bg-secondary hover:bg-[hsl(40_38%_84%)] disabled:opacity-30 disabled:pointer-events-none">{children}</button>
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
