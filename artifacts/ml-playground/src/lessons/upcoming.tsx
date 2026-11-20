import { Link } from "wouter";
import { ArrowRight } from "lucide-react";

export type Upcoming = {
  slug: string; num: string; title: string; blurb: string; explain: string[];
  equation: string[]; terms: [string, string][]; challenge: string; visual: () => import("react").ReactElement;
};

function GDVisual() {
  const pts = Array.from({ length: 7 }, (_, i) => { const t = -1.6 * Math.pow(0.62, i); return { x: 160 + t * 90, y: 30 + t * t * 55 }; });
  return (
    <svg viewBox="0 0 320 200" className="w-full">
      <path d={`M 10 ${30 + 2.56 * 55 * 1.3} Q 160 -60 310 ${30 + 2.56 * 55 * 1.3}`} fill="none" stroke="#1f2a44" strokeOpacity=".15" />
      <path d={Array.from({ length: 61 }, (_, i) => { const t = -1.7 + i * (3.4 / 60); return `${i ? "L" : "M"} ${160 + t * 90} ${200 - (30 + t * t * 55)}`; }).join(" ")} fill="none" stroke="#1c7a72" strokeWidth="3" />
      {pts.map((p, i) => <circle key={i} cx={p.x} cy={200 - p.y} r={6 - i * 0.5} fill="#ec6a4c" opacity={1 - i * 0.1} />)}
      <text x="160" y="195" textAnchor="middle" fontSize="11" fill="#8a8170" fontFamily="JetBrains Mono">minimum</text>
    </svg>
  );
}
function LogVisual() {
  const d = Array.from({ length: 61 }, (_, i) => { const x = i / 60; const y = 1 / (1 + Math.exp(-(x - 0.5) * 12)); return `${i ? "L" : "M"} ${20 + x * 280} ${180 - y * 160}`; }).join(" ");
  const zeros = [0.08, 0.17, 0.26, 0.34, 0.45], ones = [0.55, 0.63, 0.72, 0.84, 0.93];
  return (
    <svg viewBox="0 0 320 200" className="w-full">
      <line x1="20" x2="300" y1="100" y2="100" stroke="#e0a419" strokeDasharray="6 5" strokeWidth="2" />
      <path d={d} fill="none" stroke="#1c7a72" strokeWidth="3" />
      {zeros.map((x) => <circle key={x} cx={20 + x * 280} cy={180} r="6" fill="#1f2a44" />)}
      {ones.map((x) => <circle key={x} cx={20 + x * 280} cy={20} r="6" fill="#ec6a4c" />)}
      <text x="304" y="96" fontSize="10" fill="#8a8170" textAnchor="end" fontFamily="JetBrains Mono">p = 0.5</text>
    </svg>
  );
}
function NNVisual() {
  const layers = [3, 4, 4, 1];
  const pos = layers.map((n, l) => Array.from({ length: n }, (_, i) => ({ x: 40 + l * 80, y: 100 + (i - (n - 1) / 2) * 42 })));
  return (
    <svg viewBox="0 0 320 200" className="w-full">
      {pos.slice(0, -1).map((layer, l) => layer.map((a, i) => pos[l + 1].map((c, j) => (
        <line key={`${l}${i}${j}`} x1={a.x} y1={a.y} x2={c.x} y2={c.y} stroke={(i + j) % 3 ? "#1c7a72" : "#ec6a4c"} strokeOpacity={0.25 + ((i * 7 + j * 3) % 5) / 8} strokeWidth={1 + ((i + j) % 3)} />
      ))))}
      {pos.flat().map((p, k) => <circle key={k} cx={p.x} cy={p.y} r="11" fill="#fbf8f0" stroke="#1f2a44" strokeWidth="2.5" />)}
    </svg>
  );
}

export const UPCOMING: Upcoming[] = [
  {
    slug: "gradient-descent", num: "02", title: "Gradient Descent",
    blurb: "Walk downhill, one careful step at a time.",
    explain: [
      "Imagine standing on a foggy hillside, wanting the lowest valley. You cannot see it, but you can feel which way the ground slopes under your feet. Step that way. Repeat.",
      "That slope is the gradient. The size of each step is the learning rate. Too small and you crawl; too big and you leap clean over the valley and land higher than you started.",
    ],
    equation: ["θ ← θ − α · ∇L(θ)"],
    terms: [["θ", "every adjustable number in the model"], ["α", "learning rate, the step size"], ["∇L", "the direction loss increases fastest"]],
    challenge: "Find the largest learning rate that still converges, then add 10% and watch it fall apart.",
    visual: GDVisual,
  },
  {
    slug: "logistic-regression", num: "03", title: "Logistic Regression",
    blurb: "Bend the line into a yes-or-no answer.",
    explain: [
      "Some questions are not 'how much' but 'which one': spam or not, pass or fail. A straight line can shoot past 1 or below 0, which makes no sense as a probability.",
      "Logistic regression squeezes the line through an S-shaped curve called the sigmoid, so every output lands between 0 and 1. Above 0.5, guess yes.",
    ],
    equation: ["p = σ(w·x + b)", "σ(z) = 1 / (1 + e^(−z))"],
    terms: [["p", "predicted probability of class 1"], ["σ", "the sigmoid squashing function"], ["0.5", "the usual decision threshold"]],
    challenge: "Drag one point to the wrong side and see how far the boundary moves to accommodate it.",
    visual: LogVisual,
  },
  {
    slug: "neural-networks", num: "04", title: "Neural Networks",
    blurb: "Stack many small regressions into something curious.",
    explain: [
      "A single neuron is just the regression you already know, followed by a gentle bend. Wire a few side by side and their bends combine into curves no single line could draw.",
      "Stack layers of them and train every weight at once with gradient descent. That is the core of nearly every modern AI model, scaled up enormously.",
    ],
    equation: ["h = σ(W₁·x + b₁)", "ŷ = W₂·h + b₂"],
    terms: [["W, b", "weights and biases in each layer"], ["h", "the hidden layer's activations"], ["σ", "nonlinearity that lets it bend"]],
    challenge: "Separate a ring of points from its center using as few hidden neurons as possible.",
    visual: NNVisual,
  },
];

export function UpcomingLesson({ lesson, next }: { lesson: Upcoming; next?: Upcoming }) {
  const V = lesson.visual;
  return (
    <div className="space-y-8">
      <header className="rise">
        <div className="flex items-center gap-3">
          <p className="num text-xs uppercase tracking-[0.2em] text-accent font-semibold">Lesson {lesson.num}</p>
          <span className="num text-[10px] uppercase tracking-widest rounded-full border border-dashed border-foreground/40 px-2 py-0.5" data-testid="badge-upcoming">Upcoming · preview</span>
        </div>
        <h1 className="text-4xl md:text-5xl font-extrabold tracking-tight mt-2">{lesson.title}</h1>
        <p className="mt-3 text-lg text-muted-foreground">{lesson.blurb}</p>
      </header>
      <div className="grid md:grid-cols-[1.2fr_1fr] gap-6 rise" style={{ animationDelay: "80ms" }}>
        <div className="rounded-2xl border bg-card p-6 space-y-4">
          <h2 className="text-xl font-bold">In plain English</h2>
          {lesson.explain.map((p) => <p key={p} className="leading-relaxed text-muted-foreground">{p}</p>)}
        </div>
        <div className="rounded-2xl border bg-card p-6">
          <p className="text-xs uppercase tracking-[0.18em] text-muted-foreground mb-3">Sketch of the visual</p>
          <V />
          <p className="text-xs text-muted-foreground mt-3">Static preview. The interactive version is being built.</p>
        </div>
      </div>
      <div className="grid md:grid-cols-2 gap-6 rise" style={{ animationDelay: "160ms" }}>
        <div className="rounded-2xl border bg-card p-6">
          <h2 className="text-xl font-bold">Key equation</h2>
          <div className="mt-4 rounded-xl bg-secondary/70 px-4 py-5 num text-center text-lg leading-loose">
            {lesson.equation.map((e) => <div key={e}>{e}</div>)}
          </div>
          <dl className="mt-4 space-y-1.5 text-sm">
            {lesson.terms.map(([k, v]) => <div key={k} className="flex gap-3"><dt className="num font-semibold w-12 shrink-0">{k}</dt><dd className="text-muted-foreground">{v}</dd></div>)}
          </dl>
        </div>
        <div className="rounded-2xl border-2 border-dashed p-6 flex flex-col">
          <p className="num text-xs uppercase tracking-[0.18em] font-semibold text-accent">Try this · coming soon</p>
          <p className="mt-2 text-lg font-semibold leading-snug">{lesson.challenge}</p>
          <div className="mt-auto pt-6 flex flex-wrap gap-2">
            <Link href="/" className="inline-flex items-center gap-1.5 rounded-full bg-primary text-primary-foreground px-4 py-2 text-sm font-semibold" data-testid="link-back-lesson1">Practice in Lesson 1 <ArrowRight size={14} /></Link>
            {next && <Link href={`/lesson/${next.slug}`} className="inline-flex items-center gap-1.5 rounded-full bg-secondary px-4 py-2 text-sm font-semibold" data-testid="link-next-preview">Preview {next.title}</Link>}
          </div>
        </div>
      </div>
    </div>
  );
}
