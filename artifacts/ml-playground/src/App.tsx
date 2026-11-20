import { useState, type ReactNode } from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ErrorBoundary } from '@/components/error-boundary';
import { Toaster } from '@/components/ui/toaster';
import { TooltipProvider } from '@/components/ui/tooltip';
import NotFound from '@/pages/not-found';
import { Link, Route, Switch, useLocation, useParams, Router as WouterRouter } from 'wouter';
import { Menu, X } from 'lucide-react';
import LinearRegressionLesson from '@/lessons/LinearRegressionLesson';
import GradientDescentLesson from '@/lessons/GradientDescentLesson';
import LogisticRegressionLesson from '@/lessons/LogisticRegressionLesson';
import NeuralNetworkLesson from '@/lessons/NeuralNetworkLesson';
import { UPCOMING, UpcomingLesson } from '@/lessons/upcoming';

const queryClient = new QueryClient();

const NAV = [
  { href: '/', num: '01', title: 'Linear Regression', live: true },
  { href: '/lesson/gradient-descent', num: '02', title: 'Gradient Descent', live: true },
  { href: '/lesson/logistic-regression', num: '03', title: 'Logistic Regression', live: true },
  { href: '/lesson/neural-networks', num: '04', title: 'Neural Networks', live: true },
  ...UPCOMING.filter((l) => !['gradient-descent', 'logistic-regression', 'neural-networks'].includes(l.slug)).map((l) => ({ href: `/lesson/${l.slug}`, num: l.num, title: l.title, live: false })),
];

function Sidebar({ onNavigate }: { onNavigate?: () => void }) {
  const [loc] = useLocation();
  return (
    <nav className="flex flex-col h-full p-5">
      <Link href="/" onClick={onNavigate} className="flex items-center gap-2.5 mb-10" data-testid="link-home">
        <svg width="30" height="30" viewBox="0 0 30 30" aria-hidden>
          <rect width="30" height="30" rx="9" fill="#ec6a4c" />
          <line x1="6" y1="22" x2="24" y2="8" stroke="#fbf8f0" strokeWidth="2.5" strokeLinecap="round" />
          <circle cx="10" cy="14" r="2.2" fill="#1f2a44" /><circle cx="19" cy="17" r="2.2" fill="#1f2a44" />
        </svg>
        <span className="font-extrabold text-lg tracking-tight text-white">ML Playground</span>
      </Link>
      <p className="num text-[10px] uppercase tracking-[0.2em] opacity-50 mb-3 px-3">Lessons</p>
      <ul className="space-y-1">
        {NAV.map((n) => {
          const active = loc === n.href;
          return (
            <li key={n.href}>
              <Link
                href={n.href} onClick={onNavigate} data-testid={`link-lesson-${n.num}`}
                className={`group flex items-center gap-3 rounded-xl px-3 py-2.5 transition-colors ${active ? 'bg-sidebar-accent text-white' : 'hover:bg-sidebar-accent/60'}`}
              >
                <span className={`num text-xs w-7 h-7 grid place-items-center rounded-lg transition-colors ${active ? 'bg-sidebar-primary text-white' : 'bg-white/5 group-hover:bg-white/10'}`}>{n.num}</span>
                <span className="flex-1 text-sm font-semibold">{n.title}</span>
                {!n.live && <span className="num text-[9px] uppercase tracking-wider opacity-50">soon</span>}
              </Link>
            </li>
          );
        })}
      </ul>
      <div className="mt-auto rounded-xl bg-white/5 p-4 text-xs leading-relaxed opacity-80">
        Every number here is computed live in your browser. Break things freely; nothing is saved.
      </div>
    </nav>
  );
}

function Shell({ children }: { children: ReactNode }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="min-h-[100dvh] md:grid md:grid-cols-[260px_1fr] graph-paper">
      <aside className="hidden md:block sticky top-0 h-[100dvh] bg-sidebar text-sidebar-foreground"><Sidebar /></aside>
      <div className="md:hidden flex items-center justify-between px-4 py-3 bg-sidebar text-white sticky top-0 z-30">
        <span className="font-extrabold">ML Playground</span>
        <button onClick={() => setOpen(true)} aria-label="Open lessons" data-testid="button-open-menu"><Menu /></button>
      </div>
      {open && (
        <div className="md:hidden fixed inset-0 z-40 bg-foreground/40" onClick={() => setOpen(false)}>
          <aside className="w-72 h-full bg-sidebar text-sidebar-foreground relative" onClick={(e) => e.stopPropagation()}>
            <button className="absolute top-5 right-4 text-white" onClick={() => setOpen(false)} aria-label="Close" data-testid="button-close-menu"><X /></button>
            <Sidebar onNavigate={() => setOpen(false)} />
          </aside>
        </div>
      )}
      <main className="px-4 py-8 md:px-10 md:py-12 max-w-6xl w-full">{children}</main>
    </div>
  );
}

function UpcomingPage() {
  const { slug } = useParams<{ slug: string }>();
  const i = UPCOMING.findIndex((l) => l.slug === slug);
  if (i < 0) return <NotFound />;
  return <UpcomingLesson key={slug} lesson={UPCOMING[i]} next={UPCOMING[i + 1]} />;
}

function Router() {
  return (
    <Shell>
      <RoutedErrorBoundary>
        <Switch>
          <Route path="/" component={LinearRegressionLesson} />
          <Route path="/lesson/gradient-descent" component={GradientDescentLesson} />
          <Route path="/lesson/logistic-regression" component={LogisticRegressionLesson} />
          <Route path="/lesson/neural-networks" component={NeuralNetworkLesson} />
          <Route path="/lesson/:slug" component={UpcomingPage} />
          <Route component={NotFound} />
        </Switch>
      </RoutedErrorBoundary>
    </Shell>
  );
}

function RoutedErrorBoundary({ children }: { children: ReactNode }) {
  const [location] = useLocation();
  return <ErrorBoundary resetKey={location}>{children}</ErrorBoundary>;
}

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <TooltipProvider>
        <WouterRouter base={import.meta.env.BASE_URL.replace(/\/$/, '')}>
          <Router />
        </WouterRouter>
        <Toaster />
      </TooltipProvider>
    </QueryClientProvider>
  );
}

export default App;
