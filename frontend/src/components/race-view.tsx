"use client";

import { useState, useMemo, memo } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";
import { Slider } from "@/components/ui/slider";
import type {
  OTGData,
  MetricsData,
  ILSIterationEvent,
  ILSResult,
  InstanceData,
} from "@/lib/types";

const ALGO_META: Record<
  string,
  { label: string; color: string; desc: string }
> = {
  orc: {
    label: "ORC + Pert",
    color: "oklch(0.795 0.148 71.1)",
    desc: "1 ORC-directed move + d_r random moves",
  },
  random: {
    label: "Random-ILS",
    color: "oklch(0.6 0.15 250)",
    desc: "d_r + 1 random perturbation moves",
  },
  rrhc: {
    label: "RR-HC",
    color: "oklch(0.65 0.17 160)",
    desc: "Random restart from a new solution",
  },
};

const ALGO_ORDER = ["orc", "random", "rrhc"] as const;

interface RaceViewProps {
  instance: InstanceData;
  otg: OTGData;
  metrics: MetricsData | null;
  events: ILSIterationEvent[];
  results: ILSResult[] | null;
  winner: string | null;
  isRacing: boolean;
  onStartRace: (
    budget: number,
    d_r: number,
    seed: number | null,
    paceMs: number
  ) => void;
  onCancelRace: () => void;
  onExportCSV: () => void;
  seed: string;
}

export function RaceView({
  instance,
  otg,
  metrics,
  events,
  results,
  winner,
  isRacing,
  onStartRace,
  onCancelRace,
  onExportCSV,
  seed,
}: RaceViewProps) {
  const [budget, setBudget] = useState(5000);
  const [dR, setDR] = useState(2);
  const [paceMs, setPaceMs] = useState(50);

  const predictionCorrect = winner
    ? otg.has_cycles
      ? winner === "orc"
      : winner !== "orc"
    : null;
  const hasRun = results !== null;

  const latestByAlgo = useMemo(() => {
    const latest: Record<string, ILSIterationEvent> = {};
    for (const e of events) {
      latest[e.algo] = e;
    }
    return latest;
  }, [events]);

  const chartData = useMemo(() => {
    const byAlgo: Record<string, { evals: number; best_fitness: number }[]> = {
      orc: [],
      random: [],
      rrhc: [],
    };

    for (const e of events) {
      byAlgo[e.algo]?.push({ evals: e.evals, best_fitness: e.best_fitness });
    }

    const maxLen = Math.max(
      byAlgo.orc.length,
      byAlgo.random.length,
      byAlgo.rrhc.length
    );

    const MAX_CHART_POINTS = 500;
    const stride = maxLen > MAX_CHART_POINTS ? Math.ceil(maxLen / MAX_CHART_POINTS) : 1;
    const rows: Record<string, number | undefined>[] = [];

    for (let i = 0; i < maxLen; i += stride) {
      const row: Record<string, number | undefined> = {};
      const orc = byAlgo.orc[i];
      const rand = byAlgo.random[i];
      const rrhc = byAlgo.rrhc[i];
      row.evals = orc?.evals ?? rand?.evals ?? rrhc?.evals;
      if (orc) row.orc = orc.best_fitness;
      if (rand) row.random = rand.best_fitness;
      if (rrhc) row.rrhc = rrhc.best_fitness;
      rows.push(row);
    }

    if (stride > 1 && maxLen > 0) {
      const last = maxLen - 1;
      const row: Record<string, number | undefined> = {};
      const orc = byAlgo.orc[last];
      const rand = byAlgo.random[last];
      const rrhc = byAlgo.rrhc[last];
      row.evals = orc?.evals ?? rand?.evals ?? rrhc?.evals;
      if (orc) row.orc = orc.best_fitness;
      if (rand) row.random = rand.best_fitness;
      if (rrhc) row.rrhc = rrhc.best_fitness;
      rows.push(row);
    }

    return rows;
  }, [events]);

  return (
    <div className="flex flex-col h-full">
      {/* Top bar: diagnostic + controls */}
      <div className="shrink-0 border-b border-border px-6 py-4 flex items-start justify-between gap-6">
        <div className="flex flex-col gap-2">
          <div className="flex items-center gap-3">
            <h2 className="text-sm font-semibold">Algorithm Race</h2>
            <Badge
              variant="outline"
              className={
                otg.has_cycles
                  ? "border-primary/40 text-primary"
                  : "border-border text-muted-foreground"
              }
            >
              {otg.has_cycles ? "Cyclic OTG" : "Acyclic OTG"}
            </Badge>
          </div>
          <p className="text-xs text-muted-foreground max-w-md">
            {otg.has_cycles
              ? "Cyclic structure detected in the OTG. ORC-guided perturbation is recommended for this landscape."
              : "Acyclic OTG detected. Random perturbation should perform comparably or better on this landscape."}
          </p>
          {winner && (
            <div className="flex items-center gap-2 text-xs mt-0.5">
              <span className="text-muted-foreground">Winner:</span>
              <span
                className="font-semibold"
                style={{ color: ALGO_META[winner]?.color }}
              >
                {ALGO_META[winner]?.label}
              </span>
              {predictionCorrect !== null && (
                <Badge
                  variant="outline"
                  className={
                    predictionCorrect
                      ? "border-green-500/40 text-green-500"
                      : "border-destructive/40 text-destructive"
                  }
                >
                  {predictionCorrect
                    ? "Prediction correct"
                    : "Prediction refuted"}
                </Badge>
              )}
            </div>
          )}
        </div>

        {/* Compact controls */}
        <div className="flex items-end gap-4 shrink-0">
          <div className="flex flex-col gap-1 w-32">
            <span className="text-[10px] uppercase tracking-wider text-muted-foreground">
              Budget{" "}
              <span className="text-foreground font-mono">{budget}</span>
            </span>
            <Slider
              value={[budget]}
              onValueChange={(v) => setBudget(Array.isArray(v) ? v[0] : v)}
              min={1000}
              max={20000}
              step={1000}
            />
          </div>
          <div className="flex flex-col gap-1 w-20">
            <span className="text-[10px] uppercase tracking-wider text-muted-foreground">
              d_r <span className="text-foreground font-mono">{dR}</span>
            </span>
            <Slider
              value={[dR]}
              onValueChange={(v) => setDR(Array.isArray(v) ? v[0] : v)}
              min={1}
              max={5}
              step={1}
            />
          </div>
          <div className="flex flex-col gap-1 w-20">
            <span className="text-[10px] uppercase tracking-wider text-muted-foreground">
              Pace{" "}
              <span className="text-foreground font-mono">{paceMs}ms</span>
            </span>
            <Slider
              value={[paceMs]}
              onValueChange={(v) => setPaceMs(Array.isArray(v) ? v[0] : v)}
              min={0}
              max={200}
              step={10}
            />
          </div>

          <div className="flex gap-2">
            {isRacing ? (
              <Button variant="secondary" onClick={onCancelRace}>
                Cancel
              </Button>
            ) : (
              <Button
                onClick={() => {
                  const parsed = parseInt(seed, 10);
                  onStartRace(
                    budget,
                    dR,
                    Number.isFinite(parsed) ? parsed : null,
                    paceMs
                  );
                }}
              >
                {hasRun ? "Re-run" : "Run Race"}
              </Button>
            )}
            {hasRun && (
              <Button variant="secondary" onClick={onExportCSV}>
                CSV
              </Button>
            )}
          </div>
        </div>
      </div>

      {/* Main content */}
      <div className="flex-1 min-h-0 flex">
        {/* Algorithm cards + chart */}
        <div className="flex-1 flex flex-col min-w-0 p-6 gap-4">
          {/* Three algorithm cards */}
          <div className="grid grid-cols-3 gap-3 shrink-0">
            {ALGO_ORDER.map((key) => {
              const meta = ALGO_META[key];
              const latest = latestByAlgo[key];
              const result = results?.find((r) => r.algo === key);
              const isWinner = winner === key;
              const isPredicted = otg.has_cycles ? key === "orc" : key !== "orc";

              return (
                <Card
                  key={key}
                  className={`p-4 relative overflow-hidden transition-all ${
                    isWinner
                      ? "ring-2 ring-primary/60 bg-primary/5"
                      : "bg-card"
                  }`}
                >
                  {isWinner && (
                    <div className="absolute top-2 right-2">
                      <Badge className="bg-primary text-primary-foreground text-[9px]">
                        WINNER
                      </Badge>
                    </div>
                  )}
                  <div className="flex items-center gap-2 mb-3">
                    <div
                      className="w-3 h-3 rounded-full shrink-0"
                      style={{ backgroundColor: meta.color }}
                    />
                    <span className="text-sm font-semibold">{meta.label}</span>
                    {isPredicted && !hasRun && (
                      <Badge
                        variant="outline"
                        className="text-[9px] border-primary/30 text-primary ml-auto"
                      >
                        PREDICTED
                      </Badge>
                    )}
                  </div>
                  <p className="text-[11px] text-muted-foreground mb-3">
                    {meta.desc}
                  </p>

                  <div className="space-y-1.5">
                    <div className="flex justify-between text-xs">
                      <span className="text-muted-foreground">
                        Best fitness
                      </span>
                      <span className="font-mono tabular-nums font-medium">
                        {result
                          ? result.best_fitness.toFixed(4)
                          : latest
                            ? latest.best_fitness.toFixed(4)
                            : "\u2014"}
                      </span>
                    </div>
                    <div className="flex justify-between text-xs">
                      <span className="text-muted-foreground">Evaluations</span>
                      <span className="font-mono tabular-nums">
                        {result
                          ? result.total_evals.toLocaleString()
                          : latest
                            ? latest.evals.toLocaleString()
                            : "\u2014"}
                      </span>
                    </div>
                  </div>

                  {isRacing && latest && (
                    <div className="mt-2 h-0.5 bg-border rounded-full overflow-hidden">
                      <div
                        className="h-full rounded-full transition-all duration-300"
                        style={{
                          backgroundColor: meta.color,
                          width: `${Math.min((latest.evals / budget) * 100, 100)}%`,
                        }}
                      />
                    </div>
                  )}
                </Card>
              );
            })}
          </div>

          {/* Convergence chart */}
          <div className="flex-1 min-h-0">
            {events.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={chartData}>
                  <CartesianGrid
                    strokeDasharray="3 3"
                    stroke="oklch(0.5 0 0 / 15%)"
                  />
                  <XAxis
                    dataKey="evals"
                    label={{
                      value: "Fitness Evaluations",
                      position: "insideBottom",
                      offset: -5,
                      style: { fontSize: 11, fill: "oklch(0.58 0.01 75)" },
                    }}
                    tick={{ fontSize: 10 }}
                    stroke="oklch(0.5 0 0 / 30%)"
                  />
                  <YAxis
                    label={{
                      value: "Best Fitness",
                      angle: -90,
                      position: "insideLeft",
                      style: { fontSize: 11, fill: "oklch(0.58 0.01 75)" },
                    }}
                    tick={{ fontSize: 10 }}
                    stroke="oklch(0.5 0 0 / 30%)"
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "oklch(0.17 0.008 280)",
                      border: "1px solid oklch(1 0 0 / 8%)",
                      borderRadius: 6,
                      fontSize: 11,
                    }}
                    labelStyle={{ color: "oklch(0.58 0.01 75)" }}
                    formatter={(value: unknown, name: string) => [
                      typeof value === "number" ? value.toFixed(4) : String(value ?? ""),
                      ALGO_META[name]?.label ?? name,
                    ]}
                    labelFormatter={(v) => `${v} evals`}
                  />
                  <Legend
                    formatter={(value: string) =>
                      ALGO_META[value]?.label ?? value
                    }
                    wrapperStyle={{ fontSize: 11 }}
                  />
                  {ALGO_ORDER.map((key) => (
                    <Line
                      key={key}
                      type="stepAfter"
                      dataKey={key}
                      stroke={ALGO_META[key].color}
                      strokeWidth={winner === key ? 3 : 1.5}
                      dot={false}
                      isAnimationActive={false}
                      connectNulls
                    />
                  ))}
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <div className="h-full flex items-center justify-center">
                <div className="text-center space-y-2">
                  <div className="text-muted-foreground/40 text-4xl">
                    {"\u25B6"}
                  </div>
                  <p className="text-sm text-muted-foreground">
                    Press <span className="font-medium text-foreground">Run Race</span> to start comparing algorithms
                  </p>
                  <p className="text-xs text-muted-foreground/60">
                    All three ILS variants will run simultaneously on this
                    landscape instance
                  </p>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Right sidebar: difficulty metrics */}
        {metrics && (
          <div className="w-56 shrink-0 border-l border-border p-4 space-y-4">
            <div>
              <h3 className="text-[10px] uppercase tracking-wider text-muted-foreground font-semibold mb-3">
                Difficulty Metrics
              </h3>
              <div className="space-y-2.5">
                <MetricRow label="FDC" value={metrics.fdc.toFixed(3)} />
                <MetricRow
                  label="Autocorrelation"
                  value={metrics.autocorrelation_length.toFixed(1)}
                />
                <MetricRow
                  label="Info content"
                  value={metrics.information_content.toFixed(3)}
                />
                <MetricRow
                  label="Mean ORC"
                  value={metrics.mean_orc.toFixed(3)}
                  accent={metrics.mean_orc < -0.3}
                />
              </div>
            </div>

            <div className="border-t border-border pt-4">
              <h3 className="text-[10px] uppercase tracking-wider text-muted-foreground font-semibold mb-3">
                Instance
              </h3>
              <div className="space-y-2.5">
                <MetricRow label="Optima" value={instance.n_optima} />
                <MetricRow
                  label="Space"
                  value={`2^${Math.log2(instance.space_size)}`}
                />
                <MetricRow label="Degree" value={instance.degree} />
                <MetricRow
                  label="Funnels"
                  value={otg.funnels.length}
                />
                <MetricRow
                  label="DAG depth"
                  value={otg.dag_depth}
                />
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

const MetricRow = memo(function MetricRow({
  label,
  value,
  accent = false,
}: {
  label: string;
  value: string | number;
  accent?: boolean;
}) {
  return (
    <div className="flex justify-between text-xs">
      <span className="text-muted-foreground">{label}</span>
      <span
        className={`font-mono tabular-nums ${accent ? "text-primary font-medium" : ""}`}
      >
        {value}
      </span>
    </div>
  );
});
