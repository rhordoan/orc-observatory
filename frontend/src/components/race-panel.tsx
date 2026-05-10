"use client";

import { useMemo } from "react";
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
import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";
import type { ILSIterationEvent, ILSResult } from "@/lib/types";

const ALGO_META: Record<string, { label: string; color: string }> = {
  orc: { label: "ORC+Pert", color: "oklch(0.795 0.148 71.1)" },
  random: { label: "Random-ILS", color: "oklch(0.6 0.15 250)" },
  rrhc: { label: "RR-HC", color: "oklch(0.58 0.01 75)" },
};

interface RacePanelProps {
  hasCycles: boolean;
  events: ILSIterationEvent[];
  results: ILSResult[] | null;
  winner: string | null;
  isRacing: boolean;
}

export function RacePanel({
  hasCycles,
  events,
  results,
  winner,
  isRacing,
}: RacePanelProps) {
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
    const rows: Record<string, number | undefined>[] = [];

    for (let i = 0; i < maxLen; i++) {
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

    return rows;
  }, [events]);

  const predicted = hasCycles ? "orc" : "random";
  const predictionCorrect = winner ? winner === predicted : null;

  return (
    <div className="flex flex-col h-full border-t border-border bg-card">
      <div className="flex items-center gap-3 px-4 py-2.5 border-b border-border shrink-0">
        <span className="text-xs font-semibold uppercase tracking-[0.06em] text-muted-foreground">
          Algorithm Race
        </span>

        <Badge
          variant="outline"
          className={
            hasCycles
              ? "border-primary/40 text-primary"
              : "border-border text-muted-foreground"
          }
        >
          {hasCycles ? "Cyclic OTG" : "Acyclic OTG"}
        </Badge>

        <span className="text-[11px] text-muted-foreground">
          {hasCycles
            ? "ORC-guided perturbation recommended"
            : "Random perturbation recommended"}
        </span>

        {isRacing && (
          <span className="ml-auto flex items-center gap-1.5 text-[11px] text-primary">
            <span className="w-1.5 h-1.5 rounded-full bg-primary animate-pulse" />
            Racing
          </span>
        )}

        {winner && (
          <span className="ml-auto text-[11px] font-medium">
            Winner:{" "}
            <span className="text-primary">
              {ALGO_META[winner]?.label ?? winner}
            </span>
            {predictionCorrect !== null && (
              <span
                className={
                  predictionCorrect
                    ? " text-green-500 ml-1.5"
                    : " text-destructive ml-1.5"
                }
              >
                {predictionCorrect
                  ? "Prediction confirmed"
                  : "Prediction refuted"}
              </span>
            )}
          </span>
        )}
      </div>

      <div className="flex-1 min-h-0 p-4">
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
              formatter={(value: number, name: string) => [
                value.toFixed(4),
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
            {Object.entries(ALGO_META).map(([key, { color }]) => (
              <Line
                key={key}
                type="stepAfter"
                dataKey={key}
                stroke={color}
                strokeWidth={winner === key ? 3 : 1.5}
                dot={false}
                isAnimationActive={false}
                connectNulls
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>

      {results && results.length > 0 && (
        <div className="flex items-center gap-4 px-4 py-2 border-t border-border shrink-0">
          {results.map((r) => (
            <div
              key={r.algo}
              className={`flex items-center gap-2 text-xs ${
                r.algo === winner ? "text-primary font-medium" : "text-muted-foreground"
              }`}
            >
              <span
                className="w-2.5 h-2.5 rounded-full shrink-0"
                style={{
                  backgroundColor: ALGO_META[r.algo]?.color ?? "#888",
                }}
              />
              <span>{ALGO_META[r.algo]?.label ?? r.algo}</span>
              <span className="font-mono tabular-nums">
                {r.best_fitness.toFixed(4)}
              </span>
              <span className="text-muted-foreground/60">
                ({r.total_evals} FE)
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
