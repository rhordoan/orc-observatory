"use client";

import { Card } from "@/components/ui/card";
import type { OTGData, LONData } from "@/lib/types";

interface MetricsBarProps {
  otg: OTGData;
  lon: LONData | null;
}

export function MetricsBar({ otg, lon }: MetricsBarProps) {
  return (
    <div className="h-14 border-b border-border flex items-center gap-3 px-4 overflow-x-auto shrink-0">
      <Metric
        label="Funnels"
        value={otg.funnels.length}
        accent={otg.has_cycles}
      />
      <Sep />
      <Metric
        label="Compression"
        value={`${(otg.compression_ratio * 100).toFixed(1)}%`}
      />
      <Sep />
      <Metric
        label="Mean rank"
        value={otg.mean_terminal_rank.toFixed(3)}
      />
      <Sep />
      <Metric
        label="Top-5% reach"
        value={`${(otg.top5_reachability * 100).toFixed(0)}%`}
      />
      <Sep />
      <Metric label="DAG depth" value={otg.dag_depth} />
      <Sep />
      <Metric
        label="Cycles"
        value={otg.has_cycles ? "yes" : "no"}
        accent={otg.has_cycles}
      />
      {lon && (
        <>
          <Sep />
          <Metric
            label="LON self-loops"
            value={`${(lon.singleton_fraction * 100).toFixed(0)}%`}
          />
        </>
      )}
    </div>
  );
}

function Metric({
  label,
  value,
  accent = false,
}: {
  label: string;
  value: string | number;
  accent?: boolean;
}) {
  return (
    <div className="flex items-center gap-2 shrink-0">
      <span className="text-[10px] uppercase tracking-[0.06em] text-muted-foreground whitespace-nowrap">
        {label}
      </span>
      <span
        className={`text-sm font-mono tabular-nums whitespace-nowrap ${
          accent ? "text-primary" : "text-foreground"
        }`}
      >
        {value}
      </span>
    </div>
  );
}

function Sep() {
  return <div className="w-px h-4 bg-border shrink-0" />;
}
