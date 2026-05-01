"use client";

import { useEffect, useState } from "react";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { explainORC } from "@/lib/api";
import type { InstanceData, OTGData, ORCExplainData } from "@/lib/types";

interface DetailPanelProps {
  instance: InstanceData;
  otg: OTGData;
  nodeIdx: number;
  onClose: () => void;
}

export function DetailPanel({
  instance,
  otg,
  nodeIdx,
  onClose,
}: DetailPanelProps) {
  const [explain, setExplain] = useState<ORCExplainData | null>(null);
  const [loading, setLoading] = useState(false);

  const optimum = instance.optima[nodeIdx];
  const edge = otg.edges.find((e) => e.source === nodeIdx);
  const funnel = otg.funnels.find((f) =>
    f.member_indices.includes(nodeIdx)
  );

  useEffect(() => {
    if (!edge || edge.source === edge.target) return;
    setLoading(true);
    explainORC(
      instance.instance_id,
      optimum.solution_idx,
      edge.via_neighbor
    )
      .then(setExplain)
      .catch(console.error)
      .finally(() => setLoading(false));
  }, [instance.instance_id, optimum.solution_idx, edge]);

  return (
    <aside className="w-[300px] shrink-0 border-l border-border bg-card overflow-y-auto">
      <div className="h-14 flex items-center justify-between px-4 border-b border-border">
        <span className="text-sm font-semibold">Node detail</span>
        <button
          onClick={onClose}
          className="text-muted-foreground hover:text-foreground transition-colors text-lg leading-none"
        >
          x
        </button>
      </div>

      <div className="p-4 space-y-4">
        <section className="space-y-1.5">
          <SectionTitle>Local optimum</SectionTitle>
          <Row label="Solution" value={optimum.label} mono />
          <Row label="Fitness" value={optimum.fitness.toFixed(4)} mono />
          <Row label="Basin size" value={optimum.basin_size} mono />
          <Row label="Rank" value={`#${nodeIdx + 1} / ${instance.n_optima}`} />
        </section>

        <Separator />

        {funnel && (
          <section className="space-y-1.5">
            <SectionTitle>Funnel membership</SectionTitle>
            <Row
              label="Attractor"
              value={instance.optima[funnel.attractor_idx].label}
              mono
            />
            <Row
              label="Attractor fitness"
              value={funnel.attractor_fitness.toFixed(4)}
              mono
            />
            <Row label="Funnel size" value={funnel.member_indices.length} />
            {funnel.is_cycle && (
              <Badge variant="outline" className="text-[10px] mt-1">
                cyclic
              </Badge>
            )}
          </section>
        )}

        <Separator />

        {edge && (
          <section className="space-y-1.5">
            <SectionTitle>OTG escape edge</SectionTitle>
            <Row
              label="Kappa (min ORC)"
              value={edge.min_kappa.toFixed(4)}
              mono
              highlight
            />
            <Row
              label="Destination"
              value={instance.optima[edge.target]?.label ?? "self"}
              mono
            />
          </section>
        )}

        <Separator />

        {loading && (
          <p className="text-xs text-muted-foreground">
            Loading transport plan...
          </p>
        )}

        {explain && (
          <section className="space-y-2">
            <SectionTitle>Transport decomposition</SectionTitle>
            <div className="text-xs space-y-1">
              <p className="text-muted-foreground">
                Shared ({explain.shared.length}): self-match at cost 0
              </p>
              <p className="text-muted-foreground">
                Exclusive ({explain.x_exclusive.length} x{" "}
                {explain.y_exclusive.length}): matched by Hungarian algorithm
              </p>
            </div>

            <div className="text-xs font-mono space-y-0.5">
              <p className="text-muted-foreground font-sans text-[11px] mb-1">
                Optimal matching
              </p>
              {explain.matching.map(([i, j], idx) => (
                <div
                  key={idx}
                  className="flex items-center justify-between gap-2"
                >
                  <span>{explain.x_exclusive_labels[i]}</span>
                  <span className="text-muted-foreground text-[10px]">
                    {explain.pair_costs[idx].toFixed(2)}
                  </span>
                  <span>{explain.y_exclusive_labels[j]}</span>
                </div>
              ))}
            </div>

            <Row
              label="W1 distance"
              value={explain.w1.toFixed(4)}
              mono
            />
            <Row
              label="kappa = 1 - W1"
              value={explain.kappa.toFixed(4)}
              mono
              highlight
            />
          </section>
        )}
      </div>
    </aside>
  );
}

function SectionTitle({ children }: { children: React.ReactNode }) {
  return (
    <h3 className="text-[11px] font-medium uppercase tracking-[0.06em] text-muted-foreground">
      {children}
    </h3>
  );
}

function Row({
  label,
  value,
  mono = false,
  highlight = false,
}: {
  label: string;
  value: string | number;
  mono?: boolean;
  highlight?: boolean;
}) {
  return (
    <div className="flex justify-between text-xs">
      <span className="text-muted-foreground">{label}</span>
      <span
        className={`${mono ? "font-mono tabular-nums" : ""} ${
          highlight ? "text-primary font-medium" : ""
        }`}
      >
        {value}
      </span>
    </div>
  );
}
