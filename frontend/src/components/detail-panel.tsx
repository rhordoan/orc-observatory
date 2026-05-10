"use client";

import { useEffect, useState, useRef, useCallback } from "react";
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
          {instance.problem_type === "tsp" && instance.city_coords && (
            <TourMiniMap
              coords={instance.city_coords}
              tourLabel={optimum.label}
            />
          )}
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

        {edge && (
          <section className="space-y-2">
            <SectionTitle>Neighbor ORC Values</SectionTitle>
            <div className="space-y-1.5">
              {Object.entries(otg.orc_values[nodeIdx] || {})
                .sort(([, a], [, b]) => a - b)
                .map(([nbr, kappa]) => (
                  <div key={nbr} className="flex items-center gap-2 text-[10px]">
                    <span className="w-8 text-right font-mono tabular-nums text-muted-foreground">
                      {kappa.toFixed(2)}
                    </span>
                    <div className="flex-1 h-3 bg-secondary rounded-sm overflow-hidden relative">
                      <div
                        className={`absolute top-0 bottom-0 right-0 ${
                          kappa < 0 ? "bg-primary" : "bg-blue-500/50"
                        }`}
                        style={{
                          width: `${Math.min(100, Math.max(0, Math.abs(kappa) * 100))}%`,
                        }}
                      />
                    </div>
                    {parseInt(nbr) === edge.via_neighbor && (
                      <span className="w-3 text-primary">←</span>
                    )}
                  </div>
                ))}
            </div>
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

            <div className="text-[11px] font-mono space-y-0.5 overflow-x-auto">
              <p className="text-muted-foreground font-sans text-[11px] mb-1">
                Optimal matching
              </p>
              {explain.matching.map(([i, j], idx) => (
                <div
                  key={idx}
                  className="flex items-center gap-1 min-w-0"
                >
                  <span className="shrink-0 text-right">{explain.x_exclusive_labels[i]}</span>
                  <div className="flex items-center min-w-[40px]">
                    <div className="h-px bg-border flex-1" />
                    <span className="text-[11px] text-muted-foreground px-0.5 bg-card whitespace-nowrap">
                      {explain.pair_costs[idx].toFixed(2)}
                    </span>
                    <div className="h-px bg-border flex-1" />
                  </div>
                  <span className="shrink-0">{explain.y_exclusive_labels[j]}</span>
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
  const isLong = typeof value === "string" && value.length > 12;
  return (
    <div className="flex justify-between text-xs gap-2">
      <span className="text-muted-foreground shrink-0">{label}</span>
      <span
        className={`text-right truncate ${mono ? "font-mono tabular-nums" : ""} ${
          highlight ? "text-primary font-medium" : ""
        } ${isLong ? "text-[10px]" : ""}`}
      >
        {value}
      </span>
    </div>
  );
}

function TourMiniMap({
  coords,
  tourLabel,
}: {
  coords: number[][];
  tourLabel: string;
}) {
  const ref = useRef<HTMLCanvasElement>(null);

  const draw = useCallback(() => {
    const cvs = ref.current;
    if (!cvs) return;
    const ctx = cvs.getContext("2d");
    if (!ctx) return;

    const S = cvs.width;
    const pad = 14;
    const inner = S - 2 * pad;
    ctx.clearRect(0, 0, S, S);

    const tour = tourLabel.split("\u2192").map(Number);
    if (tour.some(isNaN) || tour.length < 3) return;

    const fg = getComputedStyle(cvs).getPropertyValue("--foreground").trim();
    const primary = getComputedStyle(cvs).getPropertyValue("--primary").trim();
    const fgColor = fg.startsWith("hsl") || fg.startsWith("#") ? fg : `hsl(${fg})`;
    const priColor = primary.startsWith("hsl") || primary.startsWith("#") ? primary : `hsl(${primary})`;

    const px = (x: number) => pad + x * inner;
    const py = (y: number) => pad + (1 - y) * inner;

    ctx.strokeStyle = priColor;
    ctx.lineWidth = 1.5;
    ctx.globalAlpha = 0.6;
    ctx.beginPath();
    for (let i = 0; i <= tour.length; i++) {
      const c = coords[tour[i % tour.length]];
      if (i === 0) ctx.moveTo(px(c[0]), py(c[1]));
      else ctx.lineTo(px(c[0]), py(c[1]));
    }
    ctx.stroke();
    ctx.globalAlpha = 1;

    for (let ci = 0; ci < coords.length; ci++) {
      const [cx, cy] = coords[ci];
      const inTour = tour.includes(ci);
      ctx.fillStyle = inTour ? priColor : fgColor;
      ctx.globalAlpha = inTour ? 1 : 0.25;
      ctx.beginPath();
      ctx.arc(px(cx), py(cy), inTour ? 4 : 2.5, 0, Math.PI * 2);
      ctx.fill();

      if (inTour) {
        ctx.fillStyle = fgColor;
        ctx.globalAlpha = 0.7;
        ctx.font = "bold 8px system-ui";
        ctx.textAlign = "center";
        ctx.fillText(String(ci), px(cx), py(cy) - 6);
      }
    }
    ctx.globalAlpha = 1;
  }, [coords, tourLabel]);

  useEffect(() => { draw(); }, [draw]);

  return (
    <canvas
      ref={ref}
      width={160}
      height={160}
      className="w-full rounded-md border border-border bg-background"
      style={{ imageRendering: "auto" }}
    />
  );
}
