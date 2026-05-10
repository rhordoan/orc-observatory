"use client";

import { useState, useEffect, useCallback } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Slider } from "@/components/ui/slider";
import { createInstance, buildOTG, buildLON, streamOTG, fetchMetrics } from "@/lib/api";
import type {
  InstanceData,
  OTGData,
  LONData,
  MetricsData,
  ILSIterationEvent,
  ILSResult,
} from "@/lib/types";

interface SidebarProps {
  instance: InstanceData | null;
  otg: OTGData | null;
  onInstanceCreated: (data: InstanceData) => void;
  onOtgBuilt: (data: OTGData) => void;
  onLonBuilt: (data: LONData) => void;
  isLoading: boolean;
  setIsLoading: (v: boolean) => void;
  metrics: MetricsData | null;
  onMetrics: (data: MetricsData) => void;
  isRacing: boolean;
  raceResults: ILSResult[] | null;
  raceEvents: ILSIterationEvent[];
  onStartRace: (budget: number, d_r: number, seed: number | null, paceMs: number) => void;
  onCancelRace: () => void;
}

export function Sidebar({
  instance,
  otg,
  onInstanceCreated,
  onOtgBuilt,
  onLonBuilt,
  isLoading,
  setIsLoading,
  metrics,
  onMetrics,
  isRacing,
  raceResults,
  raceEvents,
  onStartRace,
  onCancelRace,
}: SidebarProps) {
  const [problemType, setProblemType] = useState("nk");
  const [n, setN] = useState(10);
  const [k, setK] = useState(2);
  const [mu, setMu] = useState(2);
  const [nu, setNu] = useState(2);
  const [gamma, setGamma] = useState(2);
  const [seed, setSeed] = useState<string>("42");
  const [hasGenerated, setHasGenerated] = useState(false);

  /* Race controls */
  const [budget, setBudget] = useState(5000);
  const [dR, setDR] = useState(2);
  const [paceMs, setPaceMs] = useState(50);

  const handleGenerate = useCallback(async () => {
    setIsLoading(true);
    setHasGenerated(true);
    try {
      const inst = await createInstance({
        problem_type: problemType,
        n,
        k,
        mu,
        nu,
        gamma_wmodel: gamma,
        seed: seed ? parseInt(seed) : null,
      });
      onInstanceCreated(inst);

      const [otgResult, lonResult] = await Promise.all([
        buildOTG(inst.instance_id),
        buildLON(inst.instance_id),
      ]);
      onOtgBuilt(otgResult);
      onLonBuilt(lonResult);

      fetchMetrics(inst.instance_id)
        .then(onMetrics)
        .catch(() => {});
    } catch (err) {
      console.error("Generation failed:", err);
    } finally {
      setIsLoading(false);
    }
  }, [problemType, n, k, mu, nu, gamma, seed, onInstanceCreated, onOtgBuilt, onLonBuilt, setIsLoading, onMetrics]);

  useEffect(() => {
    if (!hasGenerated) return;
    const timer = setTimeout(() => {
      handleGenerate();
    }, 500);
    return () => clearTimeout(timer);
  }, [k, mu, nu, gamma, n, problemType, hasGenerated]);

  function handleAnimate() {
    if (!instance) return;
    setIsLoading(true);

    const partialOtg: OTGData = {
      instance_id: instance.instance_id,
      edges: [],
      funnels: [],
      orc_values: {},
      compression_ratio: 1,
      mean_terminal_rank: 0,
      top5_reachability: 0,
      dag_depth: 0,
      has_cycles: false,
    };
    onOtgBuilt({ ...partialOtg });

    streamOTG(
      instance.instance_id,
      1.0,
      (msg) => {
        if (msg.type === "edge_added") {
          partialOtg.edges.push({
            source: msg.source as number,
            target: msg.target as number,
            min_kappa: msg.min_kappa as number,
            via_neighbor: msg.via_neighbor as number,
          });
          onOtgBuilt({ ...partialOtg, edges: [...partialOtg.edges] });
        } else if (msg.type === "funnel_formed") {
          partialOtg.funnels.push({
            attractor_idx: msg.attractor as number,
            member_indices: msg.members as number[],
            attractor_fitness: msg.attractor_fitness as number,
            is_cycle: msg.is_cycle as boolean,
          });
          onOtgBuilt({ ...partialOtg, funnels: [...partialOtg.funnels] });
        } else if (msg.type === "complete") {
          partialOtg.compression_ratio = msg.compression as number;
          partialOtg.mean_terminal_rank = msg.mean_terminal_rank as number;
          partialOtg.top5_reachability = msg.top5_reachability as number;
          partialOtg.dag_depth = msg.dag_depth as number;
          partialOtg.has_cycles = msg.has_cycles as boolean;
          onOtgBuilt({ ...partialOtg });
        }
      },
      () => {
        buildOTG(instance.instance_id).then((fullOtg) => {
          onOtgBuilt(fullOtg);
          setIsLoading(false);
        });
      }
    );
  }

  function handleExportCSV() {
    if (!raceEvents.length) return;

    const header = "algo,evals,best_fitness\n";
    const rows = raceEvents
      .map((e) => `${e.algo},${e.evals},${e.best_fitness}`)
      .join("\n");

    let csv = header + rows;

    if (metrics) {
      csv +=
        "\n\nmetric,value\n" +
        `fdc,${metrics.fdc}\n` +
        `autocorrelation_length,${metrics.autocorrelation_length}\n` +
        `information_content,${metrics.information_content}\n` +
        `mean_orc,${metrics.mean_orc}`;
    }

    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `orc_race_${instance?.instance_id ?? "export"}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  }

  return (
    <aside className="w-[260px] shrink-0 border-r border-border bg-sidebar text-sidebar-foreground flex flex-col">
      <div className="h-14 flex items-center px-4 border-b border-border">
        <span className="text-sm font-semibold tracking-tight">
          ORC Observatory
        </span>
      </div>

      <div className="flex-1 overflow-y-auto p-4 space-y-5">
        {/* ---- Problem configuration ---- */}
        <section>
          <Label>Problem type</Label>
          <div className="flex gap-1.5 mt-1.5">
            {(["nk", "wmodel", "maxsat"] as const).map((t) => (
              <button
                key={t}
                onClick={() => setProblemType(t)}
                className={`
                  px-2.5 py-1 text-xs rounded-md border transition-colors duration-150
                  ${
                    problemType === t
                      ? "border-primary bg-primary/10 text-primary font-medium"
                      : "border-border text-muted-foreground hover:border-primary/40 hover:text-foreground"
                  }
                `}
              >
                {t === "nk" ? "NK" : t === "wmodel" ? "W-model" : "MAX-SAT"}
              </button>
            ))}
          </div>
        </section>

        <section>
          <Label>
            N (problem size) <span className="text-muted-foreground">{n}</span>
          </Label>
          <Slider
            value={[n]}
            onValueChange={(v) => setN(Array.isArray(v) ? v[0] : v)}
            min={4}
            max={14}
            step={1}
            className="mt-2"
          />
        </section>

        {problemType === "nk" && (
          <section>
            <Label>
              K (epistasis){" "}
              <span className="text-muted-foreground">{k}</span>
            </Label>
            <Slider
              value={[k]}
              onValueChange={(v) => setK(Array.isArray(v) ? v[0] : v)}
              min={1}
              max={Math.max(n - 1, 1)}
              step={1}
              className="mt-2"
            />
          </section>
        )}

        {problemType === "wmodel" && (
          <>
            <section>
              <Label>
                mu (neutrality){" "}
                <span className="text-muted-foreground">{mu}</span>
              </Label>
              <Slider
                value={[mu]}
                onValueChange={(v) => setMu(Array.isArray(v) ? v[0] : v)}
                min={1}
                max={Math.max(n - 1, 1)}
                step={1}
                className="mt-2"
              />
            </section>
            <section>
              <Label>
                nu (epistasis){" "}
                <span className="text-muted-foreground">{nu}</span>
              </Label>
              <Slider
                value={[nu]}
                onValueChange={(v) => setNu(Array.isArray(v) ? v[0] : v)}
                min={1}
                max={Math.max(n - 1, 1)}
                step={1}
                className="mt-2"
              />
            </section>
            <section>
              <Label>
                gamma (ruggedness){" "}
                <span className="text-muted-foreground">{gamma}</span>
              </Label>
              <Slider
                value={[gamma]}
                onValueChange={(v) => setGamma(Array.isArray(v) ? v[0] : v)}
                min={0}
                max={100}
                step={1}
                className="mt-2"
              />
            </section>
          </>
        )}

        <section>
          <Label>Seed</Label>
          <input
            type="text"
            value={seed}
            onChange={(e) => setSeed(e.target.value)}
            placeholder="optional"
            className="mt-1.5 w-full bg-input/50 border border-border rounded-md px-2.5 py-1.5 text-sm font-mono text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-primary"
          />
        </section>

        <div className="flex gap-2">
          <Button
            onClick={handleGenerate}
            disabled={isLoading}
            className="flex-1"
          >
            {isLoading ? (
              <span className="flex items-center gap-2">
                <LoadingDots />
                Generating
              </span>
            ) : (
              "Generate"
            )}
          </Button>

          <Button
            variant="secondary"
            onClick={handleAnimate}
            disabled={!instance || isLoading}
            title="Animate OTG construction via WebSocket"
          >
            Animate
          </Button>
        </div>

        {/* ---- Instance info card ---- */}
        {instance && (
          <Card className="p-3 space-y-1.5">
            <div className="flex items-center justify-between">
              <span className="text-xs text-muted-foreground">Instance</span>
              <Badge variant="outline" className="text-[10px] font-mono">
                {instance.instance_id}
              </Badge>
            </div>
            <Row label="Space" value={`2^${Math.log2(instance.space_size)}`} />
            <Row label="Degree" value={instance.degree} />
            <Row label="Local optima" value={instance.n_optima} />
          </Card>
        )}

        {/* ---- Difficulty metrics card (FR6) ---- */}
        {metrics && (
          <Card className="p-3 space-y-1.5">
            <span className="text-xs text-muted-foreground">
              Difficulty Metrics
            </span>
            <Row label="FDC" value={metrics.fdc.toFixed(3)} />
            <Row
              label="Autocorrelation"
              value={metrics.autocorrelation_length.toFixed(1)}
            />
            <Row
              label="Info content"
              value={metrics.information_content.toFixed(3)}
            />
            <div className="flex justify-between text-xs">
              <span className="text-muted-foreground">Mean ORC</span>
              <span
                className={`font-mono tabular-nums ${
                  metrics.mean_orc < -0.3
                    ? "text-primary"
                    : "text-foreground"
                }`}
              >
                {metrics.mean_orc.toFixed(3)}
              </span>
            </div>
          </Card>
        )}

        {/* ---- Algorithm Race section ---- */}
        {otg && (
          <>
            <div className="border-t border-border pt-4">
              <Label>Algorithm Race</Label>
            </div>

            <section>
              <Label>
                Budget (FE){" "}
                <span className="text-muted-foreground">{budget}</span>
              </Label>
              <Slider
                value={[budget]}
                onValueChange={(v) =>
                  setBudget(Array.isArray(v) ? v[0] : v)
                }
                min={1000}
                max={20000}
                step={1000}
                className="mt-2"
              />
            </section>

            <section>
              <Label>
                d_r (random moves){" "}
                <span className="text-muted-foreground">{dR}</span>
              </Label>
              <Slider
                value={[dR]}
                onValueChange={(v) => setDR(Array.isArray(v) ? v[0] : v)}
                min={1}
                max={5}
                step={1}
                className="mt-2"
              />
            </section>

            <section>
              <Label>
                Pace (ms){" "}
                <span className="text-muted-foreground">{paceMs}</span>
              </Label>
              <Slider
                value={[paceMs]}
                onValueChange={(v) =>
                  setPaceMs(Array.isArray(v) ? v[0] : v)
                }
                min={0}
                max={200}
                step={10}
                className="mt-2"
              />
            </section>

            <div className="flex gap-2">
              {isRacing ? (
                <Button
                  variant="secondary"
                  onClick={onCancelRace}
                  className="flex-1"
                >
                  Cancel
                </Button>
              ) : (
                <Button
                  onClick={() =>
                    onStartRace(
                      budget,
                      dR,
                      seed ? parseInt(seed) : null,
                      paceMs
                    )
                  }
                  disabled={isLoading}
                  className="flex-1"
                >
                  Run Race
                </Button>
              )}

              <Button
                variant="secondary"
                onClick={handleExportCSV}
                disabled={raceEvents.length === 0}
                title="Export race results as CSV"
              >
                CSV
              </Button>
            </div>
          </>
        )}
      </div>
    </aside>
  );
}

function Label({ children }: { children: React.ReactNode }) {
  return (
    <label className="text-[11px] font-medium uppercase tracking-[0.06em] text-muted-foreground flex items-center justify-between">
      {children}
    </label>
  );
}

function Row({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="flex justify-between text-xs">
      <span className="text-muted-foreground">{label}</span>
      <span className="font-mono tabular-nums">{value}</span>
    </div>
  );
}

function LoadingDots() {
  return (
    <span className="flex gap-0.5">
      {[0, 1, 2].map((i) => (
        <span
          key={i}
          className="w-1 h-1 rounded-full bg-primary-foreground animate-pulse"
          style={{ animationDelay: `${i * 150}ms` }}
        />
      ))}
    </span>
  );
}
