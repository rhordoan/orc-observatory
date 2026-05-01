"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Slider } from "@/components/ui/slider";
import { createInstance, buildOTG, buildLON } from "@/lib/api";
import type { InstanceData, OTGData, LONData } from "@/lib/types";

interface SidebarProps {
  instance: InstanceData | null;
  onInstanceCreated: (data: InstanceData) => void;
  onOtgBuilt: (data: OTGData) => void;
  onLonBuilt: (data: LONData) => void;
  isLoading: boolean;
  setIsLoading: (v: boolean) => void;
}

export function Sidebar({
  instance,
  onInstanceCreated,
  onOtgBuilt,
  onLonBuilt,
  isLoading,
  setIsLoading,
}: SidebarProps) {
  const [problemType, setProblemType] = useState("nk");
  const [n, setN] = useState(10);
  const [k, setK] = useState(2);
  const [seed, setSeed] = useState<string>("42");

  async function handleGenerate() {
    setIsLoading(true);
    try {
      const inst = await createInstance({
        problem_type: problemType,
        n,
        k,
        seed: seed ? parseInt(seed) : null,
      });
      onInstanceCreated(inst);

      const [otgResult, lonResult] = await Promise.all([
        buildOTG(inst.instance_id),
        buildLON(inst.instance_id),
      ]);
      onOtgBuilt(otgResult);
      onLonBuilt(lonResult);
    } catch (err) {
      console.error("Generation failed:", err);
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <aside className="w-[260px] shrink-0 border-r border-border bg-sidebar text-sidebar-foreground flex flex-col">
      <div className="h-14 flex items-center px-4 border-b border-border">
        <span className="text-sm font-semibold tracking-tight">
          ORC Observatory
        </span>
      </div>

      <div className="flex-1 overflow-y-auto p-4 space-y-5">
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

        <Button
          onClick={handleGenerate}
          disabled={isLoading}
          className="w-full"
        >
          {isLoading ? (
            <span className="flex items-center gap-2">
              <LoadingDots />
              Generating
            </span>
          ) : (
            "Generate & Analyze"
          )}
        </Button>

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
