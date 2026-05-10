"use client";

import { useState, useCallback, useRef, useEffect, lazy, Suspense } from "react";
import { Sidebar } from "@/components/sidebar";
import { MetricsBar } from "@/components/metrics-bar";
import { streamILS, fetchGpuStatus } from "@/lib/api";

const GraphCanvas = lazy(() =>
  import("@/components/graph-canvas").then((m) => ({ default: m.GraphCanvas }))
);
const DetailPanel = lazy(() =>
  import("@/components/detail-panel").then((m) => ({ default: m.DetailPanel }))
);
const RaceView = lazy(() =>
  import("@/components/race-view").then((m) => ({ default: m.RaceView }))
);
const LandscapeView3D = lazy(() =>
  import("@/components/landscape-3d").then((m) => ({
    default: m.LandscapeView3D,
  }))
);
import type {
  InstanceData,
  OTGData,
  LONData,
  MetricsData,
  ILSIterationEvent,
  ILSResult,
} from "@/lib/types";

type Tab = "otg" | "race";

export default function Home() {
  const [instance, setInstance] = useState<InstanceData | null>(null);
  const [otg, setOtg] = useState<OTGData | null>(null);
  const [lon, setLon] = useState<LONData | null>(null);
  const [selectedNode, setSelectedNode] = useState<number | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [seed, setSeed] = useState("42");
  const [activeTab, setActiveTab] = useState<Tab>("otg");
  const [view3D, setView3D] = useState(false);

  /* GPU state */
  const [gpuAvailable, setGpuAvailable] = useState(false);
  const [useGpu, setUseGpu] = useState(false);

  useEffect(() => {
    fetchGpuStatus().then((avail) => {
      setGpuAvailable(avail);
      if (avail) setUseGpu(true);
    });
  }, []);

  /* F2: race state */
  const [metrics, setMetrics] = useState<MetricsData | null>(null);
  const [isRacing, setIsRacing] = useState(false);
  const [raceEvents, setRaceEvents] = useState<ILSIterationEvent[]>([]);
  const [raceResults, setRaceResults] = useState<ILSResult[] | null>(null);
  const [raceWinner, setRaceWinner] = useState<string | null>(null);
  const [trajectories, setTrajectories] = useState<Record<string, number[]> | null>(null);
  const cancelRef = useRef<(() => void) | null>(null);
  const eventBufferRef = useRef<ILSIterationEvent[]>([]);
  const flushTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const FLUSH_INTERVAL_MS = 100;

  const flushEvents = useCallback(() => {
    if (eventBufferRef.current.length === 0) return;
    const batch = eventBufferRef.current;
    eventBufferRef.current = [];
    setRaceEvents((prev) => {
      const merged = new Array(prev.length + batch.length);
      for (let i = 0; i < prev.length; i++) merged[i] = prev[i];
      for (let i = 0; i < batch.length; i++) merged[prev.length + i] = batch[i];
      return merged;
    });
  }, []);

  const startFlushing = useCallback(() => {
    if (flushTimerRef.current) return;
    flushTimerRef.current = setInterval(flushEvents, FLUSH_INTERVAL_MS);
  }, [flushEvents]);

  const stopFlushing = useCallback(() => {
    if (flushTimerRef.current) {
      clearInterval(flushTimerRef.current);
      flushTimerRef.current = null;
    }
    flushEvents();
  }, [flushEvents]);

  useEffect(() => () => {
    if (flushTimerRef.current) clearInterval(flushTimerRef.current);
  }, []);

  const handleInstanceCreated = useCallback((inst: InstanceData) => {
    setSelectedNode(null);
    setOtg(null);
    setLon(null);
    setMetrics(null);
    setRaceEvents([]);
    setRaceResults(null);
    setRaceWinner(null);
    setTrajectories(null);
    eventBufferRef.current = [];
    setInstance(inst);
  }, []);

  const handleStartRace = useCallback(
    (budget: number, d_r: number, raceSeed: number | null, paceMs: number) => {
      if (!instance) return;

      setIsRacing(true);
      setRaceEvents([]);
      setRaceResults(null);
      setRaceWinner(null);
      setTrajectories(null);
      eventBufferRef.current = [];

      startFlushing();

      const cancel = streamILS(
        instance.instance_id,
        budget,
        d_r,
        raceSeed,
        paceMs,
        (event) => {
          eventBufferRef.current.push(event);
        },
        (winner, results) => {
          stopFlushing();
          setRaceResults(results);
          setRaceWinner(winner);
          setIsRacing(false);
          cancelRef.current = null;

          const trajs: Record<string, number[]> = {};
          for (const r of results) {
            trajs[r.algo] = r.trajectory;
          }
          setTrajectories(trajs);
        }
      );

      cancelRef.current = cancel;
    },
    [instance, startFlushing, stopFlushing]
  );

  const handleCancelRace = useCallback(() => {
    cancelRef.current?.();
    cancelRef.current = null;
    stopFlushing();
    setIsRacing(false);
  }, [stopFlushing]);

  const handleExportCSV = useCallback(() => {
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
  }, [raceEvents, metrics, instance]);

  const canRace = instance !== null && otg !== null;

  return (
    <div className="flex h-screen overflow-hidden">
      <Sidebar
        instance={instance}
        otg={otg}
        onInstanceCreated={handleInstanceCreated}
        onOtgBuilt={setOtg}
        onLonBuilt={setLon}
        isLoading={isLoading}
        setIsLoading={setIsLoading}
        onMetrics={setMetrics}
        seed={seed}
        onSeedChange={setSeed}
        gpuAvailable={gpuAvailable}
        useGpu={useGpu}
        onToggleGpu={() => setUseGpu((v) => !v)}
      />

      <div className="flex-1 flex flex-col min-w-0">
        {/* Tab bar */}
        {otg && (
          <div className="h-10 border-b border-border flex items-center px-4 gap-1 shrink-0 bg-card">
            <TabButton
              active={activeTab === "otg"}
              onClick={() => setActiveTab("otg")}
            >
              OTG Explorer
            </TabButton>
            <TabButton
              active={activeTab === "race"}
              onClick={() => setActiveTab("race")}
              disabled={!canRace}
            >
              Algorithm Race
              {isRacing && (
                <span className="w-1.5 h-1.5 rounded-full bg-primary animate-pulse ml-1.5" />
              )}
            </TabButton>
          </div>
        )}

        {/* Tab content */}
        <Suspense fallback={
          <div className="flex-1 flex items-center justify-center">
            <div className="w-6 h-6 border-2 border-primary/30 border-t-primary rounded-full animate-spin" />
          </div>
        }>
          {activeTab === "otg" ? (
            <div className="flex-1 flex flex-col min-h-0">
              {otg && <MetricsBar otg={otg} lon={lon} />}

              <div className="flex flex-1 min-h-0 relative">

                {view3D && instance && otg && instance.space_size <= 1024 ? (
                  <>
                    <LandscapeView3D
                      instance={instance}
                      otg={otg}
                      selectedNode={selectedNode}
                      onNodeSelect={setSelectedNode}
                    />
                    <button
                      onClick={() => setView3D(false)}
                      className="absolute top-3 left-3 z-20 px-3 py-1 text-xs rounded-md border h-8 bg-primary/15 text-primary border-primary/30 font-medium transition-colors hover:bg-primary/25"
                    >
                      ← 2D Graph
                    </button>
                  </>
                ) : (
                  <GraphCanvas
                    instance={instance}
                    otg={otg}
                    lon={lon}
                    selectedNode={selectedNode}
                    onNodeSelect={setSelectedNode}
                    trajectories={trajectories}
                    view3D={view3D}
                    onToggle3D={() => setView3D((v) => !v)}
                    show3DToggle={
                      !!instance && !!otg &&
                      !["tsp", "qap"].includes(instance.problem_type) &&
                      instance.space_size <= 1024
                    }
                  />
                )}

                {selectedNode !== null && instance && otg &&
                  selectedNode < instance.optima.length && (
                  <DetailPanel
                    instance={instance}
                    otg={otg}
                    nodeIdx={selectedNode}
                    onClose={() => setSelectedNode(null)}
                  />
                )}
              </div>
            </div>
          ) : (
            canRace && (
              <RaceView
                instance={instance}
                otg={otg}
                metrics={metrics}
                events={raceEvents}
                results={raceResults}
                winner={raceWinner}
                isRacing={isRacing}
                onStartRace={handleStartRace}
                onCancelRace={handleCancelRace}
                onExportCSV={handleExportCSV}
                seed={seed}
              />
            )
          )}
        </Suspense>
      </div>
    </div>
  );
}

function TabButton({
  active,
  disabled,
  onClick,
  children,
}: {
  active: boolean;
  disabled?: boolean;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={`
        flex items-center gap-1 px-3 py-1.5 text-xs rounded-md transition-colors
        ${
          active
            ? "bg-primary/10 text-primary font-medium"
            : disabled
              ? "text-muted-foreground/40 cursor-not-allowed"
              : "text-muted-foreground hover:text-foreground hover:bg-muted/50"
        }
      `}
    >
      {children}
    </button>
  );
}
