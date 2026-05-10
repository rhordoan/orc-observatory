"use client";

import { useState, useCallback, useRef } from "react";
import { Sidebar } from "@/components/sidebar";
import { GraphCanvas } from "@/components/graph-canvas";
import { DetailPanel } from "@/components/detail-panel";
import { MetricsBar } from "@/components/metrics-bar";
import { RacePanel } from "@/components/race-panel";
import { streamILS } from "@/lib/api";
import type {
  InstanceData,
  OTGData,
  LONData,
  MetricsData,
  ILSIterationEvent,
  ILSResult,
} from "@/lib/types";

export default function Home() {
  const [instance, setInstance] = useState<InstanceData | null>(null);
  const [otg, setOtg] = useState<OTGData | null>(null);
  const [lon, setLon] = useState<LONData | null>(null);
  const [selectedNode, setSelectedNode] = useState<number | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  /* F2: race state */
  const [metrics, setMetrics] = useState<MetricsData | null>(null);
  const [isRacing, setIsRacing] = useState(false);
  const [raceEvents, setRaceEvents] = useState<ILSIterationEvent[]>([]);
  const [raceResults, setRaceResults] = useState<ILSResult[] | null>(null);
  const [raceWinner, setRaceWinner] = useState<string | null>(null);
  const [trajectories, setTrajectories] = useState<Record<string, number[]> | null>(null);
  const cancelRef = useRef<(() => void) | null>(null);

  const handleInstanceCreated = useCallback((inst: InstanceData) => {
    setSelectedNode(null);
    setOtg(null);
    setLon(null);
    setMetrics(null);
    setRaceEvents([]);
    setRaceResults(null);
    setRaceWinner(null);
    setTrajectories(null);
    setInstance(inst);
  }, []);

  const handleStartRace = useCallback(
    (budget: number, d_r: number, seed: number | null, paceMs: number) => {
      if (!instance) return;

      setIsRacing(true);
      setRaceEvents([]);
      setRaceResults(null);
      setRaceWinner(null);
      setTrajectories(null);

      const cancel = streamILS(
        instance.instance_id,
        budget,
        d_r,
        seed,
        paceMs,
        (event) => {
          setRaceEvents((prev) => [...prev, event]);
        },
        (winner, results) => {
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
    [instance]
  );

  const handleCancelRace = useCallback(() => {
    cancelRef.current?.();
    cancelRef.current = null;
    setIsRacing(false);
  }, []);

  const showRacePanel = isRacing || raceResults !== null;

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
        metrics={metrics}
        onMetrics={setMetrics}
        isRacing={isRacing}
        raceResults={raceResults}
        raceEvents={raceEvents}
        onStartRace={handleStartRace}
        onCancelRace={handleCancelRace}
      />

      <div className="flex-1 flex flex-col min-w-0">
        {otg && <MetricsBar otg={otg} lon={lon} />}

        <div className={`flex min-h-0 ${showRacePanel ? "h-1/2" : "flex-1"}`}>
          <GraphCanvas
            instance={instance}
            otg={otg}
            lon={lon}
            selectedNode={selectedNode}
            onNodeSelect={setSelectedNode}
            trajectories={trajectories}
          />

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

        {showRacePanel && otg && (
          <div className="h-1/2 min-h-0">
            <RacePanel
              hasCycles={otg.has_cycles}
              events={raceEvents}
              results={raceResults}
              winner={raceWinner}
              isRacing={isRacing}
            />
          </div>
        )}
      </div>
    </div>
  );
}
