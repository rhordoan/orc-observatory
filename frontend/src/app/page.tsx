"use client";

import { useState, useCallback } from "react";
import { Sidebar } from "@/components/sidebar";
import { GraphCanvas } from "@/components/graph-canvas";
import { DetailPanel } from "@/components/detail-panel";
import { MetricsBar } from "@/components/metrics-bar";
import type { InstanceData, OTGData, LONData } from "@/lib/types";

export default function Home() {
  const [instance, setInstance] = useState<InstanceData | null>(null);
  const [otg, setOtg] = useState<OTGData | null>(null);
  const [lon, setLon] = useState<LONData | null>(null);
  const [selectedNode, setSelectedNode] = useState<number | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const handleInstanceCreated = useCallback((inst: InstanceData) => {
    setSelectedNode(null);
    setInstance(inst);
  }, []);

  return (
    <div className="flex h-screen overflow-hidden">
      <Sidebar
        instance={instance}
        onInstanceCreated={handleInstanceCreated}
        onOtgBuilt={setOtg}
        onLonBuilt={setLon}
        isLoading={isLoading}
        setIsLoading={setIsLoading}
      />

      <div className="flex-1 flex flex-col min-w-0">
        {otg && <MetricsBar otg={otg} lon={lon} />}

        <div className="flex-1 flex min-h-0">
          <GraphCanvas
            instance={instance}
            otg={otg}
            lon={lon}
            selectedNode={selectedNode}
            onNodeSelect={setSelectedNode}
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
      </div>
    </div>
  );
}
