"use client";

import { useEffect, useRef, useState } from "react";
import * as d3 from "d3";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import type { InstanceData, OTGData, LONData } from "@/lib/types";

const TRAJECTORY_COLORS: Record<string, string> = {
  orc: "oklch(0.795 0.148 71.1)",
  random: "oklch(0.6 0.15 250)",
  rrhc: "oklch(0.58 0.01 75)",
};

interface GraphCanvasProps {
  instance: InstanceData | null;
  otg: OTGData | null;
  lon: LONData | null;
  selectedNode: number | null;
  onNodeSelect: (idx: number | null) => void;
  trajectories?: Record<string, number[]> | null;
}

type ViewMode = "otg" | "lon" | "side-by-side";

export function GraphCanvas({
  instance,
  otg,
  lon,
  selectedNode,
  onNodeSelect,
  trajectories,
}: GraphCanvasProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [viewMode, setViewMode] = useState<ViewMode>("otg");

  const selectedNodeRef = useRef(selectedNode);
  const onNodeSelectRef = useRef(onNodeSelect);

  useEffect(() => {
    selectedNodeRef.current = selectedNode;
    onNodeSelectRef.current = onNodeSelect;
    
    if (!containerRef.current) return;
    d3.select(containerRef.current).selectAll<SVGCircleElement, any>("circle")
      .attr("stroke", (d) => d.idx === selectedNode ? "var(--primary)" : (d.isAttractor ? "var(--foreground)" : "transparent"))
      .attr("stroke-width", (d) => d.idx === selectedNode ? 3 : (d.isAttractor ? 2 : 0));
  }, [selectedNode, onNodeSelect]);

  useEffect(() => {
    if (!containerRef.current || !instance || !otg) return;

    const container = d3.select(containerRef.current);
    container.selectAll("svg").remove();

    const width = containerRef.current.clientWidth / (viewMode === "side-by-side" ? 2 : 1);
    const height = containerRef.current.clientHeight;

    const optima = instance.optima;
    const funnels = otg.funnels;

    // Build funnel membership map for coloring
    const funnelOf = new Map<number, number>();
    funnels.forEach((f, fi) => {
      f.member_indices.forEach((mi) => funnelOf.set(mi, fi));
    });

    const funnelColors = [
      "oklch(0.795 0.148 71.1)",    // amber
      "oklch(0.65 0.18 160)",       // teal
      "oklch(0.65 0.15 250)",       // blue
      "oklch(0.70 0.16 30)",        // coral
      "oklch(0.62 0.13 310)",       // purple
      "oklch(0.75 0.12 95)",        // lime
      "oklch(0.60 0.18 200)",       // cyan
      "oklch(0.68 0.15 350)",       // pink
    ];

    const minKappa = d3.min(otg.edges, (d) => d.min_kappa) ?? -1;
    const kappaColor = d3.scaleLinear<string>()
      .domain([minKappa, 0])
      .range(["oklch(0.6 0.2 25)", "oklch(0.65 0.15 150)"])
      .clamp(true);

    const maxBasin = d3.max(optima, (o) => o.basin_size) ?? 1;
    const nodeRadius = (basinSize: number) =>
      4 + 14 * Math.sqrt(basinSize / maxBasin);

    type NodeDatum = d3.SimulationNodeDatum & {
      idx: number;
      fitness: number;
      basinSize: number;
      funnelIdx: number;
      isAttractor: boolean;
    };

    const attractorSet = new Set(funnels.map((f) => f.attractor_idx));

    const nodes: NodeDatum[] = optima.map((o) => ({
      idx: o.list_idx,
      fitness: o.fitness,
      basinSize: o.basin_size,
      funnelIdx: funnelOf.get(o.list_idx) ?? 0,
      isAttractor: attractorSet.has(o.list_idx),
    }));

    const nodeByIdx = new Map(nodes.map((n) => [n.idx, n]));

    const activeOtgEdges = otg.edges
      .map((e) => ({
        source: nodeByIdx.get(e.source),
        target: nodeByIdx.get(e.target),
        kappa: e.min_kappa,
      }))
      .filter((e): e is { source: NodeDatum; target: NodeDatum; kappa: number } =>
        e.source !== undefined && e.target !== undefined
      );

    const activeLonEdges = lon ? lon.edges
      .map((e) => ({
        source: nodeByIdx.get(e.source),
        target: nodeByIdx.get(e.target),
        kappa: 0,
      }))
      .filter((e): e is { source: NodeDatum; target: NodeDatum; kappa: number } =>
        e.source !== undefined && e.target !== undefined
      ) : [];

    // The simulation runs ONLY on the OTG edges so that OTG clusters funnels nicely
    const simLinks = activeOtgEdges.filter((e) => e.source !== e.target);

    const isLarge = nodes.length > 200;
    const chargeStrength = isLarge ? -80 : -200;
    const linkDist = isLarge ? 30 : 60;

    const simulation = d3
      .forceSimulation(nodes)
      .force(
        "link",
        d3.forceLink(simLinks).id((d) => (d as NodeDatum).idx).distance(linkDist)
      )
      .force("charge", d3.forceManyBody().strength(chargeStrength).theta(isLarge ? 0.9 : 0.8))
      .force("center", d3.forceCenter(width / 2, height / 2))
      .force("collision", d3.forceCollide().radius((d) =>
        nodeRadius((d as NodeDatum).basinSize) + 2
      ));

    if (isLarge) simulation.alphaDecay(0.05);

    function createGraph(svgNode: any, edges: typeof activeOtgEdges, isOtg: boolean) {
      const svg = d3.select(svgNode)
        .attr("width", width)
        .attr("height", height)
        .style("border-right", isOtg && viewMode === "side-by-side" ? "1px solid var(--border)" : "none");

      const g = svg.append("g").attr("class", "graph-container")
        .style("will-change", "transform");

      const zoom = d3.zoom<SVGSVGElement, unknown>()
        .scaleExtent([0.1, 8])
        .on("zoom", (event) => g.attr("transform", event.transform));
      svg.call(zoom);

      // Arrow markers
      svg.append("defs").append("marker")
        .attr("id", "arrowhead")
        .attr("viewBox", "0 -4 8 8")
        .attr("refX", 8)
        .attr("refY", 0)
        .attr("markerWidth", 6)
        .attr("markerHeight", 6)
        .attr("orient", "auto")
        .append("path")
        .attr("d", "M0,-3L8,0L0,3Z")
        .attr("fill", "var(--muted-foreground)");

      const validEdges = edges.filter((e) => e.source !== e.target);

      const linkSel = g.append("g").attr("class", "edges")
        .selectAll("line")
        .data(validEdges)
        .join("line")
        .attr("stroke", (d) => isOtg ? kappaColor(d.kappa) : "var(--muted-foreground)")
        .attr("stroke-width", 1.5)
        .attr("stroke-opacity", 0.6)
        .attr("marker-end", "url(#arrowhead)");

      const nodeSel = g.append("g").attr("class", "nodes")
        .selectAll<SVGCircleElement, NodeDatum>("circle")
        .data(nodes)
        .join("circle")
        .attr("r", (d) => nodeRadius(d.basinSize))
        .attr("fill", (d) => funnelColors[d.funnelIdx % funnelColors.length])
        .attr("fill-opacity", (d) => (d.isAttractor ? 1 : 0.7))
        .attr("stroke", (d) => d.idx === selectedNodeRef.current ? "var(--primary)" : (d.isAttractor ? "var(--foreground)" : "transparent"))
        .attr("stroke-width", (d) => d.idx === selectedNodeRef.current ? 3 : (d.isAttractor ? 2 : 0))
        .attr("cursor", "pointer")
        .on("click", (_event, d) => onNodeSelectRef.current(d.idx === selectedNodeRef.current ? null : d.idx))
        .on("mouseenter", function (_, d) {
          d3.select(this)
            .attr("fill-opacity", 1)
            .attr("stroke", "var(--foreground)")
            .attr("stroke-width", 2);
        })
        .on("mouseleave", function (_, d) {
          d3.select(this)
            .attr("fill-opacity", d.isAttractor ? 1 : 0.7)
            .attr("stroke", d.idx === selectedNodeRef.current ? "var(--primary)" : (d.isAttractor ? "var(--foreground)" : "transparent"))
            .attr("stroke-width", d.idx === selectedNodeRef.current ? 3 : (d.isAttractor ? 2 : 0));
        });

      if (isOtg) {
        nodeSel.call(
          d3.drag<SVGCircleElement, NodeDatum>()
            .on("start", (event, d) => {
              if (!event.active) simulation.alphaTarget(0.3).restart();
              d.fx = d.x;
              d.fy = d.y;
            })
            .on("drag", (event, d) => {
              d.fx = event.x;
              d.fy = event.y;
            })
            .on("end", (event, d) => {
              if (!event.active) simulation.alphaTarget(0);
              d.fx = null;
              d.fy = null;
            })
        );
      }

      // FR7.7: Trajectory overlay rings
      const trajGroup = g.append("g").attr("class", "trajectories");
      if (isOtg && trajectories) {
        Object.entries(trajectories).forEach(([algo, visited], ti) => {
          const visitedSet = new Set(visited);
          const ringNodes = nodes.filter((n) => visitedSet.has(n.idx));
          trajGroup
            .selectAll(`.traj-${algo}`)
            .data(ringNodes)
            .join("circle")
            .attr("class", `traj-${algo}`)
            .attr("r", (d) => nodeRadius(d.basinSize) + 3 + ti * 3)
            .attr("fill", "none")
            .attr("stroke", TRAJECTORY_COLORS[algo] ?? "#888")
            .attr("stroke-width", 2)
            .attr("stroke-dasharray", algo === "rrhc" ? "3,3" : "none")
            .attr("opacity", 0.7)
            .attr("pointer-events", "none");
        });
      }

      const showLabels = nodes.length <= 150;
      const labelData = showLabels ? nodes : nodes.filter((n) => n.isAttractor);
      const labelSel = g.append("g").attr("class", "labels")
        .selectAll("text")
        .data(labelData)
        .join("text")
        .text((d) => d.fitness.toFixed(2))
        .attr("font-size", "9px")
        .attr("font-family", "var(--font-mono)")
        .attr("fill", "var(--foreground)")
        .attr("text-anchor", "middle")
        .attr("dy", (d) => -nodeRadius(d.basinSize) - 4)
        .attr("opacity", 0.7)
        .attr("pointer-events", "none");
        
      // Title
      svg.append("text")
        .attr("x", 20)
        .attr("y", 30)
        .attr("fill", "var(--muted-foreground)")
        .attr("font-size", "12px")
        .attr("font-weight", "500")
        .attr("letter-spacing", "0.05em")
        .text(isOtg ? "ORC Transition Graph (OTG)" : "Local Optima Network (LON-d1)");

      return { linkSel, nodeSel, labelSel, trajGroup };
    }

    const graphs: any[] = [];
    if (viewMode === "otg" || viewMode === "side-by-side") {
      const svg = container.append("svg").node();
      graphs.push(createGraph(svg, activeOtgEdges, true));
    }
    if (viewMode === "lon" || viewMode === "side-by-side") {
      const svg = container.append("svg").node();
      graphs.push(createGraph(svg, activeLonEdges, false));
    }

    const radiusCache = new Map<number, number>();
    for (const n of nodes) {
      radiusCache.set(n.idx, nodeRadius(n.basinSize));
    }

    simulation.on("tick", () => {
      graphs.forEach(({ linkSel, nodeSel, labelSel, trajGroup }) => {
        linkSel.each(function (this: SVGLineElement, d: any) {
          const sx = d.source.x ?? 0;
          const sy = d.source.y ?? 0;
          const tx = d.target.x ?? 0;
          const ty = d.target.y ?? 0;
          const dx = tx - sx;
          const dy = ty - sy;
          const dist = Math.sqrt(dx * dx + dy * dy) || 1;
          const r = radiusCache.get(d.target.idx) ?? 6;
          this.setAttribute("x1", String(sx));
          this.setAttribute("y1", String(sy));
          this.setAttribute("x2", String(tx - (dx / dist) * r));
          this.setAttribute("y2", String(ty - (dy / dist) * r));
        });

        nodeSel.each(function (this: SVGCircleElement, d: any) {
          this.setAttribute("cx", String(d.x ?? 0));
          this.setAttribute("cy", String(d.y ?? 0));
        });
        labelSel.each(function (this: SVGTextElement, d: any) {
          this.setAttribute("x", String(d.x ?? 0));
          this.setAttribute("y", String(d.y ?? 0));
        });
        trajGroup.selectAll("circle").each(function (this: SVGCircleElement, d: any) {
          this.setAttribute("cx", String(d.x ?? 0));
          this.setAttribute("cy", String(d.y ?? 0));
        });
      });
    });

    return () => {
      simulation.stop();
    };
  }, [instance, otg, lon, viewMode, trajectories]);

  return (
    <div className="flex-1 flex flex-col min-w-0 relative">
      {otg && lon && (
        <div className="absolute top-3 right-3 z-10">
          <Tabs
            value={viewMode}
            onValueChange={(v) => setViewMode(v as ViewMode)}
          >
            <TabsList className="h-8">
              <TabsTrigger value="otg" className="text-xs px-3 h-6">
                OTG
              </TabsTrigger>
              <TabsTrigger value="lon" className="text-xs px-3 h-6">
                LON-d1
              </TabsTrigger>
              <TabsTrigger value="side-by-side" className="text-xs px-3 h-6">
                Side-by-side
              </TabsTrigger>
            </TabsList>
          </Tabs>
        </div>
      )}

      {!instance ? (
        <div className="flex-1 flex items-center justify-center">
          <p className="text-sm text-muted-foreground">
            Configure parameters and generate a landscape to begin.
          </p>
        </div>
      ) : (
        <div
          ref={containerRef}
          className="flex-1 flex w-full h-full min-h-0"
        />
      )}
    </div>
  );
}
