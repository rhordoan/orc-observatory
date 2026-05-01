"use client";

import { useEffect, useRef, useState } from "react";
import * as d3 from "d3";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import type { InstanceData, OTGData, LONData } from "@/lib/types";

interface GraphCanvasProps {
  instance: InstanceData | null;
  otg: OTGData | null;
  lon: LONData | null;
  selectedNode: number | null;
  onNodeSelect: (idx: number | null) => void;
}

type ViewMode = "otg" | "lon" | "side-by-side";

export function GraphCanvas({
  instance,
  otg,
  lon,
  selectedNode,
  onNodeSelect,
}: GraphCanvasProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [viewMode, setViewMode] = useState<ViewMode>("otg");

  useEffect(() => {
    if (!svgRef.current || !instance || !otg) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const rect = svgRef.current.getBoundingClientRect();
    const width = rect.width;
    const height = rect.height;

    const g = svg
      .append("g")
      .attr("class", "graph-container");

    // Zoom behavior
    const zoom = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.1, 8])
      .on("zoom", (event) => {
        g.attr("transform", event.transform);
      });
    svg.call(zoom);

    const activeEdges =
      viewMode === "lon" && lon
        ? lon.edges.map((e) => ({
            source: e.source,
            target: e.target,
            kappa: 0,
          }))
        : otg.edges.map((e) => ({
            source: e.source,
            target: e.target,
            kappa: e.min_kappa,
          }));

    const optima = instance.optima;
    const funnels = otg.funnels;

    // Build funnel membership map for coloring
    const funnelOf = new Map<number, number>();
    funnels.forEach((f, fi) => {
      f.member_indices.forEach((mi) => funnelOf.set(mi, fi));
    });

    // Funnel color scale (distinguishable, not rainbow)
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

    // ORC color scale for edges (green = 0, red = most negative)
    const minKappa = d3.min(activeEdges, (d) => d.kappa) ?? -1;
    const kappaColor = d3.scaleLinear<string>()
      .domain([minKappa, 0])
      .range(["oklch(0.6 0.2 25)", "oklch(0.65 0.15 150)"])
      .clamp(true);

    // Basin-proportional node sizes
    const maxBasin = d3.max(optima, (o) => o.basin_size) ?? 1;
    const nodeRadius = (basinSize: number) =>
      4 + 14 * Math.sqrt(basinSize / maxBasin);

    // Node data
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

    type LinkDatum = d3.SimulationLinkDatum<NodeDatum> & {
      kappa: number;
    };

    const links: LinkDatum[] = activeEdges
      .filter((e) => e.source !== e.target)
      .map((e) => ({
        source: e.source,
        target: e.target,
        kappa: e.kappa,
      }));

    // Force simulation
    const simulation = d3
      .forceSimulation(nodes)
      .force(
        "link",
        d3
          .forceLink(links)
          .id((d) => (d as NodeDatum).idx)
          .distance(60)
      )
      .force("charge", d3.forceManyBody().strength(-200))
      .force("center", d3.forceCenter(width / 2, height / 2))
      .force("collision", d3.forceCollide().radius((d) =>
        nodeRadius((d as NodeDatum).basinSize) + 2
      ));

    // Arrow markers
    svg
      .append("defs")
      .append("marker")
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

    // Edges
    const linkSel = g
      .append("g")
      .attr("class", "edges")
      .selectAll("line")
      .data(links)
      .join("line")
      .attr("stroke", (d) =>
        viewMode === "otg" ? kappaColor(d.kappa) : "var(--muted-foreground)"
      )
      .attr("stroke-width", 1.5)
      .attr("stroke-opacity", 0.6)
      .attr("marker-end", "url(#arrowhead)");

    // Nodes
    const nodeSel = g
      .append("g")
      .attr("class", "nodes")
      .selectAll<SVGCircleElement, NodeDatum>("circle")
      .data(nodes)
      .join("circle")
      .attr("r", (d) => nodeRadius(d.basinSize))
      .attr("fill", (d) => funnelColors[d.funnelIdx % funnelColors.length])
      .attr("fill-opacity", (d) => (d.isAttractor ? 1 : 0.7))
      .attr("stroke", (d) =>
        d.isAttractor ? "var(--foreground)" : "transparent"
      )
      .attr("stroke-width", (d) => (d.isAttractor ? 2 : 0))
      .attr("cursor", "pointer")
      .on("click", (_event, d) => {
        onNodeSelect(d.idx === selectedNode ? null : d.idx);
      })
      .on("mouseenter", function (_, d) {
        d3.select(this)
          .transition()
          .duration(150)
          .attr("fill-opacity", 1)
          .attr("stroke", "var(--foreground)")
          .attr("stroke-width", 2);
      })
      .on("mouseleave", function (_, d) {
        d3.select(this)
          .transition()
          .duration(150)
          .attr("fill-opacity", d.isAttractor ? 1 : 0.7)
          .attr("stroke", d.isAttractor ? "var(--foreground)" : "transparent")
          .attr("stroke-width", d.isAttractor ? 2 : 0);
      })
      .call(
        d3
          .drag<SVGCircleElement, NodeDatum>()
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

    // Fitness labels on nodes
    const labelSel = g
      .append("g")
      .attr("class", "labels")
      .selectAll("text")
      .data(nodes)
      .join("text")
      .text((d) => d.fitness.toFixed(2))
      .attr("font-size", "9px")
      .attr("font-family", "var(--font-mono)")
      .attr("fill", "var(--foreground)")
      .attr("text-anchor", "middle")
      .attr("dy", (d) => -nodeRadius(d.basinSize) - 4)
      .attr("opacity", 0.7)
      .attr("pointer-events", "none");

    simulation.on("tick", () => {
      linkSel
        .attr("x1", (d) => ((d.source as NodeDatum).x ?? 0))
        .attr("y1", (d) => ((d.source as NodeDatum).y ?? 0))
        .attr("x2", (d) => {
          const s = d.source as NodeDatum;
          const t = d.target as NodeDatum;
          const dx = (t.x ?? 0) - (s.x ?? 0);
          const dy = (t.y ?? 0) - (s.y ?? 0);
          const dist = Math.sqrt(dx * dx + dy * dy) || 1;
          const r = nodeRadius(t.basinSize);
          return (t.x ?? 0) - (dx / dist) * r;
        })
        .attr("y2", (d) => {
          const s = d.source as NodeDatum;
          const t = d.target as NodeDatum;
          const dx = (t.x ?? 0) - (s.x ?? 0);
          const dy = (t.y ?? 0) - (s.y ?? 0);
          const dist = Math.sqrt(dx * dx + dy * dy) || 1;
          const r = nodeRadius(t.basinSize);
          return (t.y ?? 0) - (dy / dist) * r;
        });

      nodeSel.attr("cx", (d) => d.x ?? 0).attr("cy", (d) => d.y ?? 0);

      labelSel.attr("x", (d) => d.x ?? 0).attr("y", (d) => d.y ?? 0);
    });

    return () => {
      simulation.stop();
    };
  }, [instance, otg, lon, viewMode, selectedNode, onNodeSelect]);

  return (
    <div className="flex-1 flex flex-col min-w-0 relative">
      {otg && lon && (
        <div className="absolute top-3 left-3 z-10">
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
        <svg
          ref={svgRef}
          className="flex-1 w-full h-full"
          style={{ minHeight: 0 }}
        />
      )}
    </div>
  );
}
