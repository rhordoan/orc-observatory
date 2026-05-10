"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import * as d3 from "d3";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import type { InstanceData, OTGData, LONData } from "@/lib/types";

const TRAJECTORY_COLORS: Record<string, string> = {
  orc: "#d4a03c",
  random: "#5b8bd4",
  rrhc: "#8a8a8a",
};

const FUNNEL_COLORS = [
  "#d4a03c", // amber
  "#3cb88c", // teal
  "#5b8bd4", // blue
  "#d47c3c", // coral
  "#9b6bbd", // purple
  "#8bc45b", // lime
  "#3cb8c8", // cyan
  "#d46b8c", // pink
];

interface GraphCanvasProps {
  instance: InstanceData | null;
  otg: OTGData | null;
  lon: LONData | null;
  selectedNode: number | null;
  onNodeSelect: (idx: number | null) => void;
  trajectories?: Record<string, number[]> | null;
  view3D?: boolean;
  onToggle3D?: () => void;
  show3DToggle?: boolean;
}

type ViewMode = "otg" | "lon" | "side-by-side";

type NodeDatum = d3.SimulationNodeDatum & {
  idx: number;
  fitness: number;
  basinSize: number;
  funnelIdx: number;
  isAttractor: boolean;
  radius: number;
};

type EdgeDatum = {
  source: NodeDatum;
  target: NodeDatum;
  kappa: number;
  color: string;
};

export function GraphCanvas({
  instance,
  otg,
  lon,
  selectedNode,
  onNodeSelect,
  trajectories,
  view3D,
  onToggle3D,
  show3DToggle,
}: GraphCanvasProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const tooltipRef = useRef<HTMLDivElement>(null);
  const [viewMode, setViewMode] = useState<ViewMode>("otg");

  const selectedNodeRef = useRef(selectedNode);
  const onNodeSelectRef = useRef(onNodeSelect);
  const hoveredRef = useRef<NodeDatum | null>(null);
  const dragNodeRef = useRef<NodeDatum | null>(null);

  const pathAnimRef = useRef<{
    pathIndices: number[];
    progress: number;
    active: boolean;
    rafId: number;
  } | null>(null);
  const redrawRef = useRef<(() => void) | null>(null);

  useEffect(() => {
    selectedNodeRef.current = selectedNode;
    onNodeSelectRef.current = onNodeSelect;
  }, [selectedNode, onNodeSelect]);

  useEffect(() => {
    if (!containerRef.current || !instance || !otg) return;

    const el = containerRef.current;
    el.innerHTML = "";

    const totalWidth = el.clientWidth;
    const height = el.clientHeight;
    if (totalWidth <= 0 || height <= 0) return;

    const dpr = window.devicePixelRatio || 1;

    const cs = getComputedStyle(document.documentElement);
    const thFg = cs.getPropertyValue("--foreground").trim() || "#1a1a1a";
    const thMuted = cs.getPropertyValue("--muted-foreground").trim() || "#888";
    const thPrimary = cs.getPropertyValue("--primary").trim() || "#d4a03c";
    const thBorder = cs.getPropertyValue("--border").trim() || "#ddd";

    const toColor = (v: string) =>
      v.startsWith("oklch") ? v : v.startsWith("#") ? v : `oklch(${v})`;

    const fgColor = toColor(thFg);
    const mutedColor = toColor(thMuted);
    const primaryColor = toColor(thPrimary);
    const borderColor = toColor(thBorder);

    const optima = instance.optima;
    const funnels = otg.funnels;

    const funnelOf = new Map<number, number>();
    funnels.forEach((f, fi) => {
      f.member_indices.forEach((mi) => funnelOf.set(mi, fi));
    });

    const isDark = document.documentElement.classList.contains("dark");
    const minKappa = d3.min(otg.edges, (d) => d.min_kappa) ?? -1;
    const kappaColor = d3.scaleLinear<string>()
      .domain([minKappa, 0])
      .range(isDark ? ["#d4713a", "#5aaa7a"] : ["#b8432a", "#3d8a5a"])
      .clamp(true);

    const maxBasin = d3.max(optima, (o) => o.basin_size) ?? 1;
    const nodeRadius = (basinSize: number) =>
      4 + 14 * Math.sqrt(basinSize / maxBasin);

    const attractorSet = new Set(funnels.map((f) => f.attractor_idx));

    const nodes: NodeDatum[] = optima.map((o) => ({
      idx: o.list_idx,
      fitness: o.fitness,
      basinSize: o.basin_size,
      funnelIdx: funnelOf.get(o.list_idx) ?? 0,
      isAttractor: attractorSet.has(o.list_idx),
      radius: nodeRadius(o.basin_size),
    }));

    const nodeByIdx = new Map(nodes.map((n) => [n.idx, n]));

    function resolveEdges(
      raw: { source: number; target: number; min_kappa: number }[],
      isOtg: boolean
    ): EdgeDatum[] {
      return raw
        .map((e) => {
          const src = nodeByIdx.get(e.source);
          const tgt = nodeByIdx.get(e.target);
          if (!src || !tgt || src === tgt) return null;
          const color = isOtg
            ? FUNNEL_COLORS[src.funnelIdx % FUNNEL_COLORS.length]
            : mutedColor;
          return {
            source: src,
            target: tgt,
            kappa: e.min_kappa,
            color,
          };
        })
        .filter((e): e is EdgeDatum => e !== null);
    }

    const otgEdges = resolveEdges(otg.edges, true);
    const lonEdges = lon ? resolveEdges(lon.edges.map((e) => ({ ...e, min_kappa: 0 })), false) : [];

    const simLinks = otgEdges.filter((e) => e.source !== e.target);

    const isLarge = nodes.length > 200;

    const simulation = d3
      .forceSimulation(nodes)
      .force(
        "link",
        d3.forceLink(simLinks).id((d) => (d as NodeDatum).idx).distance(isLarge ? 30 : 60)
      )
      .force("charge", d3.forceManyBody().strength(isLarge ? -80 : -200).theta(isLarge ? 0.9 : 0.8))
      .force("center", d3.forceCenter(0, 0))
      .force("collision", d3.forceCollide().radius((d) => (d as NodeDatum).radius + 2));

    if (isLarge) simulation.alphaDecay(0.05);

    const showLabels = nodes.length <= 150;
    const labelNodes = showLabels ? nodes : nodes.filter((n) => n.isAttractor);

    const trajSets: { algo: string; color: string; visited: Set<number>; ring: number }[] = [];
    if (trajectories) {
      Object.entries(trajectories).forEach(([algo, visited], ti) => {
        trajSets.push({
          algo,
          color: TRAJECTORY_COLORS[algo] ?? "#888",
          visited: new Set(visited),
          ring: 3 + ti * 3,
        });
      });
    }

    type GraphView = {
      canvas: HTMLCanvasElement;
      ctx: CanvasRenderingContext2D;
      edges: EdgeDatum[];
      isOtg: boolean;
      transform: d3.ZoomTransform;
      width: number;
      title: string;
    };

    function createCanvasView(w: number, edges: EdgeDatum[], isOtg: boolean, title: string): GraphView {
      const canvas = document.createElement("canvas");
      canvas.style.width = `${w}px`;
      canvas.style.height = `${height}px`;
      canvas.width = w * dpr;
      canvas.height = height * dpr;
      el.appendChild(canvas);

      const ctx = canvas.getContext("2d")!;
      ctx.scale(dpr, dpr);

      return { canvas, ctx, edges, isOtg, transform: d3.zoomIdentity, width: w, title };
    }

    const views: GraphView[] = [];
    const isSideBySide = viewMode === "side-by-side";
    const panelW = isSideBySide ? totalWidth / 2 : totalWidth;

    if (viewMode === "otg" || isSideBySide)
      views.push(createCanvasView(panelW, otgEdges, true, "ORC Transition Graph (OTG)"));
    if (viewMode === "lon" || isSideBySide)
      views.push(createCanvasView(panelW, lonEdges, false, "Local Optima Network (LON-d1)"));

    function drawArrow(ctx: CanvasRenderingContext2D, sx: number, sy: number, ex: number, ey: number) {
      const headLen = 6;
      const angle = Math.atan2(ey - sy, ex - sx);
      ctx.beginPath();
      ctx.moveTo(ex, ey);
      ctx.lineTo(ex - headLen * Math.cos(angle - 0.4), ey - headLen * Math.sin(angle - 0.4));
      ctx.lineTo(ex - headLen * Math.cos(angle + 0.4), ey - headLen * Math.sin(angle + 0.4));
      ctx.closePath();
      ctx.fill();
    }

    function drawView(view: GraphView) {
      const { ctx, edges, isOtg, transform, width: w, title } = view;
      ctx.save();
      ctx.clearRect(0, 0, w, height);

      ctx.translate(w / 2, height / 2);
      ctx.translate(transform.x, transform.y);
      ctx.scale(transform.k, transform.k);

      // Edges
      ctx.lineWidth = 1.5;
      ctx.globalAlpha = 0.6;
      for (const e of edges) {
        const sx = e.source.x ?? 0;
        const sy = e.source.y ?? 0;
        const tx = e.target.x ?? 0;
        const ty = e.target.y ?? 0;
        const dx = tx - sx;
        const dy = ty - sy;
        const dist = Math.sqrt(dx * dx + dy * dy) || 1;
        const r = e.target.radius;
        const ex = tx - (dx / dist) * r;
        const ey = ty - (dy / dist) * r;

        ctx.strokeStyle = e.color;
        ctx.beginPath();
        ctx.moveTo(sx, sy);
        ctx.lineTo(ex, ey);
        ctx.stroke();

        ctx.fillStyle = e.color;
        drawArrow(ctx, sx, sy, ex, ey);
      }
      ctx.globalAlpha = 1;

      // Trajectory rings
      if (isOtg && trajSets.length > 0) {
        ctx.globalAlpha = 0.7;
        ctx.lineWidth = 2;
        for (const traj of trajSets) {
          ctx.strokeStyle = traj.color;
          const isDashed = traj.algo === "rrhc";
          if (isDashed) ctx.setLineDash([3, 3]);
          for (const n of nodes) {
            if (!traj.visited.has(n.idx)) continue;
            ctx.beginPath();
            ctx.arc(n.x ?? 0, n.y ?? 0, n.radius + traj.ring, 0, Math.PI * 2);
            ctx.stroke();
          }
          if (isDashed) ctx.setLineDash([]);
        }
        ctx.globalAlpha = 1;
      }

      // Nodes
      const sel = selectedNodeRef.current;
      const hov = hoveredRef.current;
      for (const n of nodes) {
        const cx = n.x ?? 0;
        const cy = n.y ?? 0;
        const isSel = n.idx === sel;
        const isHov = n === hov;
        const color = FUNNEL_COLORS[n.funnelIdx % FUNNEL_COLORS.length];

        ctx.beginPath();
        ctx.arc(cx, cy, n.radius, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.globalAlpha = n.isAttractor || isHov ? 1 : 0.7;
        ctx.fill();
        ctx.globalAlpha = 1;

        if (isSel) {
          ctx.strokeStyle = primaryColor;
          ctx.lineWidth = 3;
          ctx.stroke();
        } else if (isHov || n.isAttractor) {
          ctx.strokeStyle = fgColor;
          ctx.lineWidth = 2;
          ctx.stroke();
        }
      }

      // Labels
      ctx.font = "9px monospace";
      ctx.fillStyle = fgColor;
      ctx.textAlign = "center";
      ctx.globalAlpha = 0.8;
      for (const n of labelNodes) {
        ctx.fillText(n.fitness.toFixed(2), n.x ?? 0, (n.y ?? 0) - n.radius - 4);
      }
      ctx.globalAlpha = 1;

      // OTG path animation overlay
      const anim = pathAnimRef.current;
      if (anim?.active && anim.pathIndices.length >= 2 && isOtg) {
        const pNodes = anim.pathIndices.map((i) => nodeByIdx.get(i)).filter(Boolean) as NodeDatum[];
        if (pNodes.length >= 2) {
          const maxEdge = pNodes.length - 1;
          const progress = anim.progress % maxEdge;
          const edgeI = Math.min(Math.floor(progress), maxEdge - 1);
          const tFrac = progress - edgeI;

          // Highlighted path edges
          ctx.globalAlpha = 0.3;
          ctx.strokeStyle = primaryColor;
          ctx.lineWidth = 3;
          for (let i = 0; i < maxEdge; i++) {
            ctx.beginPath();
            ctx.moveTo(pNodes[i].x ?? 0, pNodes[i].y ?? 0);
            ctx.lineTo(pNodes[i + 1].x ?? 0, pNodes[i + 1].y ?? 0);
            ctx.stroke();
          }

          // Attractor pulsing ring
          const att = pNodes[pNodes.length - 1];
          const pulse = 0.5 + 0.5 * Math.sin(performance.now() / 300);
          ctx.globalAlpha = 0.25 + 0.2 * pulse;
          ctx.strokeStyle = primaryColor;
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.arc(att.x ?? 0, att.y ?? 0, att.radius + 6 + 3 * pulse, 0, Math.PI * 2);
          ctx.stroke();

          // Trail
          const TRAIL = 8;
          const SPAN = 0.4;
          for (let ti = TRAIL; ti > 0; ti--) {
            const tp = progress - (ti * SPAN) / TRAIL;
            if (tp < 0) continue;
            const tei = Math.min(Math.floor(tp), maxEdge - 1);
            const tt = tp - tei;
            const tsx = (pNodes[tei].x ?? 0) + tt * ((pNodes[tei + 1].x ?? 0) - (pNodes[tei].x ?? 0));
            const tsy = (pNodes[tei].y ?? 0) + tt * ((pNodes[tei + 1].y ?? 0) - (pNodes[tei].y ?? 0));
            ctx.globalAlpha = 0.18 * (1 - ti / TRAIL);
            ctx.fillStyle = primaryColor;
            ctx.beginPath();
            ctx.arc(tsx, tsy, 4 - ti * 0.3, 0, Math.PI * 2);
            ctx.fill();
          }

          // Glowing pulse particle
          const src = pNodes[edgeI];
          const dst = pNodes[edgeI + 1];
          const px = (src.x ?? 0) + tFrac * ((dst.x ?? 0) - (src.x ?? 0));
          const py = (src.y ?? 0) + tFrac * ((dst.y ?? 0) - (src.y ?? 0));
          ctx.globalAlpha = 1;
          ctx.save();
          ctx.shadowColor = primaryColor;
          ctx.shadowBlur = 14;
          ctx.fillStyle = primaryColor;
          ctx.beginPath();
          ctx.arc(px, py, 5, 0, Math.PI * 2);
          ctx.fill();
          ctx.restore();
          ctx.fillStyle = "#fff";
          ctx.beginPath();
          ctx.arc(px, py, 2, 0, Math.PI * 2);
          ctx.fill();
          ctx.globalAlpha = 1;
        }
      }

      ctx.restore();

      ctx.fillStyle = mutedColor;
      ctx.font = "500 12px sans-serif";
      ctx.textAlign = "left";
      ctx.fillText(title, 20, 30);
    }

    let animFrame = 0;
    function scheduleRedraw() {
      cancelAnimationFrame(animFrame);
      animFrame = requestAnimationFrame(() => views.forEach(drawView));
    }
    redrawRef.current = scheduleRedraw;

    simulation.on("tick", scheduleRedraw);

    // Zoom
    const zooms: d3.ZoomBehavior<HTMLCanvasElement, unknown>[] = [];
    views.forEach((view) => {
      const zoom = d3.zoom<HTMLCanvasElement, unknown>()
        .scaleExtent([0.1, 8])
        .on("zoom", (event) => {
          view.transform = event.transform;
          scheduleRedraw();
        });
      d3.select(view.canvas).call(zoom);
      zooms.push(zoom);
    });

    let fitted = false;
    simulation.on("end.fit", () => {
      if (fitted) return;
      fitted = true;

      let xMin = Infinity, xMax = -Infinity, yMin = Infinity, yMax = -Infinity;
      for (const n of nodes) {
        const r = n.radius;
        xMin = Math.min(xMin, (n.x ?? 0) - r);
        xMax = Math.max(xMax, (n.x ?? 0) + r);
        yMin = Math.min(yMin, (n.y ?? 0) - r);
        yMax = Math.max(yMax, (n.y ?? 0) + r);
      }

      const pad = 40;
      const bw = (xMax - xMin) + pad * 2;
      const bh = (yMax - yMin) + pad * 2;
      const cx = (xMin + xMax) / 2;
      const cy = (yMin + yMax) / 2;

      views.forEach((view, vi) => {
        const k = Math.min(view.width / bw, height / bh, 1.5);
        const tx = -cx * k;
        const ty = -cy * k;
        const t = d3.zoomIdentity.translate(tx, ty).scale(k);
        view.transform = t;
        d3.select(view.canvas).call(zooms[vi].transform, t);
      });
      scheduleRedraw();
    });

    // Hit testing
    function hitTest(view: GraphView, mx: number, my: number): NodeDatum | null {
      const t = view.transform;
      const x = (mx - view.width / 2 - t.x) / t.k;
      const y = (my - height / 2 - t.y) / t.k;

      let closest: NodeDatum | null = null;
      let closestDist = Infinity;
      for (const n of nodes) {
        const dx = (n.x ?? 0) - x;
        const dy = (n.y ?? 0) - y;
        const d = dx * dx + dy * dy;
        const r = n.radius + 2;
        if (d < r * r && d < closestDist) {
          closest = n;
          closestDist = d;
        }
      }
      return closest;
    }

    // Mouse interactions
    const primaryView = views[0];
    if (primaryView) {
      const canvas = primaryView.canvas;

      canvas.addEventListener("mousemove", (e) => {
        const rect = canvas.getBoundingClientRect();
        const mx = e.clientX - rect.left;
        const my = e.clientY - rect.top;

        if (dragNodeRef.current) {
          const t = primaryView.transform;
          dragNodeRef.current.fx = (mx - primaryView.width / 2 - t.x) / t.k;
          dragNodeRef.current.fy = (my - height / 2 - t.y) / t.k;
          simulation.alphaTarget(0.3).restart();
          return;
        }

        const hit = hitTest(primaryView, mx, my);
        if (hit !== hoveredRef.current) {
          hoveredRef.current = hit;
          canvas.style.cursor = hit ? "pointer" : "default";
          scheduleRedraw();
        }

        const tip = tooltipRef.current;
        if (tip) {
          if (hit && !dragNodeRef.current) {
            const funnel = otg.funnels.find((f) => f.member_indices.includes(hit.idx));
            tip.innerHTML =
              `<div class="font-semibold">${instance.optima[hit.idx]?.label ?? hit.idx}</div>` +
              `<div>Fitness: <span class="font-mono">${hit.fitness.toFixed(4)}</span></div>` +
              `<div>Basin: <span class="font-mono">${hit.basinSize}</span></div>` +
              (funnel ? `<div>Funnel: <span class="font-mono">${funnel.member_indices.length} members</span></div>` : "") +
              (hit.isAttractor ? `<div class="text-primary font-medium">Attractor</div>` : "");
            tip.style.left = `${mx + 14}px`;
            tip.style.top = `${my - 10}px`;
            tip.style.opacity = "1";
          } else {
            tip.style.opacity = "0";
          }
        }
      });

      canvas.addEventListener("mousedown", (e) => {
        const rect = canvas.getBoundingClientRect();
        const hit = hitTest(primaryView, e.clientX - rect.left, e.clientY - rect.top);
        if (hit) {
          e.stopPropagation();
          dragNodeRef.current = hit;
          hit.fx = hit.x;
          hit.fy = hit.y;
          simulation.alphaTarget(0.3).restart();

          const onUp = () => {
            if (dragNodeRef.current) {
              dragNodeRef.current.fx = null;
              dragNodeRef.current.fy = null;
              dragNodeRef.current = null;
              simulation.alphaTarget(0);
            }
            window.removeEventListener("mouseup", onUp);
            window.removeEventListener("mousemove", onMove);
          };
          const onMove = (ev: MouseEvent) => {
            if (!dragNodeRef.current) return;
            const r = canvas.getBoundingClientRect();
            const t = primaryView.transform;
            dragNodeRef.current.fx = (ev.clientX - r.left - primaryView.width / 2 - t.x) / t.k;
            dragNodeRef.current.fy = (ev.clientY - r.top - height / 2 - t.y) / t.k;
          };
          window.addEventListener("mouseup", onUp);
          window.addEventListener("mousemove", onMove);
        }
      });

      canvas.addEventListener("click", (e) => {
        if (dragNodeRef.current) return;
        const rect = canvas.getBoundingClientRect();
        const hit = hitTest(primaryView, e.clientX - rect.left, e.clientY - rect.top);
        if (hit) {
          onNodeSelectRef.current(hit.idx === selectedNodeRef.current ? null : hit.idx);
        } else {
          onNodeSelectRef.current(null);
        }
      });
    }

    return () => {
      simulation.stop();
      cancelAnimationFrame(animFrame);
    };
  }, [instance, otg, lon, viewMode, trajectories]);

  // Path-to-attractor animation lifecycle
  useEffect(() => {
    if (selectedNode === null || !otg || !instance) {
      if (pathAnimRef.current) {
        cancelAnimationFrame(pathAnimRef.current.rafId);
        pathAnimRef.current = null;
        redrawRef.current?.();
      }
      return;
    }

    const successor = new Map<number, number>();
    for (const e of otg.edges) {
      if (e.source !== e.target) successor.set(e.source, e.target);
    }

    const path: number[] = [selectedNode];
    const visited = new Set([selectedNode]);
    let cur = selectedNode;
    while (successor.has(cur)) {
      const nxt = successor.get(cur)!;
      path.push(nxt);
      if (visited.has(nxt)) break;
      visited.add(nxt);
      cur = nxt;
    }

    if (path.length < 2) {
      pathAnimRef.current = null;
      return;
    }

    const anim = { pathIndices: path, progress: 0, active: true, rafId: 0 };
    pathAnimRef.current = anim;

    const EDGES_PER_SEC = 1.8;
    let last = performance.now();

    function tick(now: number) {
      if (!anim.active) return;
      const dt = (now - last) / 1000;
      last = now;
      anim.progress += dt * EDGES_PER_SEC;
      if (anim.progress >= path.length - 1) {
        anim.progress = 0;
      }
      redrawRef.current?.();
      anim.rafId = requestAnimationFrame(tick);
    }
    anim.rafId = requestAnimationFrame(tick);

    return () => {
      anim.active = false;
      cancelAnimationFrame(anim.rafId);
    };
  }, [selectedNode, otg, instance]);

  return (
    <div className="flex-1 flex flex-col min-w-0 relative">
      {otg && (
        <div className="absolute top-3 right-3 z-10 flex items-center gap-2">
          {lon && (
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
          )}
          {show3DToggle && onToggle3D && (
            <button
              onClick={onToggle3D}
              className={`px-3 py-1 text-xs rounded-md border transition-colors h-8 ${
                view3D
                  ? "bg-primary/15 text-primary border-primary/30 font-medium"
                  : "bg-card text-muted-foreground border-border hover:text-foreground hover:bg-muted/50"
              }`}
            >
              {view3D ? "3D Landscape" : "3D"}
            </button>
          )}
        </div>
      )}

      {!instance ? (
        <div className="flex-1 flex items-center justify-center">
          <p className="text-sm text-muted-foreground">
            Configure parameters and generate a landscape to begin.
          </p>
        </div>
      ) : (
        <div className="relative flex-1 flex min-h-0">
          <div
            ref={containerRef}
            className="flex-1 flex w-full h-full min-h-0"
          />
          <div
            ref={tooltipRef}
            className="absolute z-20 pointer-events-none rounded-md border border-border bg-popover px-3 py-2 text-xs text-popover-foreground shadow-md transition-opacity duration-150 opacity-0 max-w-[220px]"
          />
        </div>
      )}
    </div>
  );
}
