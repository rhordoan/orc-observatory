"use client";

import { useEffect, useMemo, useRef, useState, useCallback } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { OrbitControls, Text } from "@react-three/drei";
import * as THREE from "three";
import { fetchFitnessGrid } from "@/lib/api";
import type { InstanceData, OTGData } from "@/lib/types";

const FUNNEL_COLORS = [
  "#d4a03c",
  "#3cb88c",
  "#5b8bd4",
  "#d47c3c",
  "#9b6bbd",
  "#8bc45b",
  "#3cb8c8",
  "#d46b8c",
];

interface LandscapeView3DProps {
  instance: InstanceData;
  otg: OTGData;
  selectedNode: number | null;
  onNodeSelect: (idx: number | null) => void;
}

function gridDims(n: number): [number, number] {
  const lowBits = Math.floor(n / 2);
  const highBits = Math.ceil(n / 2);
  return [1 << lowBits, 1 << highBits];
}

function idxToGrid(idx: number, gw: number): [number, number] {
  return [idx % gw, Math.floor(idx / gw)];
}

function fitnessToColor(f: number): THREE.Color {
  const r = 0.15 + f * 0.75;
  const g = 0.25 + f * 0.55;
  const b = 0.55 - f * 0.35;
  return new THREE.Color(r, g, b);
}

function Surface({
  fitness,
  gw,
  gh,
}: {
  fitness: number[];
  gw: number;
  gh: number;
}) {
  const meshRef = useRef<THREE.Mesh>(null);

  const geometry = useMemo(() => {
    const geo = new THREE.PlaneGeometry(gw - 1, gh - 1, gw - 1, gh - 1);
    const pos = geo.attributes.position;
    const colors = new Float32Array(pos.count * 3);

    for (let i = 0; i < pos.count; i++) {
      const gx = i % gw;
      const gy = Math.floor(i / gw);
      const idx = gy * gw + gx;
      const f = fitness[idx] ?? 0;

      pos.setZ(i, f * 4);

      const col = fitnessToColor(f);
      colors[i * 3] = col.r;
      colors[i * 3 + 1] = col.g;
      colors[i * 3 + 2] = col.b;
    }

    geo.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    geo.computeVertexNormals();
    return geo;
  }, [fitness, gw, gh]);

  return (
    <mesh ref={meshRef} geometry={geometry} rotation={[-Math.PI / 2, 0, 0]}>
      <meshStandardMaterial
        vertexColors
        side={THREE.DoubleSide}
        transparent
        opacity={0.85}
        roughness={0.6}
        metalness={0.1}
      />
    </mesh>
  );
}

function OptimumSphere({
  position,
  color,
  radius,
  isSelected,
  isAttractor,
  onClick,
}: {
  position: [number, number, number];
  color: string;
  radius: number;
  isSelected: boolean;
  isAttractor: boolean;
  onClick: () => void;
}) {
  const meshRef = useRef<THREE.Mesh>(null);
  const [hovered, setHovered] = useState(false);

  useFrame((_, delta) => {
    if (!meshRef.current) return;
    if (isAttractor) {
      meshRef.current.rotation.y += delta * 0.5;
    }
  });

  const scale = hovered ? 1.3 : isSelected ? 1.2 : 1;

  return (
    <mesh
      ref={meshRef}
      position={position}
      scale={scale}
      onClick={(e) => {
        e.stopPropagation();
        onClick();
      }}
      onPointerOver={(e) => {
        e.stopPropagation();
        setHovered(true);
        document.body.style.cursor = "pointer";
      }}
      onPointerOut={() => {
        setHovered(false);
        document.body.style.cursor = "auto";
      }}
    >
      {isAttractor ? (
        <octahedronGeometry args={[radius, 0]} />
      ) : (
        <sphereGeometry args={[radius, 16, 16]} />
      )}
      <meshStandardMaterial
        color={color}
        emissive={isSelected || hovered ? color : "#000000"}
        emissiveIntensity={isSelected ? 0.6 : hovered ? 0.3 : 0}
        roughness={0.3}
        metalness={0.4}
      />
    </mesh>
  );
}

function OTGArc({
  from,
  to,
  color,
}: {
  from: [number, number, number];
  to: [number, number, number];
  color: string;
}) {
  const points = useMemo(() => {
    const mid: [number, number, number] = [
      (from[0] + to[0]) / 2,
      Math.max(from[1], to[1]) + 0.8,
      (from[2] + to[2]) / 2,
    ];
    const curve = new THREE.QuadraticBezierCurve3(
      new THREE.Vector3(...from),
      new THREE.Vector3(...mid),
      new THREE.Vector3(...to)
    );
    return curve.getPoints(20);
  }, [from, to]);

  return (
    <line>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          args={[new Float32Array(points.flatMap((p) => [p.x, p.y, p.z])), 3]}
        />
      </bufferGeometry>
      <lineBasicMaterial color={color} transparent opacity={0.5} linewidth={1} />
    </line>
  );
}

function AxisLabels({ gw, gh }: { gw: number; gh: number }) {
  return (
    <>
      <Text
        position={[0, -0.6, (gh - 1) / 2 + 1.5]}
        fontSize={0.6}
        color="#888"
        anchorX="center"
      >
        x (low bits)
      </Text>
      <Text
        position={[-(gw - 1) / 2 - 1.5, -0.6, 0]}
        fontSize={0.6}
        color="#888"
        anchorX="center"
        rotation={[0, Math.PI / 2, 0]}
      >
        y (high bits)
      </Text>
      <Text
        position={[-(gw - 1) / 2 - 0.5, 2, -(gh - 1) / 2 - 0.5]}
        fontSize={0.6}
        color="#888"
        anchorX="center"
        rotation={[0, 0, Math.PI / 2]}
      >
        fitness
      </Text>
    </>
  );
}

function CameraSetup({ gw, gh }: { gw: number; gh: number }) {
  const { camera } = useThree();

  useEffect(() => {
    const dist = Math.max(gw, gh) * 1.2;
    camera.position.set(dist * 0.6, dist * 0.5, dist * 0.6);
    camera.lookAt(0, 1, 0);
  }, [camera, gw, gh]);

  return null;
}

function Scene({
  instance,
  otg,
  fitness,
  selectedNode,
  onNodeSelect,
}: LandscapeView3DProps & { fitness: number[] }) {
  const n = Math.log2(instance.space_size);
  const [gw, gh] = gridDims(n);

  const funnelOf = useMemo(() => {
    const map = new Map<number, number>();
    otg.funnels.forEach((f, fi) => {
      f.member_indices.forEach((mi) => map.set(mi, fi));
    });
    return map;
  }, [otg.funnels]);

  const attractorSet = useMemo(
    () => new Set(otg.funnels.map((f) => f.attractor_idx)),
    [otg.funnels]
  );

  const maxBasin = useMemo(
    () => Math.max(...instance.optima.map((o) => o.basin_size), 1),
    [instance.optima]
  );

  const optimaPositions = useMemo(() => {
    return instance.optima.map((o) => {
      const [gx, gy] = idxToGrid(o.solution_idx, gw);
      const x = gx - (gw - 1) / 2;
      const z = gy - (gh - 1) / 2;
      const y = o.fitness * 4 + 0.2;
      return [x, y, z] as [number, number, number];
    });
  }, [instance.optima, gw, gh]);

  return (
    <>
      <CameraSetup gw={gw} gh={gh} />
      <ambientLight intensity={0.5} />
      <directionalLight position={[10, 15, 10]} intensity={0.8} />
      <directionalLight position={[-5, 10, -5]} intensity={0.3} />

      <Surface fitness={fitness} gw={gw} gh={gh} />

      {instance.optima.map((o, i) => {
        const fi = funnelOf.get(o.list_idx) ?? 0;
        const color = FUNNEL_COLORS[fi % FUNNEL_COLORS.length];
        const r = 0.15 + 0.35 * Math.sqrt(o.basin_size / maxBasin);
        return (
          <OptimumSphere
            key={o.list_idx}
            position={optimaPositions[i]}
            color={color}
            radius={r}
            isSelected={selectedNode === o.list_idx}
            isAttractor={attractorSet.has(o.list_idx)}
            onClick={() => onNodeSelect(o.list_idx)}
          />
        );
      })}

      {otg.edges.map((e, i) => {
        const srcOpt = instance.optima.findIndex(
          (o) => o.list_idx === e.source
        );
        const tgtOpt = instance.optima.findIndex(
          (o) => o.list_idx === e.target
        );
        if (srcOpt < 0 || tgtOpt < 0) return null;
        const fi = funnelOf.get(e.source) ?? 0;
        const color = FUNNEL_COLORS[fi % FUNNEL_COLORS.length];
        return (
          <OTGArc
            key={`${e.source}-${e.target}-${i}`}
            from={optimaPositions[srcOpt]}
            to={optimaPositions[tgtOpt]}
            color={color}
          />
        );
      })}

      <AxisLabels gw={gw} gh={gh} />
      <OrbitControls makeDefault enableDamping dampingFactor={0.08} />
    </>
  );
}

export function LandscapeView3D(props: LandscapeView3DProps) {
  const { instance } = props;
  const [fitness, setFitness] = useState<number[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setFitness(null);
    setError(null);

    fetchFitnessGrid(instance.instance_id)
      .then((data) => {
        if (!cancelled) setFitness(data.fitness);
      })
      .catch((err) => {
        if (!cancelled) setError(String(err));
      });

    return () => {
      cancelled = true;
    };
  }, [instance.instance_id]);

  if (error) {
    return (
      <div className="flex-1 flex items-center justify-center text-destructive text-sm">
        Failed to load fitness grid: {error}
      </div>
    );
  }

  if (!fitness) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <div className="w-6 h-6 border-2 border-primary/30 border-t-primary rounded-full animate-spin" />
      </div>
    );
  }

  return (
    <div className="flex-1 relative" style={{ minHeight: 400 }}>
      <Canvas
        camera={{ fov: 50, near: 0.1, far: 200 }}
        gl={{ antialias: true, alpha: true }}
        style={{ background: "transparent" }}
      >
        <Scene {...props} fitness={fitness} />
      </Canvas>
    </div>
  );
}
