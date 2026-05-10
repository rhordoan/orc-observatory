import type { ILSIterationEvent, ILSResult, MetricsData } from "./types";

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export async function fetchGpuStatus(): Promise<boolean> {
  try {
    const res = await fetch(`${API}/api/gpu/status`);
    if (!res.ok) return false;
    const data = await res.json();
    return data.available === true;
  } catch {
    return false;
  }
}

export async function createInstance(params: {
  problem_type: string;
  n: number;
  k?: number;
  mu?: number;
  nu?: number;
  gamma_wmodel?: number;
  n_clauses?: number | null;
  clause_length?: number;
  seed?: number | null;
  use_gpu?: boolean;
}) {
  const res = await fetch(`${API}/api/instances`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function buildOTG(instanceId: string, gamma = 1.0) {
  const res = await fetch(`${API}/api/otg/build`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ instance_id: instanceId, gamma }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function buildLON(instanceId: string) {
  const res = await fetch(`${API}/api/lon/build`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ instance_id: instanceId }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function explainORC(
  instanceId: string,
  fromOptimum: number,
  toNeighbor: number,
  gamma = 1.0
) {
  const params = new URLSearchParams({
    instance_id: instanceId,
    from_optimum: String(fromOptimum),
    to_neighbor: String(toNeighbor),
    gamma: String(gamma),
  });
  const res = await fetch(`${API}/api/orc/explain?${params}`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export function streamOTG(
  instanceId: string,
  gamma: number,
  onMessage: (event: Record<string, unknown>) => void,
  onComplete: () => void
) {
  const wsUrl = API.replace(/^http/, "ws");
  const ws = new WebSocket(`${wsUrl}/ws/otg/stream`);
  let completed = false;

  ws.onopen = () => {
    ws.send(JSON.stringify({ instance_id: instanceId, gamma }));
  };

  ws.onmessage = (evt) => {
    let data: Record<string, unknown>;
    try {
      data = JSON.parse(evt.data);
    } catch {
      return;
    }
    onMessage(data);
    if (data.type === "complete") {
      completed = true;
      onComplete();
      ws.close();
    }
  };

  ws.onerror = () => ws.close();

  ws.onclose = () => {
    if (!completed) {
      completed = true;
      onComplete();
    }
  };

  return () => ws.close();
}

/* ---- F2: ILS Race streaming ---- */

export function streamILS(
  instanceId: string,
  budget: number,
  d_r: number,
  seed: number | null,
  paceMs: number,
  onEvent: (event: ILSIterationEvent) => void,
  onComplete: (winner: string, results: ILSResult[]) => void
): () => void {
  const wsUrl = API.replace(/^http/, "ws");
  const ws = new WebSocket(`${wsUrl}/ws/ils/stream`);
  let completed = false;

  ws.onopen = () => {
    ws.send(
      JSON.stringify({
        instance_id: instanceId,
        budget,
        d_r,
        seed,
        pace_ms: paceMs,
      })
    );
  };

  ws.onmessage = (evt) => {
    let data: Record<string, unknown>;
    try {
      data = JSON.parse(evt.data);
    } catch {
      return;
    }
    if (data.type === "iteration") {
      onEvent(data as unknown as ILSIterationEvent);
    } else if (data.type === "complete" || data.type === "cancelled") {
      completed = true;
      onComplete(
        (data.winner as string) ?? "",
        (data.results as ILSResult[]) ?? []
      );
      ws.close();
    }
  };

  ws.onerror = () => ws.close();

  ws.onclose = () => {
    if (!completed) {
      completed = true;
      onComplete("", []);
    }
  };

  return () => {
    try {
      ws.send(JSON.stringify({ type: "cancel" }));
    } catch {
      /* ws already closed */
    }
    ws.close();
  };
}

/* ---- 3D landscape: full fitness vector ---- */

export async function fetchFitnessGrid(
  instanceId: string
): Promise<{ fitness: number[] }> {
  const res = await fetch(`${API}/api/instances/${instanceId}/fitness-grid`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

/* ---- F2: Difficulty metrics ---- */

export async function fetchMetrics(instanceId: string): Promise<MetricsData> {
  const res = await fetch(`${API}/api/metrics/${instanceId}`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}
