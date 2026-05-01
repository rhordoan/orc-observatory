const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

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

  ws.onopen = () => {
    ws.send(JSON.stringify({ instance_id: instanceId, gamma }));
  };

  ws.onmessage = (evt) => {
    const data = JSON.parse(evt.data);
    onMessage(data);
    if (data.type === "complete") {
      onComplete();
      ws.close();
    }
  };

  ws.onerror = () => ws.close();

  return () => ws.close();
}
