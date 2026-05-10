export interface OptimumInfo {
  list_idx: number;
  solution_idx: number;
  label: string;
  fitness: number;
  basin_size: number;
}

export interface InstanceData {
  instance_id: string;
  problem_type: string;
  name: string;
  space_size: number;
  degree: number;
  n_optima: number;
  optima: OptimumInfo[];
}

export interface OTGEdge {
  source: number;
  target: number;
  min_kappa: number;
  via_neighbor: number;
}

export interface FunnelInfo {
  attractor_idx: number;
  member_indices: number[];
  attractor_fitness: number;
  is_cycle: boolean;
}

export interface OTGData {
  instance_id: string;
  edges: OTGEdge[];
  funnels: FunnelInfo[];
  orc_values: Record<string, Record<string, number>>;
  compression_ratio: number;
  mean_terminal_rank: number;
  top5_reachability: number;
  dag_depth: number;
  has_cycles: boolean;
}

export interface LONEdge {
  source: number;
  target: number;
  via_neighbor: number;
  neighbor_fitness: number;
}

export interface LONData {
  instance_id: string;
  edges: LONEdge[];
  n_self_loops: number;
  singleton_fraction: number;
}

export interface ORCExplainData {
  x_idx: number;
  y_idx: number;
  kappa: number;
  w1: number;
  shared: number[];
  x_exclusive: number[];
  y_exclusive: number[];
  matching: number[][];
  pair_costs: number[];
  x_exclusive_fitness: number[];
  y_exclusive_fitness: number[];
  shared_labels: string[];
  x_exclusive_labels: string[];
  y_exclusive_labels: string[];
}

/* ---- F2: ILS Race types ---- */

export interface ILSIterationEvent {
  type: "iteration";
  algo: "orc" | "random" | "rrhc";
  evals: number;
  best_fitness: number;
  current_optimum: number;
}

export interface ILSResult {
  algo: string;
  best_fitness: number;
  total_evals: number;
  trajectory: number[];
}

export interface MetricsData {
  instance_id: string;
  fdc: number;
  autocorrelation_length: number;
  information_content: number;
  mean_orc: number;
}
