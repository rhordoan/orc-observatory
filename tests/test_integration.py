"""End-to-end integration tests for the full analysis pipeline.

Runs NK, W-model, and MAX-SAT instances through the complete pipeline:
search space -> local optima -> OTG -> LON -> metrics.
"""

import pytest

from lib.search_spaces import NKSearchSpace, WModelSearchSpace, MaxSATSearchSpace
from lib.hill_climb import enumerate_local_optima
from lib.orc import compute_orc, compute_all_orc
from lib.otg import build_otg
from lib.lon import build_lon_d1
from lib.metrics import fitness_distance_correlation, autocorrelation_length, information_content


class TestNKPipeline:
    """Full pipeline on a small NK instance."""

    @pytest.fixture
    def nk_setup(self):
        space = NKSearchSpace(n=8, k=4, seed=42)
        optima = enumerate_local_optima(space)
        return space, optima

    def test_optima_found(self, nk_setup):
        space, optima = nk_setup
        assert len(optima) > 1, "Should have multiple local optima for K=2"

    def test_optima_are_sorted(self, nk_setup):
        _, optima = nk_setup
        for i in range(len(optima) - 1):
            assert optima[i].fitness >= optima[i + 1].fitness

    def test_optima_are_genuine(self, nk_setup):
        space, optima = nk_setup
        for opt in optima:
            for nbr in space.neighbors(opt.idx):
                assert space.fitness(opt.idx) >= space.fitness(nbr), (
                    f"Optimum {opt.idx} has fitter neighbor {nbr}"
                )

    def test_otg_is_functional_graph(self, nk_setup):
        """Every node has exactly one outgoing edge (functional graph)."""
        space, optima = nk_setup
        result = build_otg(space, optima)
        assert len(result.edges) == len(optima)
        sources = [e.source for e in result.edges]
        assert sorted(sources) == list(range(len(optima)))

    def test_funnels_partition_all_optima(self, nk_setup):
        space, optima = nk_setup
        result = build_otg(space, optima)
        all_members = set()
        for f in result.funnels:
            for m in f.member_indices:
                all_members.add(m)
        assert all_members == set(range(len(optima)))

    def test_lon_mostly_self_loops(self, nk_setup):
        """LON-d1 typically has 90%+ self-loops on standard benchmarks."""
        space, optima = nk_setup
        result = build_lon_d1(space, optima)
        assert result.singleton_fraction > 0.5

    def test_otg_compression_in_unit_interval(self, nk_setup):
        """Compression ratio should be between 0 and 1."""
        space, optima = nk_setup
        result = build_otg(space, optima)
        assert 0 < result.compression_ratio <= 1.0

    def test_metrics_return_finite(self, nk_setup):
        space, _ = nk_setup
        optima = enumerate_local_optima(space)
        fdc = fitness_distance_correlation(space, optima)
        acl = autocorrelation_length(space, seed=0)
        ic = information_content(space, seed=0)
        for name, val in [("FDC", fdc), ("ACL", acl), ("IC", ic)]:
            assert not (val != val), f"{name} is NaN"


class TestWModelPipeline:

    def test_wmodel_pipeline(self):
        space = WModelSearchSpace(n=8, mu=1, nu=1, gamma=0, seed=42)
        optima = enumerate_local_optima(space)
        assert len(optima) >= 1
        result = build_otg(space, optima)
        assert len(result.edges) == len(optima)


class TestMaxSATPipeline:

    def test_maxsat_pipeline(self):
        space = MaxSATSearchSpace(n_vars=8, seed=42)
        optima = enumerate_local_optima(space)
        assert len(optima) >= 1
        result = build_otg(space, optima)
        assert len(result.edges) == len(optima)
