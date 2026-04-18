"""Tests for PerturbedArchitecture wrapper."""

import pytest

from leo.arch import get_architecture, PerturbedArchitecture
from leo.arch.base import GPUArchitecture


class TestPerturbedArchitecture:
    """Test latency scaling wrapper across all three vendors."""

    @pytest.fixture(params=["a100", "mi300", "pvc"])
    def base_arch(self, request):
        return get_architecture(request.param)

    def test_isinstance_gpu_architecture(self, base_arch):
        """PerturbedArchitecture must pass isinstance check."""
        perturbed = PerturbedArchitecture(base_arch, latency_scale=1.5)
        assert isinstance(perturbed, GPUArchitecture)

    def test_latency_scaling_up(self, base_arch):
        """Doubling scale should roughly double latency values."""
        perturbed = PerturbedArchitecture(base_arch, latency_scale=2.0)
        # Use a vendor-appropriate opcode
        opcode = {"nvidia": "FADD", "amd": "v_add_f32", "intel": "add"}[base_arch.vendor]
        base_min, base_max = base_arch.latency(opcode)
        pert_min, pert_max = perturbed.latency(opcode)
        assert pert_min == max(1, round(base_min * 2.0))
        assert pert_max == max(1, round(base_max * 2.0))

    def test_latency_scaling_down(self, base_arch):
        """Halving scale should roughly halve latency values."""
        perturbed = PerturbedArchitecture(base_arch, latency_scale=0.5)
        opcode = {"nvidia": "FADD", "amd": "v_add_f32", "intel": "add"}[base_arch.vendor]
        base_min, base_max = base_arch.latency(opcode)
        pert_min, pert_max = perturbed.latency(opcode)
        assert pert_min == max(1, round(base_min * 0.5))
        assert pert_max == max(1, round(base_max * 0.5))

    def test_latency_floor_at_one(self, base_arch):
        """Extreme downscaling should never produce 0-cycle latency."""
        perturbed = PerturbedArchitecture(base_arch, latency_scale=0.01)
        opcode = {"nvidia": "FADD", "amd": "v_add_f32", "intel": "add"}[base_arch.vendor]
        pert_min, pert_max = perturbed.latency(opcode)
        assert pert_min >= 1
        assert pert_max >= 1

    def test_identity_scale(self, base_arch):
        """Scale 1.0 should return identical latency values."""
        perturbed = PerturbedArchitecture(base_arch, latency_scale=1.0)
        opcode = {"nvidia": "FADD", "amd": "v_add_f32", "intel": "add"}[base_arch.vendor]
        assert perturbed.latency(opcode) == base_arch.latency(opcode)

    def test_vendor_unchanged(self, base_arch):
        perturbed = PerturbedArchitecture(base_arch, latency_scale=2.0)
        assert perturbed.vendor == base_arch.vendor

    def test_inst_size_unchanged(self, base_arch):
        perturbed = PerturbedArchitecture(base_arch, latency_scale=2.0)
        assert perturbed.inst_size == base_arch.inst_size

    def test_sms_unchanged(self, base_arch):
        perturbed = PerturbedArchitecture(base_arch, latency_scale=2.0)
        assert perturbed.sms == base_arch.sms

    def test_warp_size_unchanged(self, base_arch):
        perturbed = PerturbedArchitecture(base_arch, latency_scale=2.0)
        assert perturbed.warp_size == base_arch.warp_size

    def test_frequency_unchanged(self, base_arch):
        perturbed = PerturbedArchitecture(base_arch, latency_scale=2.0)
        assert perturbed.frequency == base_arch.frequency

    def test_schedulers_unchanged(self, base_arch):
        perturbed = PerturbedArchitecture(base_arch, latency_scale=2.0)
        assert perturbed.schedulers == base_arch.schedulers

    def test_warps_per_sm_unchanged(self, base_arch):
        perturbed = PerturbedArchitecture(base_arch, latency_scale=2.0)
        assert perturbed.warps_per_sm == base_arch.warps_per_sm

    def test_issue_unchanged(self, base_arch):
        """Issue rate should not be affected by latency scaling."""
        perturbed = PerturbedArchitecture(base_arch, latency_scale=2.0)
        opcode = {"nvidia": "FADD", "amd": "v_add_f32", "intel": "add"}[base_arch.vendor]
        assert perturbed.issue(opcode) == base_arch.issue(opcode)

    def test_classify_opcode_unchanged(self, base_arch):
        perturbed = PerturbedArchitecture(base_arch, latency_scale=2.0)
        opcode = {"nvidia": "FADD", "amd": "v_add_f32", "intel": "add"}[base_arch.vendor]
        assert perturbed.classify_opcode(opcode) == base_arch.classify_opcode(opcode)

    def test_memory_prefixes_unchanged(self, base_arch):
        perturbed = PerturbedArchitecture(base_arch, latency_scale=2.0)
        assert perturbed.memory_prefixes == base_arch.memory_prefixes

    def test_sync_prefixes_unchanged(self, base_arch):
        perturbed = PerturbedArchitecture(base_arch, latency_scale=2.0)
        assert perturbed.sync_prefixes == base_arch.sync_prefixes

    def test_atomic_prefixes_unchanged(self, base_arch):
        perturbed = PerturbedArchitecture(base_arch, latency_scale=2.0)
        assert perturbed.atomic_prefixes == base_arch.atomic_prefixes

    def test_get_memory_type_unchanged(self, base_arch):
        perturbed = PerturbedArchitecture(base_arch, latency_scale=2.0)
        mem_opcode = {
            "nvidia": "LDG.E",
            "amd": "global_load_dword",
            "intel": "send",
        }[base_arch.vendor]
        assert perturbed.get_memory_type(mem_opcode) == base_arch.get_memory_type(mem_opcode)

    def test_is_memory_op_unchanged(self, base_arch):
        perturbed = PerturbedArchitecture(base_arch, latency_scale=2.0)
        mem_opcode = {
            "nvidia": "LDG.E",
            "amd": "global_load_dword",
            "intel": "send",
        }[base_arch.vendor]
        assert perturbed.is_memory_op(mem_opcode) == base_arch.is_memory_op(mem_opcode)

    def test_has_static_stall_field_unchanged(self, base_arch):
        perturbed = PerturbedArchitecture(base_arch, latency_scale=2.0)
        assert perturbed.has_static_stall_field == base_arch.has_static_stall_field

    def test_name_includes_scale(self, base_arch):
        perturbed = PerturbedArchitecture(base_arch, latency_scale=1.50)
        assert "1.50" in perturbed.name
        assert base_arch.name in perturbed.name

    def test_memory_latency_scales(self, base_arch):
        """Memory latency should also scale correctly."""
        perturbed = PerturbedArchitecture(base_arch, latency_scale=1.5)
        mem_opcode = {
            "nvidia": "LDG.E",
            "amd": "global_load_dword",
            "intel": "send",
        }[base_arch.vendor]
        base_min, base_max = base_arch.latency(mem_opcode)
        pert_min, pert_max = perturbed.latency(mem_opcode)
        assert pert_min == max(1, round(base_min * 1.5))
        assert pert_max == max(1, round(base_max * 1.5))
