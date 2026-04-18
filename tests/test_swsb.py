"""Tests for Intel SWSB (Software Scoreboarding) decoder."""

from leo.binary.swsb import decode_swsb_xehpc


class TestSWSBNoAnnotation:
    """Test cases where no SWSB annotation is present."""

    def test_zero_means_no_swsb(self):
        result = decode_swsb_xehpc(0x00)
        assert not result.has_reg_dist
        assert not result.has_sbid
        assert result.raw == 0

    def test_noaccsbset_special(self):
        result = decode_swsb_xehpc(0xF0)
        assert not result.has_reg_dist
        assert not result.has_sbid
        assert result.raw == 0xF0


class TestSWSBRegDist:
    """Test RegDist (register dependency distance) decoding."""

    def test_generic_dist_3(self):
        # 0x00 + 3 = 0x03
        result = decode_swsb_xehpc(0x03)
        assert result.has_reg_dist
        assert result.reg_dist_pipe == "generic"
        assert result.reg_dist_distance == 3
        assert not result.has_sbid

    def test_all_dist_1(self):
        # 0x08 + 1 = 0x09
        result = decode_swsb_xehpc(0x09)
        assert result.has_reg_dist
        assert result.reg_dist_pipe == "all"
        assert result.reg_dist_distance == 1

    def test_float_dist_2(self):
        # 0x10 + 2 = 0x12
        result = decode_swsb_xehpc(0x12)
        assert result.has_reg_dist
        assert result.reg_dist_pipe == "float"
        assert result.reg_dist_distance == 2

    def test_int_dist_5(self):
        # 0x18 + 5 = 0x1D
        result = decode_swsb_xehpc(0x1D)
        assert result.has_reg_dist
        assert result.reg_dist_pipe == "int"
        assert result.reg_dist_distance == 5

    def test_long_dist_4(self):
        # 0x20 + 4 = 0x24
        result = decode_swsb_xehpc(0x24)
        assert result.has_reg_dist
        assert result.reg_dist_pipe == "long"
        assert result.reg_dist_distance == 4

    def test_math_dist_7(self):
        # 0x28 + 7 = 0x2F
        result = decode_swsb_xehpc(0x2F)
        assert result.has_reg_dist
        assert result.reg_dist_pipe == "math"
        assert result.reg_dist_distance == 7

    def test_all_pipes_distance_1(self):
        """All 6 pipe types should decode correctly with distance 1."""
        expected = {
            0x01: "generic",
            0x09: "all",
            0x11: "float",
            0x19: "int",
            0x21: "long",
            0x29: "math",
        }
        for raw_val, pipe_name in expected.items():
            result = decode_swsb_xehpc(raw_val)
            assert result.has_reg_dist, f"Failed for {pipe_name} (0x{raw_val:02X})"
            assert result.reg_dist_pipe == pipe_name
            assert result.reg_dist_distance == 1

    def test_max_distance_7(self):
        """Distance 7 (max) should decode for all pipes."""
        for base in (0x00, 0x08, 0x10, 0x18, 0x20, 0x28):
            result = decode_swsb_xehpc(base + 7)
            assert result.has_reg_dist
            assert result.reg_dist_distance == 7

    def test_distance_zero_is_not_regdist(self):
        """A pipe base with distance 0 should NOT decode as RegDist."""
        for base in (0x08, 0x10, 0x18, 0x20, 0x28):
            result = decode_swsb_xehpc(base)
            assert not result.has_reg_dist


class TestSWSBSBID:
    """Test SBID (scoreboard token) decoding."""

    def test_dst_wait_sbid_5(self):
        # 0x80 + 5 = 0x85
        result = decode_swsb_xehpc(0x85)
        assert result.has_sbid
        assert result.sbid_type == "dst_wait"
        assert result.sbid == 5
        assert not result.has_reg_dist

    def test_src_wait_sbid_10(self):
        # 0xA0 + 10 = 0xAA
        result = decode_swsb_xehpc(0xAA)
        assert result.has_sbid
        assert result.sbid_type == "src_wait"
        assert result.sbid == 10

    def test_set_sbid_0(self):
        # 0xC0 + 0 = 0xC0
        result = decode_swsb_xehpc(0xC0)
        assert result.has_sbid
        assert result.sbid_type == "set"
        assert result.sbid == 0

    def test_set_sbid_31(self):
        # 0xC0 + 31 = 0xDF
        result = decode_swsb_xehpc(0xDF)
        assert result.has_sbid
        assert result.sbid_type == "set"
        assert result.sbid == 31

    def test_dst_wait_sbid_0(self):
        # 0x80 + 0 = 0x80
        result = decode_swsb_xehpc(0x80)
        assert result.has_sbid
        assert result.sbid_type == "dst_wait"
        assert result.sbid == 0


class TestSWSBRawPreserved:
    """Test that raw value is always preserved."""

    def test_raw_preserved_for_regdist(self):
        result = decode_swsb_xehpc(0x12)
        assert result.raw == 0x12

    def test_raw_preserved_for_sbid(self):
        result = decode_swsb_xehpc(0x85)
        assert result.raw == 0x85

    def test_raw_preserved_for_unknown(self):
        # 0x30-0x7F range is not clearly defined for FourDistPipe
        result = decode_swsb_xehpc(0x35)
        assert result.raw == 0x35
        assert not result.has_reg_dist
        assert not result.has_sbid
