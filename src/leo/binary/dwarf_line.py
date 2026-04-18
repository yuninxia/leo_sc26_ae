"""DWARF line information parser for extracting source location with column info.

This module parses DWARF debug information from GPU binaries to extract
address -> (file, line, column) mappings. This enables precise attribution
of instructions to specific expressions within a source line.

The key insight is that AMD's LLVM compiler includes column information in
the DWARF line number program via the DW_LNS_set_column opcode. Tools like
llvm-objdump and readelf's decoded output drop this information, but we can
extract it directly using pyelftools.

Usage:
    parser = DWARFLineParser("kernel.gpubin")
    loc = parser.get_source_location(0x1900)
    # loc -> SourceLocation(address=0x1900, file="...", line=264, column=58)

References:
    - DWARF Standard: http://dwarfstd.org/
    - pyelftools: https://github.com/eliben/pyelftools
"""

import bisect
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from elftools.elf.elffile import ELFFile

logger = logging.getLogger(__name__)


@dataclass
class SourceLocation:
    """Maps instruction address to source location with column information.

    Attributes:
        address: PC/offset in the binary (virtual address for AMD, file offset for NVIDIA)
        file: Full path to source file
        line: Line number (1-indexed, 0 means unknown)
        column: Column number (0 means unknown)
        is_stmt: True if this is the start of a statement
    """

    address: int
    file: str
    line: int
    column: int
    is_stmt: bool = True

    def __str__(self) -> str:
        """Format as file:line:column."""
        if self.column > 0:
            return f"{self.file}:{self.line}:{self.column}"
        return f"{self.file}:{self.line}"

    @property
    def has_column(self) -> bool:
        """Check if column information is available."""
        return self.column > 0

    @property
    def short_file(self) -> str:
        """Get just the filename without directory path."""
        return Path(self.file).name


class DWARFLineParser:
    """Parse DWARF line number tables to extract source locations with column info.

    This parser reads the .debug_line section of ELF binaries and extracts
    the full address-to-source mapping including column information that is
    typically dropped by tools like llvm-objdump.

    The DWARF line number program is a compact state machine that encodes:
    - DW_LNS_set_column: Set the column register
    - DW_LNS_advance_line: Update line number
    - DW_LNE_set_address: Set instruction address
    - Special opcodes: Compact encoding for common patterns

    pyelftools handles all the decoding; we just need to extract the state.
    """

    def __init__(self, binary_path: str):
        """Initialize parser and load DWARF line info from binary.

        Args:
            binary_path: Path to GPU binary (gpubin, cubin, or code object)
        """
        self._path = Path(binary_path)
        self._locations: Dict[int, SourceLocation] = {}
        self._sorted_addresses: List[int] = []
        self._parsed = False
        self._has_dwarf = False

        # Parse immediately
        self._parse()

    def _parse(self) -> None:
        """Parse DWARF line info from binary."""
        if self._parsed:
            return

        self._parsed = True

        try:
            with open(self._path, "rb") as f:
                elf = ELFFile(f)

                if not elf.has_dwarf_info():
                    logger.debug(f"No DWARF info in {self._path}")
                    return

                self._has_dwarf = True
                dwarfinfo = elf.get_dwarf_info()

                # Iterate through all compilation units
                for CU in dwarfinfo.iter_CUs():
                    self._parse_compilation_unit(CU, dwarfinfo)

            # Sort addresses for binary search
            self._sorted_addresses = sorted(self._locations.keys())

            logger.debug(
                f"Parsed {len(self._locations)} source locations from {self._path}"
            )

        except FileNotFoundError:
            logger.warning(f"Binary not found: {self._path}")
        except Exception as e:
            # Intel zebin binaries use GPU-specific ELF relocation types
            # (e.g. R_INTEL_GT_64) that pyelftools doesn't handle.
            # This is harmless — HPCToolkit's database provides source mapping.
            if "relocation" in str(e).lower():
                logger.debug(f"DWARF parsing skipped for {self._path.name}: {e}")
            else:
                logger.warning(f"Error parsing DWARF from {self._path}: {e}")

    def _parse_compilation_unit(self, CU, dwarfinfo) -> None:
        """Parse line program for one compilation unit.

        Args:
            CU: Compilation unit from pyelftools
            dwarfinfo: DWARF info object
        """
        try:
            line_prog = dwarfinfo.line_program_for_CU(CU)
            if line_prog is None:
                return

            # Get directory and file tables from header
            # DWARF 5 uses different attribute names
            header = line_prog.header

            # Handle both DWARF 4 and DWARF 5 directory/file tables
            dwarf_version = header.get("version", 4)
            directories = self._get_directories(header)
            files = self._get_files(header)

            # Iterate through line program entries
            for entry in line_prog.get_entries():
                if entry.state is None:
                    continue

                state = entry.state

                # Skip end_sequence markers (they reset state)
                if state.end_sequence:
                    continue

                address = state.address
                file_idx = state.file
                line = state.line
                column = state.column
                is_stmt = state.is_stmt

                # Resolve file path
                file_path = self._resolve_file_path(
                    file_idx, files, directories, dwarf_version
                )

                # Create location entry
                self._locations[address] = SourceLocation(
                    address=address,
                    file=file_path,
                    line=line,
                    column=column,
                    is_stmt=is_stmt,
                )

        except Exception as e:
            logger.debug(f"Error parsing CU line program: {e}")

    def _get_directories(self, header) -> List[str]:
        """Extract directory table from line program header.

        Handles both DWARF 4 and DWARF 5 formats.
        """
        directories = []

        # Try DWARF 5 format first (include_directories is a list of entries)
        if hasattr(header, "include_directory"):
            # DWARF 5: include_directory is list of DWARFv5 entries
            for entry in header.include_directory:
                if isinstance(entry, bytes):
                    directories.append(entry.decode("utf-8", errors="replace"))
                elif hasattr(entry, "DW_LNCT_path"):
                    directories.append(
                        entry.DW_LNCT_path.decode("utf-8", errors="replace")
                    )
                else:
                    directories.append(str(entry))
        elif hasattr(header, "include_directories") and header.include_directories:
            # DWARF 4: include_directories is simple list
            for d in header.include_directories:
                if isinstance(d, bytes):
                    directories.append(d.decode("utf-8", errors="replace"))
                else:
                    directories.append(str(d))

        return directories

    def _get_files(self, header) -> List[Tuple[str, int]]:
        """Extract file table from line program header.

        Returns list of (filename, dir_index) tuples.
        Handles both DWARF 4 and DWARF 5 formats.
        """
        files = []

        # Try DWARF 5 format first
        if hasattr(header, "file_entry"):
            for entry in header.file_entry:
                if hasattr(entry, "name"):
                    name = entry.name
                    if isinstance(name, bytes):
                        name = name.decode("utf-8", errors="replace")
                    dir_idx = getattr(entry, "dir_index", 0)
                    files.append((str(name), dir_idx))
                elif hasattr(entry, "DW_LNCT_path"):
                    name = entry.DW_LNCT_path
                    if isinstance(name, bytes):
                        name = name.decode("utf-8", errors="replace")
                    dir_idx = getattr(entry, "DW_LNCT_directory_index", 0)
                    files.append((str(name), dir_idx))

        return files

    def _resolve_file_path(
        self,
        file_idx: int,
        files: List[Tuple[str, int]],
        directories: List[str],
        dwarf_version: int = 4,
    ) -> str:
        """Resolve full file path from file index.

        Args:
            file_idx: File index from line state
            files: List of (filename, dir_index) tuples
            directories: List of directory paths
            dwarf_version: DWARF version (4 = 1-indexed files/dirs, 5 = 0-indexed)

        Returns:
            Full file path or just filename if directory not found
        """
        # DWARF 4: file indices are 1-indexed, directory indices are 1-indexed
        # DWARF 5: file indices are 0-indexed, directory indices are 0-indexed
        if dwarf_version >= 5:
            # DWARF 5: file_idx is already 0-indexed
            idx = file_idx
        else:
            # DWARF 4: file_idx is 1-indexed, convert to 0-indexed
            idx = file_idx - 1

        if idx < 0 or idx >= len(files):
            return f"<unknown file {file_idx}>"

        filename, dir_idx = files[idx]

        # Resolve directory
        if dwarf_version >= 5:
            # DWARF 5: dir_idx is 0-indexed
            if 0 <= dir_idx < len(directories):
                dir_path = directories[dir_idx]
                return f"{dir_path}/{filename}"
        else:
            # DWARF 4: dir_idx is 1-indexed (0 = compilation directory)
            if dir_idx > 0 and dir_idx <= len(directories):
                dir_path = directories[dir_idx - 1]
                return f"{dir_path}/{filename}"

        return filename

    @property
    def has_dwarf_info(self) -> bool:
        """Check if binary has DWARF debug information."""
        return self._has_dwarf

    @property
    def location_count(self) -> int:
        """Number of address-to-source mappings."""
        return len(self._locations)

    def get_source_location(self, address: int) -> Optional[SourceLocation]:
        """Get source location for an instruction address.

        If exact address is not found, returns the location for the nearest
        preceding address (as instructions may span multiple bytes).

        Args:
            address: Instruction address (PC)

        Returns:
            SourceLocation or None if no mapping found
        """
        # Exact match
        if address in self._locations:
            return self._locations[address]

        # Binary search for nearest preceding address
        if not self._sorted_addresses:
            return None

        idx = bisect.bisect_right(self._sorted_addresses, address) - 1
        if idx >= 0:
            nearest_addr = self._sorted_addresses[idx]
            return self._locations[nearest_addr]

        return None

    def get_all_locations(self) -> Dict[int, SourceLocation]:
        """Get all address-to-source mappings.

        Returns:
            Dictionary mapping address -> SourceLocation
        """
        return dict(self._locations)

    def get_locations_for_file(self, filename: str) -> List[SourceLocation]:
        """Get all locations for a specific source file.

        Args:
            filename: Filename to filter by (can be partial match)

        Returns:
            List of SourceLocation for matching file
        """
        return [loc for loc in self._locations.values() if filename in loc.file]

    def get_locations_for_line(self, filename: str, line: int) -> List[SourceLocation]:
        """Get all locations for a specific source line.

        Useful for finding all instructions generated from one line.

        Args:
            filename: Source filename (partial match)
            line: Line number

        Returns:
            List of SourceLocation for that line, sorted by address
        """
        matches = [
            loc
            for loc in self._locations.values()
            if filename in loc.file and loc.line == line
        ]
        return sorted(matches, key=lambda x: x.address)

    def get_column_statistics(self) -> Dict[str, int]:
        """Get statistics about column information availability.

        Returns:
            Dictionary with 'total', 'with_column', 'without_column' counts
        """
        total = len(self._locations)
        with_column = sum(1 for loc in self._locations.values() if loc.column > 0)
        return {
            "total": total,
            "with_column": with_column,
            "without_column": total - with_column,
            "column_coverage_pct": (with_column / total * 100) if total > 0 else 0,
        }

    def dump_line_table(self, max_entries: int = 0) -> str:
        """Dump line table in human-readable format.

        Args:
            max_entries: Maximum entries to dump (0 = all)

        Returns:
            Formatted string with address -> source mapping
        """
        lines = [f"DWARF Line Table for {self._path.name}"]
        lines.append(f"Total entries: {len(self._locations)}")
        lines.append("-" * 80)
        lines.append(f"{'Address':>12}  {'Line':>6}  {'Col':>4}  File")
        lines.append("-" * 80)

        entries = sorted(self._locations.items())
        if max_entries > 0:
            entries = entries[:max_entries]

        for addr, loc in entries:
            col_str = str(loc.column) if loc.column > 0 else "-"
            lines.append(f"{addr:#12x}  {loc.line:>6}  {col_str:>4}  {loc.short_file}")

        return "\n".join(lines)


def get_dwarf_line_parser(binary_path: str) -> Optional[DWARFLineParser]:
    """Factory function to get DWARF line parser for a binary.

    Returns None if parsing fails or binary has no DWARF info.

    Args:
        binary_path: Path to GPU binary

    Returns:
        DWARFLineParser instance or None
    """
    try:
        parser = DWARFLineParser(binary_path)
        if parser.has_dwarf_info:
            return parser
        return None
    except Exception as e:
        logger.debug(f"Could not create DWARF parser for {binary_path}: {e}")
        return None
