"""Link capacities from specifications."""
from __future__ import annotations

# PCIe: transfer rate per lane (GT/s) and line encoding, per generation (PCI-SIG specifications)
PCIE = {1: (2.5, 8 / 10), 2: (5.0, 8 / 10), 3: (8.0, 128 / 130), 4: (16.0, 128 / 130), 5: (32.0, 128 / 130),
        6: (64.0, 242 / 256)}


def pcie_bps(gen: int, lanes: int) -> float:
    """Raw payload capacity per direction (before protocol overhead: TLP headers, flow control)."""
    gts, enc = PCIE[gen]
    return gts * 1e9 * enc / 8 * lanes
