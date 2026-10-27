"""The global constants of the hardware model (proposal §5, "spec-sheet first", 2026-10-27).

Everything device-specific comes from vendor documentation (gpus.py, cpus.py). Measurements set only these few
constants, each shared by every device it applies to, never tuned per device. Each value is re-derived from its
data by yardsticks/fit_globals.py (a test fails if this file and the derivation disagree); yardsticks/
validate_globals.py shows how far every published and own measurement then is from the model.
"""

# --- memory ------------------------------------------------------------------------------------------------------
# Sustained read bandwidth of a DRAM channel / its peak (rate x 8 bytes), all-read traffic. Median of the pure-read
# yardsticks (MLC on 3 Xeon generations, likwid ladders on Cascade Lake and Rome, MLC on Arrow Lake).
DRAM_READ_EFFICIENCY = 0.847

# Idle load-to-use latency beyond the DRAM's own CAS time (controller, fabric, caches missed on the way), ns.
# Median of the MLC idle latencies of 3 Xeon generations.
FIXED_LATENCY_NS = 75.35

# Added latency of a socket-to-socket link, ns. Median over the same 3 Xeon generations (MLC remote latency).
SOCKET_LINK_LATENCY_NS = 54.1

# Sustained read bandwidth of a chiplet-to-I/O-die link / its documented peak (bytes per fabric clock x clock).
# Median over Rome (ladders), Zen 4 and Zen 5 desktop (AIDA64, two sources).
CHIPLET_LINK_EFFICIENCY = 0.884

# --- GPU -----------------------------------------------------------------------------------------------------------
# Sustained SM clock at the board's rated power (TDP), compute-heavy load, / the spec boost clock. GeForce cards run
# above their rated boost when power allows. Median over the compute-heavy points (RTX 3090 at 250/300 W scaled with
# CLOCK_POWER_EXPONENT, RTX 4090 at 450 W).
SUSTAINED_CLOCK_FACTOR = 1.039

# Clock vs power limit below the rated power: f ~ P^(1/3) (dynamic power ~ f V^2, V ~ f above the V/F knee). Fixed
# from physics, not fitted; how well it holds is what the validation shows.
CLOCK_POWER_EXPONENT = 1 / 3

# Memory-heavy load runs at a higher clock than compute-heavy load at the same limit (less power per MHz): the ratio.
# Median over the RTX 3090 pairs at 220 / 250 / 300 W.
MEMORY_LOAD_CLOCK_GAIN = 1.059

# Sustained VRAM bandwidth / peak (bus width x data rate). Not yet measured with a controlled probe (bench question
# gpu-vram); 1.0 until then, and marked unmeasured wherever it is used.
VRAM_EFFICIENCY = 1.0
VRAM_EFFICIENCY_MEASURED = False
