# CPU compute: budgets instead of clocks (decided 2026-10-29)

A detailed clock model needs power, power depends on the workload, and that is the hardware-workload loop the
overhaul avoids. CPU compute is therefore modelled as capacities that work draws from, solved like flows:

- **Per core: cycles.** Capacity = units x clock (the maximum turbo clock the machine reports). Operations draw
  from it per instruction class: an issue budget (instructions per cycle) and per-class limits (execution ports:
  vector ALU, shuffle, integer multiply, FMA, loads), from the vendor's optimisation manuals.
- **Per package: power.** Capacity = the package power limit (the machine's RAPL limit where recorded, else the
  documented TDP). Operations draw energy from it. Energy per operation is documented nowhere: it is a global
  constant per microarchitecture, set once from measurements (the stick's cpu-gemm, one core vs all cores) — the
  rule that measurements set only global constants.
- Concurrent work shares both budgets max-min fairly; whichever runs out first sets the pace. One core alone runs at
  its full clock; many cores are throttled by the package budget (turbo behaviour); memory-bound work barely draws on
  it. The clock the model implies (budget / consumption) is checked against the clocks the stick samples — a check,
  not an input.

## Open point: time

Budgets are static. Real CPUs change limits over time (PL2 for up to tau, 28-56 s, then PL1), and with temperature.
For now one budget per workload duration is chosen and stated. Bringing time into the system is the next problem.
