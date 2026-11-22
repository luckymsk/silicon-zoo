# ASIC Architecture Compendium — Corpus

Markdown mirror of every entry in the ASIC Architecture Compendium wiki (Space slug: `asic-architecture-bible`). Entries are appended here in the same run they are added to the wiki. Originally titled "ASIC Architecture Bible"; renamed "Compendium" at Sai's request on Nov 22, 2026.

---

## Foundations & techniques

### Out-of-Order Execution & Tomasulo's Algorithm (IBM System/360 Model 91)

**Added: 2026-11-22**

**What it is / why it matters.** The founding idea of modern high-performance CPU design: let hardware reorder instructions at run time so independent work fills the execution units instead of stalling behind a slow operation. Developed by Robert Tomasulo at IBM and first implemented in the floating-point unit of the IBM System/360 Model 91 (paper published January 1967), it is the direct ancestor of the scheduler in every out-of-order core shipping today — from Pentium Pro to Apple Silicon.

**The architecture.** The Model 91 splits execution into separate fixed-point and floating-point areas, and the floating-point area has two pipelined functional units that can run concurrently: an adder and a multiply/divide unit. Around them Tomasulo built three mechanisms:

- **Reservation stations.** Each functional unit gets several input buffers (three on the adder, two on the multiply/divide unit) that each hold one waiting instruction plus its operands. An instruction is issued in order from the floating-point operation stack (FLOS) into a free reservation station even while the unit is busy, so issue never stalls on a busy unit — the station waits for its operands instead.
- **Register renaming by tags.** The machine's four architectural floating-point registers are renamed in hardware. When an instruction issues, each source operand is either copied from the register file (if ready) or replaced by the *tag* — the reservation-station number — of the instruction that will produce it. This eliminates false dependencies (WAR and WAW hazards): two instructions writing the same register no longer serialize, because only the last writer's result ever reaches the register file.
- **The common data bus (CDB).** All results travel on one broadcast bus from the functional units to every reservation station and to the register file. A waiting station watches the bus for its tag, captures the value the moment it appears, and can start executing without ever touching the register file. One bus, one broadcast, every consumer served at once.
- **Issue–execute–writeback lifecycle.** Issue is in order (from FLOS into a reservation station); execution starts whenever a station's operands are all ready, in any order; results are written back on the CDB. Loads and stores are kept in program order relative to each other, and no instruction may begin execution until earlier branches have resolved.

**Key numbers.** IBM System/360 Model 91, announced 1964, first shipped 1967; Tomasulo's paper, "An Efficient Algorithm for Exploiting Multiple Arithmetic Units," IBM Journal of R&D, January 1967. Five reservation stations in the floating-point unit (3 add, 2 multiply/divide). Planning latencies: floating-point add 2 cycles, multiply 6 cycles, divide 18 cycles. Issue window roughly 14 instructions deep — nearly three times the CDC 6600 scoreboard's window of about 5.

**Signature ideas.**
1. **Hardware register renaming** — the tag system that made out-of-order execution safe, now universal in every superscalar core.
2. **Reservation stations** — distributed, per-unit waiting buffers replacing a single central hazard unit; the direct ancestor of the modern scheduler / issue queue.
3. **The common data bus** — result broadcast with tag matching, so a producer's value reaches all consumers in one cycle; the ancestor of today's bypass/forwarding networks.
4. **Dynamic scheduling without compiler help** — the hardware "looks ahead" about eight instructions and optimizes locally, so ordinary compiled code gets the speedup.

**Limits & tradeoffs.** The CDB is a single broadcast resource: only one result per cycle, so it becomes a bottleneck as functional units multiply — modern cores use multiple broadcast buses and clustered bypass networks instead. The original design had no speculation and no precise-exception machinery; instructions execute out of order but commit in order only in later descendants. It covered floating-point only; integer out-of-order execution came later. And it was built from discrete transistor logic (SLT modules), so the five reservation stations and the tag comparators were expensive in board area and power — the idea had to wait for VLSI to become cheap.

**Lineage.** Descended from the CDC 6600 scoreboard (1964), which centralized hazard detection; Tomasulo distributed the control into the reservation stations and added renaming. Revived and generalized in the 1990s: MIPS R10000, HP PA-8000, PowerPC 604/620, DEC Alpha 21264, Intel Pentium Pro (P6). Every modern out-of-order core — Zen, Apple Firestorm/Avalanche, Cortex-X, Golden Cove — is a Tomasulo machine with a reorder buffer, multiple CDBs, and speculation added on top.

**One-line summary.** Tomasulo's algorithm taught hardware to rename registers, wait in reservation stations, and broadcast results on a common bus — the three ideas that let processors execute instructions out of order, and the foundation of every high-performance CPU since.
