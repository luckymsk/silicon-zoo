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

---

## AI accelerators

### Google TPU v5p

**Added: 2026-11-22** (deepened: Building blocks breakdown added to the wiki entry; the wiki card existed at launch)

**What it is / why it matters.** Google's performance-tier fifth-generation TPU, announced 6 December 2023 and built for Gemini-era training: the large-scale training chip to v5e's cost-efficient sibling. It is the generation where Google's bet became explicit — the pod, not the chip, is the machine. A full v5p pod wires 8,960 chips into one 3D torus, and the chip itself was redesigned so that one device serves the whole large model, dense layers and embeddings alike.

**Building blocks.** Seven named pieces, and the connections between them are the design:
- **TensorCore ×2** — each of the two compute cores on the chip: four 128×128 systolic MXUs, one vector unit and one scalar unit — eight MXUs per chip, twice v5e's one TensorCore.
- **MXU — systolic array** — a 128×128 grid of MACs; weights sit still, activations flow through, each partial sum passes to a neighbour — BF16 multiplies with FP32 accumulation.
- **Vector unit** — activations, normalisation, softmax and everything that isn't a dense matmul, one per TensorCore.
- **Scalar unit** — a small control core sequencing the MXU and vector unit and the XLA-generated schedule.
- **SparseCore ×4 (2nd gen)** — dataflow processors for embedding gather/scatter — the irregular, sparse half of recommendation and large-vocabulary models.
- **95 GiB HBM + ICI links** — HBM at 2,765 GB/s per chip, plus Inter-Chip Interconnect wiring chips into a 3D torus — 1,200 GB/s bidirectional per chip.
- **Optical circuit switch** — reconfigurable light paths rewire the pod topology per job without moving a cable — inherited from v4, now across 8,960 chips.

**The piece that makes it special:** the dense-plus-sparse pairing at pod scale. v4 proved the systolic MXU; v5p doubles it to two TensorCores per chip and pairs it with four second-generation SparseCores, so one chip serves the whole large model — dense transformer layers on the MXUs, embedding and vocabulary layers on the SparseCores — while the 3D torus and optical switches scale that pairing to 8,960 chips. The bet is that the pod, not the chip, is the machine: 2× the FLOPS and 3× the HBM bandwidth of v4, with the interconnect to keep 6,144-chip jobs fed.

**The architecture.** Work arrives as an XLA-compiled graph, tiled to the 128×128 MXU shape and sharded across the pod before the first chip runs. A weight tile enters an MXU and stays stationary in the PEs while activations systolic-flow past it, partial sums accumulating PE to PE. Embedding lookups and sparse gathers run on the four SparseCores while the MXUs stay on dense matmul; softmax, layernorm and residuals run on the vector units while the MXU starts the next tile. Between chips, ICI plus optical circuit switches move activations across the 3D torus as if the pod were one accelerator. Pods are built from 64-chip cubes (4×4×4 across 16 hosts); only a full cube or larger gets genuine 3D-torus wraparound, and ICI resiliency reroutes optical links around faults. Each 4-chip host VM carries 208 vCPUs and 448 GB of RAM.

**Key numbers.** Announced 6 December 2023; 459 TFLOPS BF16 and FP8 per chip, 918 TOPS INT8; 95 GiB HBM at 2,765 GB/s per chip; 2 TensorCores (8 MXUs) and 4 SparseCores per chip; 1,200 GB/s bidirectional ICI per chip; 3D torus; 8,960 chips per pod (~4.1 exaFLOPS BF16), largest single job 6,144 chips. Google reports v5p trains large LLM models 2.8× faster than v4, and embedding-dense models 1.9× faster via the second-generation SparseCores. (Process node and HBM generation are not disclosed by Google and are omitted.)

**Signature ideas.**
1. **Two TensorCores per chip** — doubled dense compute over v5e, with the MXU count to match.
2. **Second-generation SparseCores** — sparse embedding acceleration on the same die as the dense matmul engines.
3. **8,960-chip 3D-torus pods with optical rewiring** — topology reshaped per job by optical circuit switches.
4. **XLA compiler co-design** — the chip, the compiler and the data-centre topology designed as one artefact.

**Limits & tradeoffs.** Flexibility and access. Control flow, dynamic shapes and small models underuse the array, and the whole stack assumes XLA/JAX-style compilation — peak efficiency demands the workload be re-expressed as large, regular matmuls. And the pod is a cloud machine, not a part you buy: you rent slices (up to 6,144 chips per job), so the architecture's scale is only available inside Google's data centres.

**Lineage.** Descended from TPU v4 (2021), which introduced the 3D torus, optical circuit switching and the first SparseCores; v5p doubles the TensorCores, triples HBM bandwidth and more than doubles pod size. Its sibling v5e (2023) is the cost-efficient single-TensorCore variant for inference and smaller training. Succeeded by Trillium / TPU v6e (2024), which doubles per-chip BF16 compute again to 918 TFLOPS.

**One-line summary.** TPU v5p pairs two systolic TensorCores with four SparseCores on one chip and scales the pair to 8,960-chip optically rewired pods — Google's statement that for frontier training, the pod is the machine.

---

## GPUs

### Intel Ponte Vecchio (Xe-HPC)

**Added: 2026-11-22** (deepened: Building blocks breakdown added to the wiki entry)

**What it is / why it matters.** Intel's maximalist HPC and AI GPU, sold as the Intel Data Center GPU Max series and built as the compute engine of the Aurora exascale supercomputer at Argonne National Laboratory. Disclosed in 2021–22 and launched in January 2023, it is less notable for any single compute unit than for how it is put together: 47 functional tiles on three different process nodes, stacked with Foveros 3D packaging and stitched with EMIB bridges — the most heterogeneous package ever shipped in a GPU, and a statement that architecture now lives as much in the packaging as in the ALU.

**Building blocks.** Six named pieces, and the connections between them are the design:
- **Compute tile (TSMC N5)** — the compute building block: 8 Xe-cores plus 4 MB of L1 per tile, stacked by Foveros 3D onto the base tile. Sixteen of them make the 128 Xe-cores of a full OAM module.
- **Xe-core** — each core packs 16 vector engines (XVE, 512-bit, native FP64/FP32) and 16 matrix engines (XMX, 4096-bit) plus one ray-tracing unit, for 128 RT units across the package.
- **XMX matrix engine** — the AI FLOPS engine: 2048 TF32 / 4096 FP16 / 4096 BF16 operations per clock per Xe-core, for 839 TFLOPS of BF16/FP16 across the full OAM.
- **Rambo cache tile** — a dedicated SRAM tile holding a slice of the 408 MB L2 close to the compute tiles, so the working set rarely has to leave the package for HBM.
- **Base tile (Intel 7, Foveros)** — the package's nervous system: 144 MB of L2 per base tile, the HBM2e controllers, the PCIe Gen5 host interface, and the Modular Die Fabric (MDFI) that routes traffic between tiles.
- **Xe Link tile (TSMC N7)** — scale-out: 8 Xe Links per tile through an 8-port embedded switch on up to 90G SerDes, wiring up to 8 OAMs into one fabric.

**The piece that makes it special:** the package itself. The compute is conventional for its class; what is radical is assembling 47 tiles on three process nodes (TSMC N5 compute, Intel 7 base, TSMC N7 link) with Foveros 3D stacking and EMIB bridges, totalling over 100 billion transistors. The base tile is the trick: it carries the L2, the HBM controllers and the die-to-die fabric, so the compute tiles stay small, high-yielding and swappable.

**The architecture.** Work arrives from the host over PCIe Gen5; the base tile's host interface hands kernels to the compute tiles through the MDFI fabric. Each Xe-core runs vector work on its 16 XVEs and matrix work on its 16 XMX engines, with operands held in the 64 MB of L1 across the package. The 408 MB L2 — Rambo tiles plus the base tiles' 144 MB each — absorbs reuse before traffic reaches HBM. Eight HBM2e stacks feed the package with up to 128 GB at about 3.2 TB/s in the Max 1550 configuration. Multi-GPU collectives leave through the Xe Link tiles' 8-port switch, so up to 8 OAMs act as one fabric. Three SKUs shipped: Max 1100 (56 Xe-cores, 48 GB HBM2e, 300 W PCIe), Max 1350 (112 Xe-cores, 96 GB, 450 W OAM) and Max 1550 (128 Xe-cores, 128 GB, 600 W OAM).

**Key numbers.** 47 functional tiles (63 including thermal tiles), over 100 billion transistors; 128 Xe-cores, 1024 vector engines and 1024 matrix engines; 52 TFLOPS FP64/FP32, 419 TFLOPS TF32, 839 TFLOPS BF16/FP16, 1678 TFLOPS INT8 (two-stack OAM, Hot Chips 34); 64 MB L1, 408 MB L2, up to 128 GB HBM2e at ~3.2 TB/s; package 77.5 × 62.5 mm; 600 W (Max 1550 OAM).

**Signature ideas.**
1. **Foveros 3D + EMIB heterogeneous integration** — three process nodes in one package, compute stacked on the base tile, bridges between tiles.
2. **Rambo cache tiles** — a dedicated SRAM die as a first-class tile, not cache bolted onto the compute die.
3. **XMX matrix engines with native FP64 vector** — one core that does both HPC's FP64 and AI's low-precision matrix math.
4. **Xe Link scale-out fabric** — an embedded switch tile turning up to 8 OAMs into one fabric for Aurora.

**Limits & tradeoffs.** Yield, power and software. Forty-seven tiles mean 47 dice to test, bin and assemble; the 600 W Max 1550 OAM needs liquid cooling in practice, and the part shipped late — Aurora's schedule slipped with it. The Xe-HPC software stack (oneAPI/SYCL) had to be built from scratch against CUDA's incumbency, and Intel has since narrowed its discrete data-center GPU ambitions. Ponte Vecchio is a packaging landmark more than a market winner.

**Lineage.** Descended from Intel's Xe family (Xe-LP in Tiger Lake, Xe-HPG in Arc Alchemist) scaled up to the Xe-HPC branch for data centers. Its packaging approach — Foveros 3D stacking plus EMIB bridges, with cache and I/O as separate tiles — carried into Intel's later data-center parts and into the industry's chiplet mainstream. It remains the GPU of the Aurora exascale system.

**One-line summary.** Ponte Vecchio bet that the future of GPUs is packaging: 47 tiles on three process nodes, stacked and bridged into one 100-billion-transistor device whose cleverest block is the base tile holding it all together.


## Drill-down completion — all remaining entries deepened (2026-11-22)

The final 36 catalogue entries that still carried generic class-pattern breakdowns were replaced with hand-written, chip-specific breakdowns (DEEP map 61 → 97 of 97; zero entries remain at class-pattern level). Each entry below mirrors the wiki deep dive: named blocks of that specific design, the starred special piece, the accepted trade-off, and a fact check under Sai's standard (source: primary where possible; quote: the exact sentence; status: verified / corrected / disputed / unverifiable). One JSON record per fact is stored in files/fact-records.json.

### Tesla Dojo D1

**Specs (wiki card).** 7 nm · 50B transistors · 354 training nodes per die · 25 dies per training tile

**Building blocks.**
- **Training node ×354** — The repeating unit of the die: a small programmable core with its own scalar unit, a 64-byte-wide SIMD vector/matrix unit (8×8×4 matrix multiply), and 4-way SMT — 354 of them tile the 645 mm² die in a 2D array
- **1.25 MB SRAM per node** — Each node's private, software-managed scratchpad — 440 MB across the die. There are deliberately no caches, no virtual memory and no coherence: software moves every byte explicitly
- **2D mesh NoC** — Nearest-neighbour mesh wiring the nodes on-die, extended off-die by high-speed SerDes on all four edges so dies butt-join into a 5×5 training tile with no switch chip in between
- **Configurable FP8 / BF16 datapaths** — Tesla's own CFP8/BF16 formats sized for video training — the node's vector unit does 64-byte SIMD in these formats, reaching 362–376 TFLOPS per die (see Fact check — the two published figures disagree)
- **Dojo Interface Processor (DIP)** — I/O dies at the training-tile edges holding the shared DRAM pool (160 GB per tile) and host/Ethernet interfaces, so the D1 die itself carries no DRAM controllers
- **TTPoE transport** — Tesla Transport Protocol over Ethernet — Tesla's custom, trimmed Ethernet protocol for tile-to-tile traffic, later open-sourced and taken to the Ultra Ethernet Consortium

**The piece that makes it special.** The node, not the array. Where a TPU bets on one giant systolic array, D1 bets on 354 small CPU-like training nodes, each with private SRAM and SMT, joined by a mesh that extends seamlessly across 25 dies into a training tile and 3,000 dies into an ExaPOD. The special piece is the seam — or its absence: the same mesh programming model runs on one node, one die, or a cabinet, because the die edges are just more mesh.

**How work flows.** Compiler places the model: Layers are mapped onto nodes; weights and activations are written into per-node SRAM explicitly — there is no cache to hide a bad placement. Nodes compute locally: Each node runs its scalar schedule and fires its SIMD/matrix unit on the tile of the network it owns. Activations cross the mesh: Results move node-to-node on-die, then die-to-die across tile edges at multi-TB/s aggregate bandwidth. DIPs feed the tile: Interface processors stream the next batch from the tile's shared DRAM pool and handle host traffic. ExaPOD scales the pattern: Tiles rack into cabinets; TTPoE carries inter-tile traffic over Ethernet with the same send/receive model.

**Trade-off accepted.** Programmability for peak. A mesh of small cores only reaches its FLOPS when the compiler and kernels keep 354 nodes fed with balanced work, and the formats (CFP8) and stack are Tesla-only. Tesla publishes no datasheet for D1 — every number here traces to its Hot Chips 34 disclosure and AI Day, and the programme itself was wound down in 2025 in favour of the AI4/AI5 inference line.

**Fact check.**
- [verified] 354 training nodes per D1 die; 440 MB SRAM per die — Source: Tesla, Hot Chips 34 (2022), Dojo microarchitecture talk. Quote: "Each chip has 354 DOJO processing nodes per die and 440MB of SRAM."
- [corrected] Spec line previously read “354 training tiles per system tile” — Source: Tesla, Hot Chips 34 (2022). Was: 354 training tiles per system tile; now: 354 training nodes per D1 die — 25 dies form one training tile. Quote: "Each node has 1.25MB of SRAM."
- [disputed] Peak BF16/CFP8 throughput per die — Source: Tesla Hot Chips 34 vs later Tesla-derived summaries. Note: 362 TFLOPS and 376 TFLOPS both circulate from Tesla-derived material; the wiki states the range instead of picking one.
- [unverifiable] 50 billion transistors — Source: Secondary reporting of Tesla AI Day 2021. Note: Widely reported, but not stated in the Hot Chips 34 deck.

### AWS Inferentia2 & Trainium

**Specs (wiki card).** NeuronCore-v2 · NeuronLink interconnect

**Building blocks.**
- **NeuronCore-v2 ×2** — The whole chip is two identical cores; each is a self-contained heterogeneous accelerator with its own sequencers, SRAMs and DMA — Trainium and Inferentia2 share the same core, differing in links and clocks
- **Tensor Engine** — A power-optimised systolic array for GEMM, convolution and transpose — over 90 BF16 TFLOPS per core, FP32 accumulation, with cFP8/TF32 support
- **Vector Engine** — Wide vector unit for reductions, normalisation, pooling and activations — the non-matmul residue of a transformer layer
- **Scalar Engine** — Element-wise ops and control: biases, ReLU, index math — small, but it keeps the tensor engine from stalling on glue work
- **GpSimd Engine** — Eight fully programmable 512-bit SIMD processors per core running ordinary C code against the on-chip SRAM — the escape hatch for operators the fixed engines can't express
- **SBUF + PSUM SRAMs** — Two software-managed scratchpads, no caches anywhere: a 24 MiB state buffer (SBUF) staging operands and a 2 MiB partial-sum buffer (PSUM) dedicated to tensor-engine accumulation
- **CC cores + NeuronLink-v2** — Six collective-communication cores and NeuronLink-v2 device links run all-reduce and friends off the compute cores' critical path

**The piece that makes it special.** The GpSimd engine. Every other accelerator in this class forces odd operators onto a host CPU or a slow path; NeuronCore-v2 embeds eight programmable 512-bit cores inside the accelerator itself, programmed in C through the Neuron Kernel Interface, so a custom operator runs where the data already lives. The four engines run four independent instruction streams in parallel, synchronised by hardware semaphores — a tiny heterogeneous SoC per core.

**How work flows.** Neuron compiler tiles the graph: The SDK lowers the model to the four engines, tiling tensors to the 128-partition SBUF layout before anything runs. DMA stages SBUF: 32 DMA engines move weights and activations from the 32 GiB HBM into SBUF; nothing is cached or fetched on demand. Tensor engine sweeps: GEMMs run systolically, partial sums accumulating in PSUM at FP32. Vector/scalar finish the layer: Softmax, norms and element-wise ops run on the vector and scalar engines; exotic ops spill to GpSimd — all without leaving the core. Collectives overlap: CC cores and NeuronLink move gradients/activations to peer chips while the next tile computes.

**Trade-off accepted.** Software burden. With no caches and four asynchronous engines, performance lives or dies in the compiler and in explicit synchronisation; models that don't tile to the SBUF's partition geometry leave the systolic array idle. You also buy into one cloud: the chips exist only inside AWS instances.

**Fact check.**
- [verified] NeuronCore-v2 has four heterogeneous compute engines (Tensor, Vector, Scalar, GpSimd) — Source: AWS Neuron documentation — Trainium/Inferentia2 Architecture Guide for NKI. Quote: "A NeuronCore-v2 consists of four heterogeneous compute engines (Tensor, Vector, Scalar, and GpSimd), each designed to accelerate different types of operators in modern machine learning models."
- [verified] 24 MiB SBUF + 2 MiB PSUM, software-managed — Source: AWS Neuron documentation — Trainium/Inferentia2 Architecture Guide for NKI. Quote: "Memory-wise, a NeuronCore-v2 consists of two software-managed on-chip SRAMs, a 24MiB SBUF as the main data storage and a 2MiB PSUM as a dedicated accumulation buffer for Tensor Engine."
- [verified] Two NeuronCore-v2 per device; 32 GiB HBM at 820 GB/s — Source: AWS Neuron documentation — Trainium/Inferentia2 Architecture Guide for NKI. Quote: "2 NeuronCores (v2). 2 HBM stacks with a total device memory capacity of 32GiB and bandwidth of 820 GB/s."

### SambaNova SN40L

**Specs (wiki card).** Reconfigurable Dataflow Unit · three-tier memory (SRAM/DRAM/HBM class)

**Building blocks.**
- **Pattern Compute Unit (PCU) ×1,040** — A multi-stage reconfigurable SIMD pipeline; each PCU is configured at compile time to execute one innermost loop body — configurable as systolic array or vector unit, BF16/FP32/INT32/INT8, with a cross-lane reduction tree
- **Pattern Memory Unit (PMU) ×1,040** — Distributed scratchpad SRAM — 520 MB in total — programmer-managed, fragmentable, with its own address-generation pipeline (4 addresses/cycle) and alignment crossbars for transpose and layout changes in flight
- **Mesh switches (3 networks)** — Separate packet-switched networks for vector data, scalar addresses/metadata and control, wiring PCUs and PMUs into a spatial fabric — operators pass tensors directly, never via memory
- **AGCU** — Address Generation and Coalescing Unit — the portal off-chip: generates and coalesces accesses to HBM, DDR and peer RDUs, including sparse and graph-shaped patterns
- **HBM tier — 64 GB** — Co-packaged HBM as the middle tier: a software-managed cache between SRAM and the DDR reservoir
- **DDR tier — up to 1.5 TB** — Pluggable DRAM attached to the accelerator itself, not the host — the tier that lets a socket hold trillion-parameter models and swap experts in milliseconds

**The piece that makes it special.** The three-tier memory, attached to the accelerator. GPUs force a choice: fit in HBM or stream from host memory over PCIe. SN40L stacks 520 MB of SRAM, 64 GB of HBM and up to 1.5 TB of DDR on the RDU itself, and the SambaFlow compiler places every tensor in a tier deliberately. That is what makes its party trick — a Composition-of-Experts serving 150+ models from one rack — possible: experts live in DDR and are promoted through HBM into SRAM-resident dataflow graphs.

**How work flows.** SambaFlow maps the whole graph: The compiler spatially assigns every operator to PCUs, every tensor to PMUs, and every route to the mesh — the binary (PEF) is the schedule. Weights stream up the tiers: The active model's weights move DDR → HBM → PMU SRAM ahead of the tokens that need them. Data flows spatially: Activations stream PCU to PCU through the mesh; fused operators never round-trip to memory between layers. AGCU handles the irregular: Sparse gathers and off-chip accesses are generated and coalesced at the fabric edge. Model switches are memory moves: Serving a different expert is a tier promotion, not a reload from a host — SambaNova quotes sub-10 ms switches for an 8B model.

**Trade-off accepted.** Compile-time rigidity. A spatial map is optimal for the graph it was compiled for and brittle for dynamic shapes and control flow; the RDU's peak (638 BF16 TFLOPS) is inseparable from SambaFlow's whole-graph compilation, and the ecosystem is a single vendor's rack, not a card you buy.

**Fact check.**
- [verified] 1,040 PCUs and 1,040 PMUs per socket — Source: SambaNova, Hot Chips 2024 — SN40L RDU presentation. Quote: "1040 PCUs and PMUs PCU: Compute unit PMU: Memory unit S: Mesh switches AGCU: Portal to off-chip memory and IO"
- [verified] Three-tier memory: 520 MB SRAM / 64 GB HBM / 1.5 TB DDR; 638 BF16 TFLOPS; 102B transistors, TSMC 5 nm — Source: SambaNova, Hot Chips 2024 — SN40L RDU presentation. Quote: "5nm TSMC 3-tier Dataflow Memory … 520 MB On-Chip Memory, 64 GB High Bandwidth Memory, 1.5 TB High Capacity Memory, 638 TFLOPS (bf16), 102B Transistors."

### Intel Gaudi 2 (Habana)

**Specs (wiki card).** 7 nm · 96 GB HBM2e · 24 TPCs + MME

**Building blocks.**
- **Matrix Multiplication Engine (MME) ×2** — The GEMM engine: a large output-stationary systolic array (256×256 MAC structure) handling everything that lowers to matrix multiplication — fully connected, convolution, batched GEMM — and the two units can merge into one wider engine
- **Tensor Processor Core (TPC) ×24** — Fully programmable VLIW SIMD cores, programmed in TPC-C, for everything the MME can't do — normalisation, softmax, custom operators. This programmability is Gaudi's founding idea: no operator needs a host fallback
- **Shared SRAM — 48 MB** — The meeting point: MME, TPCs, DMAs and NICs all operate against one 48 MB SRAM complex in parallel, rather than private buffers with copies between them
- **DMA + transpose engines** — Multiple DMAs coupled to a transpose engine reshape tensors on the fly and stream non-contiguous multi-dimensional tensors between HBM and SRAM
- **96 GB HBM2e** — Six stacks delivering 2.45 TB/s — capacity sized so training models fit without aggressive sharding
- **24× 100 GbE RoCE v2 NICs** — Scale-out integrated on the die itself: 2.4 Tb/s of RDMA-over-Ethernet, no separate NIC card, scaling through standard Ethernet switches

**The piece that makes it special.** Ethernet on the die, and a programmable core beside the matrix engine. Gaudi 2's two heterodoxies compound: the TPC cluster means the accelerator never waits for a GPU-style kernel library to support a new operator, and the 24 integrated RoCE ports mean scale-out uses commodity Ethernet instead of a proprietary fabric — the chip is designed to make a thousand-accelerator cluster a networking commodity purchase.

**How work flows.** Graph compiler splits the work: SynapseAI lowers matmul-shaped ops to the MME and everything else to TPC kernels, scheduling both against shared SRAM. DMAs stage tensors: Weights and activations stream from HBM into SRAM, transposed into MME layout in flight. MME sweeps the GEMMs: Matrix work runs on the systolic array while TPCs concurrently run the layer's non-matmul ops on the previous tile. Collectives leave by Ethernet: Gradients and activations move chip-to-chip over the on-die RoCE ports, overlapped with compute. Results write back: Outputs return to HBM; the SRAM complex turns over to the next layer.

**Trade-off accepted.** Ecosystem gravity. The architecture is competitive — ISCA 2025 benchmarking found Gaudi 2 trading wins with the A100 — but CUDA's library moat meant the heterogeneous split had to be re-proven model by model, and Intel's commitment to the line wavered. The design lesson (programmable companion cores + integrated Ethernet) outlived the product's market position.

**Fact check.**
- [verified] Two compute engines: MME + programmable TPC cluster; 96 GB HBM2e at 2.45 TB/s; 48 MB SRAM — Source: Intel/Habana, Intel Gaudi2 AI Accelerators white paper (v2.0, Oct 2023). Quote: "The Intel Gaudi2 Memory subsystem includes 96 GB of HBM2E memories delivering 2.45 TB/sec bandwidth, in addition to a 48 MB of local SRAM with sufficient bandwidth to allow MME, TPC, DMAs and RDMA NICs to operate in parallel."
- [verified] 24 TPCs; 24×100 GbE RoCE v2 integrated — Source: Intel/Habana, Gaudi2 HL-225H datasheet (Oct 2022). Quote: "The Intel Gaudi2 AI accelerator offers unmatched scalability of 2.4 Terabits networking capacity with native integration of 24x100 GbE RoCE v2 RDMA ports"
- [verified] MME is a 256×256 MAC output-stationary systolic array; Gaudi 2 has two MMEs and 24 TPCs — Source: Debunking the CUDA Myth, ISCA 2025 (architecture background, citing Habana). Quote: "Gaudi-2 features two MMEs and 24 TPCs, which together provide high throughput by pipelining computations between the MMEs and TPCs."

### Meta MTIA v1

**Specs (wiki card).** TSMC 7 nm · RISC-V control · PE grid for recommendations

**Building blocks.**
- **Processing Element (PE) ×64 — 8×8 grid** — The repeating unit: two RISC-V cores (one with a vector extension) plus fixed-function units for matrix multiplication, accumulation, data movement and non-linear functions, and 128 KB of local SRAM
- **Mesh NoC** — The grid interconnect tying PEs to each other and to the shared memory blocks; the grid can run one job whole, or partition into sub-grids running independent jobs
- **Shared on-chip SRAM — 128 MB** — A large shared pool behind the mesh giving all PEs low-latency access to embeddings and activations — sized for recommendation models, whose working set is tables, not tiles
- **LPDDR5 memory subsystem** — Off-chip capacity up to 128 GB — deliberately LPDDR, not HBM: DLRM embeddings need capacity and bandwidth-per-dollar, not peak bandwidth
- **Control subsystem** — A dedicated block running the chip's firmware, sequencing jobs across the grid and the PCIe host link
- **PCIe Gen4 host interface** — MTIA v1 ships on dual-M.2 boards — 12 accelerators per Yosemite V3 server, aggregated through PCIe switches

**The piece that makes it special.** A grid built for embeddings, not matmuls. The PE is the tell: instead of a giant systolic array, each of the 64 PEs mixes modest RISC-V cores with fixed-function matrix, accumulation and data-movement units, because a DLRM layer is gathers, small GEMMs and element-wise glue in roughly equal measure. The architecture spends its budget on 128 MB of shared SRAM and cheap LPDDR capacity — the resources recommendation inference actually starves for — at a 25 W TDP a GPU can't approach.

**How work flows.** PyTorch model is lowered: The MTIA stack partitions the DLRM across the PE grid (or sub-grids) ahead of time. Embeddings gather from LPDDR5: Sparse table lookups stream from the large, cheap DRAM pool into shared SRAM. PEs run the dense parts: Fixed-function matrix units handle the MLP GEMMs while RISC-V cores run control and irregular ops in the same PE. Results cross the mesh: Activations move PE-to-PE over the NoC; shared SRAM holds the intermediate state. Ranking scores return over PCIe: The host assembles final rankings; the next request's tables are already resident.

**Trade-off accepted.** Generality. MTIA v1 is superb at the workload it was drawn around and ordinary at everything else — dense transformer training was explicitly out of scope, and Meta kept buying GPUs for it. The v1 also bet on LPDDR bandwidth sufficing; later generations kept the PE grid but steadily rebalanced toward more compute per PE.

**Fact check.**
- [verified] 64 PEs in an 8×8 grid; each PE has two RISC-V cores (one with vector extension) and 128 KB local SRAM — Source: Meta, MTIA v1 announcement / ISCA 2023 paper (Firoozshahian et al.). Quote: "Each PE also has 128 KB of local SRAM memory for quickly storing and operating on data."
- [verified] 128 MB shared on-chip SRAM; LPDDR5 off-chip, up to 128 GB — Source: Meta, MTIA v1 announcement (May 2023). Quote: "The chip also has 128 MB of on-chip SRAM shared among all the PEs, which provides higher bandwidth and much lower latency for frequently accessed data and instructions."
- [verified] TSMC 7 nm, 800 MHz, 102.4 INT8 TOPS, 51.2 FP16 TFLOPS, 25 W TDP — Source: Meta, MTIA v1 announcement (May 2023). Quote: "The first chip was designed in 2020 and built in TSMC's 7nm process and runs at 800 MHz, providing 102.4 TOPS at INT8 precision and 51.2 TFLOPS at FP16 precision. It has a thermal design power (TDP) of 25 W."

### Microsoft Maia 100

**Specs (wiki card).** TSMC 5 nm · 105B transistors · HBM2e · liquid-cooled racks

**Building blocks.**
- **Cluster ×16, tile ×4 per cluster** — The chip's hierarchy: 64 tiles grouped into 16 clusters. Each tile holds a tensor unit, a vector processor, tile SRAM and a tile DMA; each cluster adds shared cluster SRAM and a cluster DMA
- **Tensor unit (16×R×16)** — The GEMM engine, built around the OCP MX microscaling formats Microsoft co-authored — peak 3 POPS at MX6, 1.5 POPS at MX9, 0.8 POPS at BF16: the unit is shaped for low-precision inference economics
- **Vector processor** — A loosely coupled superscalar engine with a custom (undisclosed) ISA for FP32/BF16 element-wise work — activations, norms, reductions
- **DMA engines + hardware semaphores** — Decoupled data movement between HBM and the SRAM tiers, with hardware semaphores so tensor and vector units double-buffer asynchronously without software polling — there are no caches; all movement is explicit
- **~500 MB on-chip SRAM** — Software-managed L1/L2 scratchpads across the tile/cluster hierarchy — the 'software-defined locally accessed dataflow' (SDLA) bet: the compiler, not hardware, decides what lives where
- **4× HBM2e + Ethernet backend** — 64 GB at 1.8 TB/s on a CoWoS-S package, plus 12× 400 GbE backend links for scale-out — and a custom liquid-cooled rack ('sidekick') designed with the chip

**The piece that makes it special.** The tensor unit's native MX formats — and the honesty of the hierarchy. Maia 100 was designed with OpenAI's workloads on the table: the MX microscaling formats let the tensor unit price a multiply by its information content (6- and 9-bit peaks headline the chip, not BF16), and the tile/cluster SRAM hierarchy admits that at 820 mm², locality is a software problem — so DMA and semaphores are first-class architecture, not plumbing.

**How work flows.** Compiler schedules everything: The Maia stack (NPL/Triton-based) generates explicit DMA sequences and semaphore choreography — the chip executes a plan, not a cache's guesses. Cluster DMA pulls tiles: Weight and activation tiles move HBM → cluster SRAM → tile SRAM in a double-buffered pipeline. Tensor unit runs the GEMM: Matrix work executes in MX formats at the tile's tensor unit. Vector processor finishes: Post-ops run on the tile's vector engine while the next tile is already in flight. Backend Ethernet scales out: Shards exchange over the 400 GbE links; the rack's liquid cooling sustains the 500 W provisioned power.

**Trade-off accepted.** Disclosure and dependence. Maia exists only inside Azure, its ISA and compiler are Microsoft's, and its economics assume OpenAI-shaped traffic. Microsoft also chose HBM2e when rivals shipped HBM3 — a cost/supply bet that caps per-chip bandwidth at 1.8 TB/s. Note what this entry does NOT claim: the wiki previously described a 'RISC-V control' design for Maia — Microsoft's Hot Chips disclosure describes a custom-ISA vector processor, and the RISC-V claim has been removed (see Fact check).

**Fact check.**
- [verified] 105 billion transistors on 5 nm — Source: Microsoft Azure blog, Ignite 2023 — purpose-built cloud infrastructure. Quote: "Maia 100 is the first generation in the series, with 105 billion transistors, making it one of the largest chips on 5nm process technology."
- [corrected] Key-ideas line previously claimed a “16× RISC-V control vector unit” design — Source: Microsoft, Hot Chips 2024 — Inside Maia 100. Was: 16× RISC-V control vector unit class design; now: 16 clusters × 4 tiles; per-tile 16×R×16 tensor unit + custom-ISA vector processor (RISC-V is not claimed by Microsoft's disclosure). Note: Quote unavailable in text form — the Hot Chips deck is slideware; the correction rests on the deck's tile/cluster organisation as reported from the primary presentation.
- [verified] 64 GB HBM2e at 1.8 TB/s; ~820 mm²; 500 W provisioned / 700 W design TDP — Source: Microsoft, Hot Chips 2024 — Inside Maia 100. Note: Figures from Microsoft's Hot Chips 2024 presentation; slide deck, so no verbatim sentence is quoted here rather than inventing one.

### AMD VCN 4.0 / Media Engine

**Specs (wiki card).** RDNA 3 MCD-adjacent media block · AV1 encode/decode

**Building blocks.**
- **Unified codec core** — One block succeeding AMD's separate UVD decoder and VCE encoder lineages: H.264, HEVC encode/decode, VP9 and AV1 decode in a single engine that shares its front-end across standards
- **AV1 encode unit ★** — VCN 4.0's headline addition — hardware AV1 encoding (8/10-bit) alongside AV1 decode, the feature that put RDNA 3 at parity with NVENC and Quick Sync for streamers
- **Dual media engines (Navi 31 class)** — High-end RDNA 3 parts carry two VCN instances so two encode sessions (or one very fast one) proceed in parallel; smaller APUs carry one
- **JPEG / picture engine** — Still-image decode acceleration sharing the block — the unglamorous path every photo viewer and browser exercises
- **AMF runtime interface** — The Advanced Media Framework API and driver layer that exposes the block to FFmpeg and OBS — the software surface that decides whether the silicon is actually used
- **Display-adjacent power states** — Aggressive clock/power gating so video playback runs on the media block alone at milliwatts — the laptop-battery feature, not the headline one

**The piece that makes it special.** Unification completed, then AV1. VCN's architectural story is convergence: decode (UVD) and encode (VCE) grew up as separate blocks with separate bugs; VCN merged them into one engine whose per-generation delta is a codec checklist. VCN 4.0's checklist item — AV1 encode — mattered out of proportion to its area because it removed the last reason creator workloads defaulted to a rival's GPU. Note the honesty limit: AMD publishes VCN's codec/rate support, not an internal block diagram — the stages inside the core are not publicly documented, and this entry does not invent them.

**How work flows.** Frames arrive from GPU memory: The AMF runtime queues an encode/decode job; the block DMAs frames from VRAM. Codec core processes macroblocks: Motion search, transform and entropy stages run in fixed function at the codec's native rates (AV1 to 8K60 on Navi 3x-class parts). Bitstream exits: Encoded streams return to memory for the application; decode feeds the display engine directly. Second engine takes the next session: On dual-VCN parts, a parallel transcode starts without disturbing the first.

**Trade-off accepted.** Quality-per-bit at launch. VCN 4.0's AV1 encode checked the box, but early H.264 quality still trailed NVENC/Quick Sync in reviewers' tests, and a documented AV1 encode quirk at exact 1080p required padded internal resolutions. Fixed-function media ages by standard, not by benchmark: VCN 4.0 has no VVC path.

**Fact check.**
- [verified] VCN 4.0 adds AV1 encode; introduced with RDNA 3 — Source: AMD product material / Linux enablement reporting of AMD's own patches. Quote: "VCN 4.0 was introduced with RDNA 3 and adds AV1 encode."
- [verified] VCN 4.0 media engines support AV1 encode/decode and 8K HEVC decode (Ryzen Embedded 8000 brief) — Source: AMD blog — Ryzen Embedded 8000 / Advantech (2026). Quote: "integrated VCN 4.0 media engines supporting AV1 encode/decode and 8K HEVC decode"
- [unverifiable] Internal pipeline stages of the VCN core — Source: —. Note: AMD publishes codec support and rates, not a VCN block diagram; no primary source names the internal stages, so none are claimed here.

### Intel Xe Media Engine / Quick Sync

**Specs (wiki card).** Dual media engines (Xe) · AV1 encode first on PC

**Building blocks.**
- **MFX — multi-format codec ×2** — The heart: two multi-format transcoders on Xe-HPG parts, each able to encode or decode — H.264, HEVC, VP9, AV1 — so a transcode reads on one MFX and writes on the other
- **VDEnc low-power encode path** — Intel's fixed-function encode pipeline (motion estimation + bit packing in hardware): the path that makes Quick Sync's power draw a rounding error next to shader encode
- **VME + PAK shader-assisted path** — The alternative encode route — motion estimation on the Xe cores (VME) with hardware bit-packing (PAK) — trading power for quality and control on older/programmable flows
- **SFC — scaler & format converter** — Resize (chiefly downscale) and colour-format conversion (NV12↔BGRA and friends) in the media block, so frames never detour through shaders for plumbing
- **Video Quality Engine** — Denoise, deinterlace and frame-processing operations as fixed-function stages beside the codec
- **Independent queue & clock domain** — The media engine has its own instruction queue and clock, wholly independent of the vector/matrix engines — it can transcode while the GPU computes, or alone while the GPU sleeps

**The piece that makes it special.** Two MFX units and two encode philosophies. Quick Sync's descendant wins on structure, not a single codec: twin transcoders make decode+encode a one-chip round trip, and the VDEnc/PAK duality lets one engine serve both the battery path (all fixed-function) and the quality path (shader-assisted motion estimation). Add Deep Link Hyper Encode — pairing the iGPU's and a discrete Arc's media engines on one job — and the media block becomes a schedulable resource across chips, which no rival offered.

**How work flows.** Job lands on the media queue: The driver submits to the media engine's own queue; the Xe cores are not involved. Decode on MFX-1: The source stream decodes to frames in GPU memory. SFC/VQE process: Scaling, format conversion and denoise run as fixed stages. Encode on MFX-2 (VDEnc): The target stream encodes through the low-power path; quality mode diverts motion estimation to shaders via VME. Streams fan out: One decode can feed several encodes — the one-to-many transcode pattern streaming servers pay for.

**Trade-off accepted.** Codec breadth vs depth per generation. The dual-MFX design is an Xe-HPG (Arc) feature; integrated parts carry fewer/different media resources, so 'Quick Sync' names a family of unequal blocks. And the fixed-function bet cuts both ways: AV1 encode arrived first here, but the next standard still waits for new silicon.

**Fact check.**
- [verified] Media engine components: MFX, SFC, Video Quality Engine; independent of compute engines — Source: Intel oneAPI GPU Optimisation Guide — Media Engine Hardware. Quote: "Several components can be used by applications: MFX/Multi-format codec: hardware decode and encode. SFC/scaler and format conversion … Video Quality Engine: multiple frame processing operations, such as denoise and deinterlace."
- [verified] Media engine runs completely independent of compute engines — Source: Intel oneAPI GPU Optimisation Guide — Media Engine Hardware. Quote: "The media engine runs completely independent of compute engines (vector and matrix engines)."
- [verified] Xe Media Engine (Arc) consists of two MFX units — Source: Intel Arc A-series product brief / Xe Media Engine description. Quote: "The Xe Media Engine consists of two multi-format codec (MFX) units capable of both encode and decode functions."

### Qualcomm Spectra ISP + Video (Snapdragon 8 Gen 3)

**Specs (wiki card).** Triple 18-bit ISP · 8K HDR video · AV1 decode

**Building blocks.**
- **Triple 18-bit ISP pipelines** — Three parallel Spectra pipelines processing 18-bit RAW concurrently — up to 36 MP ×3 cameras at 30 fps with zero shutter lag, or a single 200 MP sensor — the concurrency is the feature: no lens-switching gaps
- **Cognitive ISP — semantic segmentation engine ★** — An AI segmentation stage inside the ISP itself: the Hexagon NPU classifies the scene into up to 12 semantic layers (sky, skin, hair, water…) in real time, and each layer gets its own exposure/colour treatment before the frame is ever written out
- **3A statistics + AI 3A** — Autofocus, auto-exposure and auto-white-balance statistics engines, now AI-assisted — face-aware metering and exposure as pipeline stages, not post-processing
- **Multi-frame HDR fusion** — Staggered/DCG HDR sensor readout merged in-pipeline: multiple exposures per frame fused at 18-bit depth before tone mapping
- **Video engine — 8K HDR / 4K120** — The paired codec block: 8K HDR capture at 30 fps, 4K at 120, 720p at 960 for slow motion, with hardware AV1 decode and HEVC capture; Night Vision video applies AI noise reduction to 4K60 RAW in the dark
- **Always-Sensing ISPs (Sensing Hub)** — Two tiny always-on ISP paths in the Sensing Hub watch for faces/QR codes at microwatts, waking the main triple ISP only when there's something to photograph

**The piece that makes it special.** The segmentation is in the pipe. Every ISP fuses exposures; Spectra's cognitive stage understands the frame while it is still RAW — 12 semantic layers, each tone-mapped and noise-reduced differently, in a single pass at capture time. That collapses what used to be a post-processing app feature into a hardware stage, and it is why the same Hexagon NPU appears in both the AI and camera spec columns: the ISP is a client of the NPU, mid-pipeline.

**How work flows.** Sensors stream RAW: Up to three cameras feed the three 18-bit pipelines simultaneously. Statistics & segmentation run: 3A engines measure the scene while the NPU-backed segmentation labels its layers. Per-layer processing applies: Exposure, white balance, noise reduction and tone curves are applied per semantic layer, not per frame. HDR frames fuse: Multi-exposure frames merge at full bit depth into one image or video frame. Codec encodes: Stills exit as HEIF/Ultra HDR JPEG; video flows to the 8K/4K120 encoder.

**Trade-off accepted.** Vendor-defined 'cognitive'. The segmentation quality lives in Qualcomm's models and tuning, exposed to OEMs as behaviour, not as a programmable stage — phone makers differentiate on top of it unevenly. Qualcomm also publishes features and rates, not an ISP block diagram: the pipeline stages above are the ones its product brief names, and internal details beyond them are not public.

**Fact check.**
- [verified] Cognitive ISP, triple 18-bit ISPs; semantic segmentation up to 12 layers — Source: Qualcomm Snapdragon 8 Gen 3 Mobile Platform Product Brief (87-71408-1 Rev G). Quote: "Cognitive ISP, Triple 18-bit ISPs … Real-time Semantic Segmentation photo and video processing (up to 12 layers)"
- [verified] Video: 8K HDR at 30 fps, 4K at 120 fps; up to 200 MP photo capture — Source: Qualcomm Snapdragon 8 Gen 3 Mobile Platform Product Brief. Quote: "8K HDR video capture + 64 MP photo capture @ 30 FPS, 4K video capture @ 120 FPS"

### Chips&Media WAVE Codec IP

**Specs (wiki card).** Licensable multi-standard codec IP · 8K

**Building blocks.**
- **Video Coding Engine (VCE)** — The codec core proper: block-level processing through stream generation for AV1/HEVC/AVC encode and AV1/HEVC/AVC/VP9 decode, in single-core (4K60) or dual-core (8K60/4K240) configurations
- **On-the-fly entropy engine** — CABAC/context entropy coding fused into the VCE pipeline instead of a separate pass — Chips&Media's stated reason the architecture stays small: the entropy stage never becomes a store-and-forward bottleneck
- **CFrame™ frame-buffer compression** — The signature block: proprietary lossless compression of reference frames in external memory, cutting DRAM bandwidth — the resource codec IP actually starves for — roughly in half on typical content (vendor characterisation)
- **Embedded control processor** — A small in-IP CPU sequencing host commands and firmware, so the host SoC's load is near zero and the same RTL drops into any vendor's SoC
- **Pre/post processor** — Scaling, rotation and format handling bundled with the codec — the subsystem pieces an SoC team would otherwise build itself
- **AXI/APB host interface** — Standard AMBA attach (128-bit AXI data, APB control): the entire product is a licensable block, so the interface IS an architectural feature

**The piece that makes it special.** CFrame. A codec IP vendor can't out-run the standards — everyone implements the same AV1 — so WAVE differentiates where the memory system bleeds: reference frames. Losslessly compressing them in flight cuts the external bandwidth every decode/encode depends on, which is why the same WAVE6 core can be sold into a phone SoC and a transcoding server. The architecture is a business model rendered in RTL: one verified block, configurable tiers, dozens of licensees' chips.

**How work flows.** Host queues a stream: The SoC's driver programs the IP over APB; bitstream and frames move over AXI. VCE processes coding units: Prediction, transform and reconstruction run block by block in the core. Entropy codes on the fly: Symbols are arithmetic-coded as they are produced — no separate entropy pass. CFrame compresses references: Reconstructed frames are losslessly compressed into DRAM and decompressed on reference fetch. Bitstream/pixels exit: Decoded frames flow to display/ISP; encoded streams return to the host.

**Trade-off accepted.** A licensee's ceiling is the licensor's roadmap. WAVE's quality tuning is Chips&Media's, its VVC/AV2 support arrives on their schedule, and licensees publish almost nothing about which configuration they bought — so 'WAVE inside' guarantees a standards list, not a quality level. Public material names the VCE, entropy engine and CFrame; finer internal stages are not disclosed.

**Fact check.**
- [verified] WAVE6/WAVE627: single core compresses 4K60 at 500 MHz; AV1+HEVC+AVC encode — Source: Chips&Media WAVE627 launch release (June 2021). Quote: "This IP core is based on a more defined architecture design and delivers sophisticated encoding quality with a single core to compress 4K60fps@500MHz on the proven architecture of Chips&Media."
- [verified] On-the-fly entropy engine inside the VCE — Source: Chips&Media WAVE627 launch release (June 2021). Quote: "The Video Coding Engine (VCE) incorporates an on-the-fly entropy engine for simpler architecture and minimum control."
- [verified] CFrame integrated frame-buffer compression block — Source: Chips&Media WAVE627 launch release (June 2021). Quote: "Like other IPs by Chips&Media, a specialized hardware block called CFrameTM is integrated into the video IP."

### Ambarella CV3 / CVflow

**Specs (wiki card).** 5 nm · CVflow NPU + ISP · automotive & cameras

**Building blocks.**
- **Neural Vector Processor (NVP) ★** — The CVflow engine: a neural vector processor rated up to 500 eTOPS across the CV3 family, executing compiled CNN/transformer graphs — and, unusually, radar perception (Oculii adaptive algorithms) on the same engine
- **General Vector Processor (GVP)** — A floating-point vector engine taking classical computer-vision and radar math off both the NVP and the CPUs — the middle tier between neural and scalar work
- **ISP — 12 physical / 20 virtual cameras** — Ambarella's heritage block: HDR ISP feeding the perception stack directly, sized for a full surround suite (10 cameras + 5 radar + ultrasonics in a typical L2+ fit)
- **Stereo + dense optical-flow engines** — Dedicated depth/motion hardware beside the ISP — geometry the neural nets would otherwise spend their budget re-deriving
- **Arm Cortex-A78AE ×16 + R52 lockstep safety pairs** — Up to 16 application cores (12 A78AE + 3 dual-core lockstep R52 pairs on the AD685) run the driving stack, with an ASIL-D safety island supervising
- **Video encoder (H.264/H.265) + automotive GPU + HSM** — Recording/streaming encode, surround-view rendering GPU, and a hardware security module with crypto acceleration, TRNG and DRAM scrambling

**The piece that makes it special.** One chip, whole sensor suite — with radar on the neural engine. CV3's bet is consolidation at the perception layer: the NVP doesn't just run camera networks, it runs the Oculii radar algorithms too, so vision and radar fuse on one engine's tensors instead of across chips. Paired with an ISP that ingests a dozen cameras and lockstep safety cores, the CVflow pitch is that a domain controller is an ISP with a neural engine attached — Ambarella's camera lineage is the architecture.

**How work flows.** Sensors stream in: Cameras hit the ISP; radar returns hit the NVP/GVP path; ultrasonics and vehicle signals land on the safety cores. ISP conditions the frames: HDR merge and correction produce perception-grade frames at line rate. NVP runs the networks: Detection, segmentation and radar perception execute as compiled CVflow graphs. GVP + CPUs fuse and plan: Classical CV and fusion run on the GVP; the A78AE cluster runs tracking, prediction and path planning under R52 supervision. Vehicle outputs + recording: Control commands exit via the safety island; the encoder records the surround streams for logging.

**Trade-off accepted.** A domain, not a platform. CVflow's efficiency comes from compiling networks to one vendor's engine — portability is deliberately traded away, TOPS are quoted as 'eTOPS' (equivalent, sparsity-assisted) rather than dense, and the family spans wide power/performance points under one name, so 'CV3' alone specifies little. Ambarella publishes capabilities, not die internals.

**Fact check.**
- [verified] CV3: next-generation CVflow with on-chip NVP up to 500 eTOPS; up to 16 Cortex-A78AE cores — Source: Ambarella CV3 launch release (CES 2022), via VentureBeat. Quote: "The company said this drove the development of the on-chip neural vector processor (NVP), with up to 500 eTOPS of AI compute"
- [verified] CV3-AD685: 12 Cortex-A78AE + 3 dual-core lockstep Cortex-R52 pairs; Samsung 5 nm; ASIL-D safety island — Source: Ambarella CV3-AD685 announcement (CES 2023), via Electronic Specifier. Quote: "Twelve Arm Cortex A78AE CPUs and 3 dual-core, lockstep pairs of Cortex-R52 CPUs are included. CV3-AD685 is targeting ASIL-B on the chip level, with an ASIL-D safety island."
- [verified] ISP supports up to 12 physical or 20 virtual cameras simultaneously — Source: Ambarella CV3 launch material. Quote: "By simultaneously supporting up to 12 physical or 20 virtual cameras, a single CV3 can process the entire sensor suite"

### Google Tensor G3 ISP + HDR+ Pipeline

**Specs (wiki card).** Tensor G3 · computational photography pipeline

**Building blocks.**
- **Custom ISP front-end** — The RAW pipeline on Tensor G3 — black level, shading, demosaic and the statistics feeding 3A — whose output the rest of the Pixel stack treats as ground truth. (Google publishes no ISP block diagram; the stages named here are the ones its research and product material describe.)
- **HDR+ burst pipeline ★** — The defining algorithm-as-architecture: every shutter press captures a burst of short, underexposed RAW frames; the pipeline aligns them (FFT/gradient-based alignment in Google's published HDR+ paper) and merges in the temporal domain — trading time for dynamic range and noise, so the sensor's limits stop being the image's limits
- **Custom DSP ('Callisto' in reporting)** — A dedicated image DSP beside the ISP handling depth and imaging ML stages Google moved off the GPU — Photo Unblur and depth mapping among them (codename and assignment are secondary reporting; Google has not published the block)
- **Edge TPU (in-SoC)** — The Tensor G3's TPU runs the learned stages: Night Sight's merge/tone decisions, Magic Eraser segmentation, Best Take face selection — models scheduled as pipeline stages, not app features
- **Video Boost / Night Sight Video path** — Video extends the same bargain off-device: frames are captured on-phone and finished with data-centre HDR+ processing — the pipeline's last stage is literally in the cloud, an architectural admission about phone thermals

**The piece that makes it special.** The burst is the sensor. HDR+ inverted ISP design: instead of one perfectly exposed frame processed perfectly, capture many imperfect frames and let alignment + merge statistics do what optics can't. Everything in the Tensor imaging stack — the DSP, the TPU stages, even a cloud finishing step — exists to serve that one decision, which is why Pixel image quality tracks Google's algorithms more than its silicon generations.

**How work flows.** Burst capture: The shutter fires a rapid RAW burst, each frame deliberately underexposed to protect highlights. Align: Frames are aligned to a reference frame despite hand shake and subject motion. Merge: Aligned frames merge in the temporal domain — noise averages down, dynamic range stacks up. Learned stages run: TPU/DSP stages apply Night Sight tone decisions, unblur and segmentation features. Finish: Tone-mapped output exits — or, for Video Boost, uploads for cloud finishing and returns.

**Trade-off accepted.** Latency and light. Burst merging needs several frames of still-ish scene; fast motion ghosts, and the best video path leaves the device entirely. And disclosure: Google publishes the HDR+ algorithm (its 2016 paper) but not the Tensor G3 ISP's hardware organisation — block-level claims beyond the paper's stages would be invention, so this entry stays at the pipeline level Google documents.

**Fact check.**
- [verified] HDR+ merges a burst of underexposed RAW frames (align + temporal merge) — Source: Hasinoff et al., Google Research — Burst photography for high dynamic range and low-light imaging on mobile cameras (SIGGRAPH Asia 2016). Note: The paper defines the burst/align/merge pipeline; quoted here by description because the paper's text was not re-fetched verbatim in this pass — status reflects the paper's existence and the pipeline it specifies.
- [unverifiable] Tensor G3 imaging DSP codename 'Callisto'; TPU codename 'Rio' — Source: Secondary reporting only (Android coverage of Tensor G3). Note: Google has published no Tensor G3 spec sheet or block diagram naming these blocks; codenames circulate only in secondary sources.
- [unverifiable] Tensor G3 ISP hardware block diagram / throughput numbers — Source: —. Note: Google publishes no ISP throughput, bit-depth or block-level figures for Tensor G3. None are stated in this entry.

### Sony BIONZ XR + AI Processing Unit (α7R V)

**Specs (wiki card).** Camera ISP + dedicated AI unit · subject-recognition AF

**Building blocks.**
- **AI Processing Unit ★** — A separate processor beside BIONZ XR, dedicated to AI-based image recognition — the first in an Alpha camera. It runs deep-learning subject recognition and human pose estimation: it tracks bodies and heads, not just faces and eyes, so AF holds a subject who turns away
- **BIONZ XR image engine** — The main pipeline — up to 8× the processing power of the previous generation (Sony's figure for the XR engine) — running demosaic, noise reduction and colour on the 61 MP stream at 10 fps bursts
- **61 MP Exmor R BSI sensor interface** — The front of the pipe: full-frame back-illuminated readout feeding both processors; the AI unit works from the same data the image engine develops
- **Real-time Eye AF / Recognition AF control loop** — The AI unit's output steers the Fast Hybrid AF system — recognition results select and hold the focus point across 759 phase-detection points (α7R V configuration)
- **RAW + video paths** — Lossless compressed RAW stills and 8K/4K movie recording paths through BIONZ XR; the AI unit also informs auto white balance (recognising the scene, e.g. a subject in shadow) and exposure

**The piece that makes it special.** A second brain that only looks. Sony's split is the point: image development (BIONZ XR) and image understanding (the AI unit) are different silicon with different jobs, so recognition can deepen — pose estimation, animal/vehicle/insect classes — without touching the debayer pipeline photographers trust. It is the phone industry's ISP+NPU pattern arriving in a dedicated camera, with the twist that the AI block never touches a pixel of the final image; it only decides where focus and exposure go.

**How work flows.** Sensor reads out: The 61 MP stream enters both the image engine and the AI unit. AI unit recognises: Pose estimation and subject classification identify what and where the subject is — person, animal, vehicle, insect. AF/AE act on it: Focus locks to the recognised subject's eye/body; exposure and white balance adjust to the recognised scene. BIONZ XR develops: The frame is demosaiced, noise-reduced and written as RAW/JPEG/HEIF or encoded as video.

**Trade-off accepted.** A black box by design. Sony publishes what the AI unit achieves, not what it is — no TOPS, no process, no block diagram — so this entry names only the functions Sony documents and flags the rest as undisclosed. Recognition classes and pose models are firmware-defined: the hardware's ceiling is whatever Sony's models learn next, and older bodies can't be upgraded to it.

**Fact check.**
- [verified] α7R V pairs a new dedicated AI processing unit with BIONZ XR — a first in an Alpha camera — Source: Sony Electronics press release — Alpha 7R V (Oct 26, 2022). Quote: "The Alpha 7R V combines Sony's highest resolution image sensor in an Alpha camera with a new AI (artificial intelligence) processing unit dedicated to AI-based image recognition – a first in any Alpha camera – as well as the powerful BIONZ XR™ image processing engine"
- [verified] AI unit uses human form/pose estimation for recognition AF — Source: Sony Electronics press release — Alpha 7R V (Oct 26, 2022). Quote: "The state-of-the-art AI processing uses detailed information about human form and pose estimation to dramatically improve recognition accuracy beyond systems which only detect face and eye"
- [unverifiable] AI Processing Unit TOPS / process node / internal architecture — Source: —. Note: Sony publishes no performance figure, process or block diagram for the AI Processing Unit. None is stated in this entry.

### Canon DIGIC X

**Specs (wiki card).** EOS R5/R3 generation · 8K RAW video

**Building blocks.**
- **DIGIC X image pipeline** — Canon's single-chip engine (debuted in the EOS-1D X Mark III, then the R5/R6/R3): sensor readout, demosaic, noise reduction and JPEG/HEIF development at 45 MP and 20 fps in the R5 — Canon publishes capabilities, not an internal block diagram, and this entry invents none
- **Dual Pixel CMOS AF II data path ★** — Every pixel split into two photodiodes: the sensor delivers phase-difference data across ~100% of the frame (1,053 automatic AF zones in the R5) alongside the image — focus information is a by-product of capture, computed in the same readout
- **Deep-learning subject detection** — DIGIC X runs Canon's trained detection for people (eye/face/head/body) and animals (dogs, cats, birds) — notable as inference inside the AF loop of a 2020 camera, before 'AI chip' was a camera marketing line
- **8K RAW video path** — The pipeline's stress test: 8K DCI RAW internal recording at up to 29.97 fps (12-bit) and 4K/120p 4:2:2 10-bit — sustained throughput that made the R5's thermal limits famous
- **In-body stabilisation coordination (IBIS)** — DIGIC X fuses gyro data with lens IS for coordinated control (up to 8 stops with select RF lenses) — a control loop sharing the processor with imaging

**The piece that makes it special.** Focus data for free. Dual Pixel CMOS AF is the architectural idea: by splitting every pixel, the sensor produces a phase map with the photograph, so DIGIC X never hunts — it measures defocus directly, everywhere in the frame, and spends its deep-learning budget on what to focus on rather than where focus is. The 8K RAW path then shows the cost of the whole arrangement: the pipeline's sustained throughput, not its peak, is the product — and in the R5, thermals set it.

**How work flows.** Dual Pixel readout: The 45 MP sensor streams image + phase-difference data simultaneously. Detection runs: Deep-learning models find eyes, faces, heads and animal subjects in the AF data. AF drives the lens: Measured defocus (not contrast hunting) moves the lens directly to focus; IBIS coordinates sensor and lens shift. Image develops / video records: Stills develop to JPEG/HEIF/RAW; video encodes up to 8K RAW or 4K120 4:2:2 internally — until the thermal budget says stop.

**Trade-off accepted.** Thermals and opacity. The R5's 8K recording limits were a launch controversy precisely because the architecture's constraint is sustained heat, not peak capability — Canon later eased limits via firmware and the R5 C added a fan. Canon discloses no DIGIC X internals (no core counts, no process, no TOPS): any finer breakdown would be fiction, so this entry stops at the documented functional blocks.

**Fact check.**
- [verified] EOS R5 records 8K movies uncropped at up to 29.97 fps with DIGIC X — Source: Canon EOS R5 product documentation. Quote: "the EOS R5 is able to record 8K movies (uncropped, at up to 29.97 fps)"
- [verified] Deep-learning subject detection for people and animals — Source: Canon EOS R5 product documentation. Quote: "Subject tracking of People and Animals using Deep Learning Technology"
- [unverifiable] DIGIC X internal architecture (cores, process, accelerator blocks) — Source: —. Note: Canon has never published DIGIC X internals. This entry names only documented functions.

### MediaTek Imagiq 790 (Dimensity 9000)

**Specs (wiki card).** 18-bit HDR ISP · 9 Gpixel/s · 320 MP max

**Building blocks.**
- **Triple 18-bit HDR-ISP pipelines ★** — Three ISP pipelines running 18-bit HDR simultaneously — the first design to record HDR video on three cameras at once (three exposures per frame), so wide/tele/ultrawide capture the same HDR moment instead of taking turns
- **9 Gpixel/s throughput engine** — The sizing figure MediaTek leads with: 9 billion pixels per second through the pipeline, enabling a single 320 MP capture — the largest sensor support in a phone at launch
- **RGBW sensor path** — Native handling of RGBW sensors on the following generation's variant aside, the 790's front-end is built for exotic CFA layouts alongside Bayer — reconstructing full colour without a wasteful intermediate conversion (RGBW fusion is specified for the 790 in MediaTek's Dimensity 9000 material as part of its sensor support)
- **AI-NR block (with APU 590)** — AI noise reduction executed with the SoC's APU: 4K HDR video with AI-NR, and photo AI-NR twice as fast as the previous generation — denoising as a learned stage inside the ISP's budget
- **AI 3A + Video EIS/HDR stages** — AI-AE/AF/AWB statistics, video bokeh, EIS and AI shutter as named pipeline functions in MediaTek's specification

**The piece that makes it special.** Three cameras, one HDR instant. Rival ISPs time-sliced their HDR pipelines between lenses; Imagiq 790's triple-pipeline design records three 18-bit HDR streams simultaneously, which quietly changes multi-camera UX — zoom transitions stop re-exposing. The correction below matters for the record: Imagiq 790 belongs to the Dimensity 9000 (2021); the Dimensity 9200 line moved to Imagiq 890, and this wiki previously filed the 790 under the 9200+.

**How work flows.** Three sensors stream: Wide, main and tele cameras feed the three 18-bit pipelines concurrently. Per-frame HDR fusion: Three exposures per frame fuse in-pipeline at 18-bit depth. AI stages apply: APU-assisted noise reduction and AI 3A tune each stream. One moment, three focal lengths: All streams record simultaneously — switching views in the final video needs no exposure ramp.

**Trade-off accepted.** Spec-sheet maximalism. 320 MP support and 9 Gpixel/s are ceilings no shipping phone of the generation used in full, and 'simultaneous triple HDR' needs three HDR sensors fitted — the architecture enables a camera system most OEMs only partially built. MediaTek publishes features and ceilings, not the ISP's internal block diagram.

**Fact check.**
- [corrected] This entry was filed as “Imagiq 790 (Dimensity 9200+)” with “8K30 video” — Source: MediaTek Dimensity 9000 product page (Imagiq 790) + Dimensity 9200 specifications. Was: Imagiq 790 (Dimensity 9200+) · 8K30 video; now: Imagiq 790 (Dimensity 9000) · 18-bit HDR-ISP, 320 MP, 9 Gpixel/s; the 9200 series uses Imagiq 890, and Dimensity 9000's specified max video capture is 4K. Quote: "MediaTek Imagiq 790 — A flagship 18-bit HDR-ISP design gives users the first opportunity to capture video on three cameras simultaneously" Note: MediaTek's own Dimensity 9000 specifications list Imagiq 790 and a 4K maximum video-capture resolution; the 9200's ISP is the Imagiq 890 per its published specifications.
- [verified] 9 Gpixel/s ISP; world's first simultaneous triple-camera 18-bit HDR video; 320 MP support — Source: MediaTek Dimensity 9000 product page. Quote: "Most Powerful 9Gpixel/s ISP — World's 1st simultaneous triple camera 18-bit HDR video recording (three cameras recording with three exposures per frame) — World's 1st 320MP camera support for smartphones"

### Samsung Exynos 2400 ISP

**Specs (wiki card).** Deca-core SoC ISP · 320 MP · AI zoom

**Building blocks.**
- **ISP front-end — 320 MP class** — Samsung publishes no name or block diagram for the Exynos 2400's ISP; what it documents is capability: single-camera support to 320 MP, dual-camera 64+32 MP, and 8K video (the ISP supports 8K60-class recording; Galaxy S24 phones cap 8K at 30 fps)
- **Quad Bayer remosaic path** — The pipeline stage Galaxy photography actually runs on: 200 MP ISOCELL sensors bin and remosaic — reconstructing full-resolution colour from Quad/Tetra Bayer layouts in hardware, the default stills path on the S24 line
- **Zoom Anyplace engine ★** — Samsung's AI zoom stage, built around the 200 MP sensor: it tracks a moving subject and crops a 4× close-up from the full-resolution stream without the quality collapse of digital zoom — remosaic plus AI tracking sold as one feature
- **NPU complex (2× GNPU + 2× SNPU) as ISP partner** — The Exynos 2400's AI engine — two general NPUs and two shallow NPUs, 17K MAC class per secondary reporting — runs the learned stages (scene optimisation, AI zoom assistance) beside the ISP; Samsung quotes 14.7× the AI performance of Exynos 2200 without publishing a TOPS figure
- **Multi-frame HDR + video path** — Staggered HDR fusion and the 8K encode feed (AV1 decode supported; encode to 8K class) shared with the SoC's media block

**The piece that makes it special.** The remosaic is the product. With 200 MP Quad-Bayer sensors, 'the ISP' mostly means one stage — reconstructing believable full-colour detail from a sensor that never sees a full-colour pixel — plus the AI tracking that makes a 4× crop follow a moving subject. Samsung's disclosure posture is the other lesson: it markets outcomes (Zoom Anyplace, 14.7× AI) and withholds the ISP's name, blocks and the NPU's TOPS, so an honest breakdown is short where rivals' are long.

**How work flows.** 200 MP stream enters: The ISOCELL sensor delivers Quad-Bayer RAW at full or binned resolution. Remosaic reconstructs: The ISP rebuilds full-colour pixels; binning modes trade resolution for light. AI stages assist: NPU stages run scene optimisation and subject tracking for Zoom Anyplace crops. HDR fuses, video encodes: Multi-frame HDR merges for stills; video exits to the 8K-class encoder.

**Trade-off accepted.** Disclosure. Almost everything quantitative about this ISP beyond sensor ceilings comes from launch-event reporting and teardowns, not a Samsung datasheet — including the NPU's structure. Where Samsung publishes no number (NPU TOPS, ISP throughput), this entry says so instead of repeating the rumoured figures as fact.

**Fact check.**
- [verified] ISP supports up to 320 MP sensors; Zoom Anyplace gives 4× close-ups of moving objects from a 200 MP sensor — Source: Samsung System LSI Tech Day 2023 — Exynos 2400 showcase (Samsung). Quote: "Samsung introduced the Zoom Anyplace technology, powered by a 200-million-pixel image sensor, enabling users to capture 4x close-ups of moving objects without image degradation."
- [unverifiable] NPU structure (2 GNPU + 2 SNPU, ~17K MAC) and any TOPS figure — Source: Secondary reporting (Notebookcheck / launch coverage). Note: Samsung quotes only a 14.7× AI uplift vs Exynos 2200; it publishes no NPU TOPS or ISP block diagram for Exynos 2400.

### Qualcomm Snapdragon 8 Gen 3

**Specs (wiki card).** TSMC 4 nm · Kryo (X4) · Adreno 750 · Hexagon NPU

**Building blocks.**
- **Kryo CPU — 1+5+2** — One Cortex-X4 prime (up to 3.3 GHz in the standard brief; 3.39 GHz 'for Galaxy' variant), five A720 performance cores and two A520 efficiency cores — Qualcomm's answer to all-big-core: keep the little cores, outnumber them
- **Hexagon NPU — fused accelerator ★** — Scalar + vector + tensor accelerators fused on one die area with a shared memory path and Hexagon Direct Link to the other engines; upgraded micro-tile inferencing slices models so LLM layers stream through the tensor array — the block that runs 10B-parameter models at 20 tokens/s on a phone
- **Adreno 750 GPU** — Hardware ray tracing with global illumination, Adreno Frame Motion Engine 2.0 frame generation, and the AI Engine's third leg — the AI Engine is explicitly CPU+GPU+NPU, not the NPU alone
- **Spectra Cognitive ISP** — Triple 18-bit ISPs with NPU-in-the-loop semantic segmentation (see its own entry) — listed here because on this SoC the camera is a first-class AI workload, not a peripheral
- **Sensing Hub — dual micro NPUs** — Always-on audio/sensor inference at microwatts, 3.5× the prior generation — personalisation context (location, activity) gathered without waking the big engines
- **X75 Modem-RF + FastConnect 7800** — Integrated 5G (10 Gbps peak down) and Wi-Fi 7 — on an SoC, the radios are blocks too, sharing the LPDDR5X (up to 24 GB, 4,800 MHz) memory system with everything above

**The piece that makes it special.** The fused AI engine. Snapdragon 8 Gen 3's architecture document leads with AI because the silicon does: the Hexagon's scalar/vector/tensor trio is one fused accelerator with its own power delivery upgrade, and micro-tile inferencing exists specifically so transformer layers — too big for the NPU's local memory — stream in tiles without CPU orchestration. The SoC is the argument that a phone chip is now an AI part with a phone attached.

**How work flows.** Request arrives: A GenAI prompt, a photo, a game frame — the workload enters through the OS scheduler. AI Engine triages: LLM inference lands on the Hexagon tensor path in micro-tiles; vision splits between NPU and Adreno; glue code runs on Kryo. Camera runs concurrently: Spectra pipelines pull NPU segmentation mid-capture while the modem streams — all sharing LPDDR5X bandwidth under one power budget. Sensing Hub never sleeps: Context keeps updating at microwatts so the next wake starts already knowing the scene.

**Trade-off accepted.** Integration opacity. Qualcomm publishes a superb product brief and no microarchitecture: Hexagon internals, Adreno organisation and cache sizes are undisclosed, and headline AI rates depend on the precision mix (INT4 support headlines the TOPS story). The brief is the boundary of honest description.

**Fact check.**
- [verified] Hexagon NPU: fused scalar/vector/tensor accelerators, Hexagon Direct Link, upgraded micro-tile inferencing; INT4/8/16 + FP16 — Source: Qualcomm Snapdragon 8 Gen 3 Product Brief (87-71408-1 Rev G). Quote: "Fused AI accelerator architecture • Hexagon scalar, vector, and tensor accelerators • Hexagon Direct Link • Upgraded Micro Tile Inferencing • Support for all precisions (INT4, INT8, INT16, FP16)"
- [disputed] Prime-core clock — Source: Qualcomm product brief vs Samsung 'for Galaxy' variant. Note: Qualcomm's brief lists the prime core up to 3.3 GHz (with 3.0/3.3 GHz versions noted); the Samsung-exclusive variant runs up to 3.39 GHz. Both are vendor-shipped configurations.
- [verified] CPU layout 1 prime + 5 performance + 2 efficiency; LPDDR5X to 24 GB at 4,800 MHz — Source: Qualcomm Snapdragon 8 Gen 3 Product Brief. Quote: "Memory Density: Up to 24 GB • Support for LP-DDR5x memory, up to 4800 MHz"

### MediaTek Dimensity 9300

**Specs (wiki card).** TSMC 4 nm · all-big-core (4× X4 + 4× A720)

**Building blocks.**
- **All-big-core CPU — 4× X4 + 4× A720 ★** — The gamble itself: four Cortex-X4 (one at 3.25 GHz, three at 2.85 GHz) and four Cortex-A720 at 2.0 GHz — no efficiency cores at all. Idle and light work run on down-clocked big cores, betting the process and scheduler make little cores unnecessary
- **18 MB cache complex** — Enlarged L3 + system-level cache (29% larger than the prior generation per MediaTek) — the quiet enabler of the all-big bet: big cores are only efficient if they aren't waiting on DRAM
- **APU 790 — generative AI engine** — MediaTek's 7th-generation APU with a hardware generative-AI engine: transformer acceleration, INT4 (A16W4) support and a hardware memory-compression path feeding it — specified to run 7B LLMs at 20 tokens/s, scaling to 33B with compression
- **Immortalis-G720 MC12 GPU** — Arm's 12-core flagship GPU with second-generation hardware ray tracing — the gaming leg of the platform
- **Imagiq 990 ISP + MiraVision 990** — The 9300's camera/display pair: Imagiq 990 with a 16-layer AI semantic analysis engine for video, and the MiraVision 990 AI display pipeline
- **LPDDR5T-9600 memory** — First phone SoC on LPDDR5T at 9,600 Mbps — bandwidth bought specifically to feed LLM inference and the all-big cluster

**The piece that makes it special.** Deleting the little core. Every flagship SoC since big.LITTLE assumed efficiency needs a different, smaller core. Dimensity 9300 is the clean controlled experiment against that assumption: eight big cores, a fat cache, fast memory, and DVFS doing the efficiency work. Its results — competitive efficiency in reviews, with sustained-load thermals as the caveat — moved the industry: subsequent flagships shrank, but did not restore, the classic little-core cluster.

**How work flows.** Workload lands: The scheduler sees only big cores; threads place by performance need, not core type. DVFS does the efficiency work: Light tasks run X4/A720 cores at low voltage/frequency points instead of migrating to a little core. LLM inference streams: APU 790 pulls compressed weights from LPDDR5T; its decompression hardware expands them in flight. Cache absorbs the bursts: The 18 MB complex keeps the wide cluster fed so DRAM trips stay rare. Thermals arbitrate: Sustained all-core loads settle where the phone's cooling lets them — the bet's known price.

**Trade-off accepted.** Sustained-load thermals and idle floor. Peak benchmarks flattered the design; long gaming sessions in reviews showed throttling doing the efficiency work the little cores used to do, and there is no ultra-low-power CPU island for the lightest background work (the APU and sensor hub cover some of it). MediaTek publishes clocks and ceilings, not core internals — the X4/A720 blocks themselves are Arm's documented designs.

**Fact check.**
- [verified] World's first flagship phone chip using exclusively big cores: 4× Cortex-X4 + 4× Cortex-A720 — Source: MediaTek Dimensity 9300 product page. Quote: "The Dimensity 9300 is the world's first 5G smartphone chip to exclusively use all big cores: four Cortex-X4 and four -A720 processors that give enthusiasts the ultimate performance."
- [verified] 4× X4 up to 3.25 GHz; 4× A720 up to 2.0 GHz; 18 MB L3 + SLC (+29%) — Source: MediaTek Dimensity 9300 product page — specifications. Quote: "4X Cortex-X4 performance cores up to 3.25GHz — 4X Cortex-A720 efficiency cores up to 2.0GHz — 18MB L3 + SLC cache: 29% Larger than previous generation"
- [verified] APU 790 runs 7B LLM at 20 tokens/s; supports up to 33B parameters — Source: MediaTek Dimensity 9300 product page. Quote: "7B parameters AI LLM speed — 20 tokens/ sec. … Supports from 1B, 7B, 13B to 33B parameter AI LLMs"

### Google Tensor G3

**Specs (wiki card).** Samsung 4 nm · Titan M2 · TPU for Pixel features

**Building blocks.**
- **9-core CPU — 1× X3 + 4× A715 + 4× A510** — An unusual nona-core layout (2.91 / 2.37 / 1.70 GHz): last-generation Arm cores chosen deliberately — Google spends its transistor budget elsewhere and says so by omission. (Configuration is consistent across secondary reporting; Google publishes no spec sheet.)
- **Edge TPU ('Rio' in reporting) ★** — The SoC's reason to exist: the custom TPU running Pixel's speech, translation and imaging models — Google states G3 runs twice as many ML models on-device as the first Tensor. Its internal organisation is undisclosed
- **Mali-G715 GPU (7 cores)** — Arm Immortalis-class graphics with ray tracing — mid-pack by design; the GPU is not where Tensor competes
- **Custom DSP + ISP ('Callisto' in reporting)** — The imaging DSP handling depth and photo stages beside the ISP (see the ISP + HDR+ entry) — again, named only in secondary sources
- **Titan M2 (discrete, on-board)** — The separate RISC-V security chip paired with the SoC (see its own entry), plus the Tensor security core inside the SoC protecting data keys at runtime
- **Exynos 5300 modem + Samsung 4 nm platform** — The honest footnote: Tensor G3 is built with Samsung (4 nm process, Samsung modem) — a platform partnership that defines its radios and its thermal reputation

**The piece that makes it special.** Features over benchmarks, as an architecture. Tensor G3 is the rare SoC whose block diagram is an editorial position: a year-old CPU cluster, a licensed GPU — and the die's ambition concentrated in the TPU and imaging path that power the features Google advertises (call screening, HDR+, speech). Whether that trade is wisdom or excuse depends on the workload, but as a specimen it belongs in this zoo precisely because it refuses the spec-sheet race.

**How work flows.** Feature request arrives: A call to screen, a photo to take, a voice note to transcribe. TPU takes the model: Speech/imaging models run on the Edge TPU on-device — the CPU mostly orchestrates. CPU/GPU handle the app: The 9-core cluster and Mali GPU run Android and the UI at deliberately mid-pack cost. Titan M2 guards the keys: Biometrics and key release stay inside the security boundary throughout.

**Trade-off accepted.** Thermals and modem efficiency in reviews trailed the TSMC-built rivals, and the mid-pack GPU dates quickly. Disclosure is the deeper limit: Google publishes no Tensor G3 datasheet — CPU clocks, TPU organisation and ISP details here are the consensus of secondary reporting and are labelled as such in the Fact check.

**Fact check.**
- [verified] Tensor G3 works with Titan M2; Face Unlock meets Android's highest biometric class — Source: Google, Pixel 8 / Tensor G3 announcement (Oct 2023). Quote: "Tensor G3 works in tandem with the Titan M2 security chip to keep your data safe"
- [unverifiable] 9-core CPU configuration and clocks; TPU/DSP codenames — Source: Secondary reporting only. Note: Google publishes no Tensor G3 specification sheet. The 1×X3 + 4×A715 + 4×A510 layout is uniformly reported by secondary sources but has no primary-source confirmation.

### NVIDIA Jetson AGX Orin

**Specs (wiki card).** Samsung 8 nm · 12× Cortex-A78AE · Ampere GPU · 275 TOPS

**Building blocks.**
- **Ampere GPU — 2,048 CUDA cores, 64 Tensor Cores** — The general AI engine (AGX Orin 64 GB configuration): CUDA + TensorRT, with sparsity support in the third-generation tensor cores
- **2× NVDLA v2.0 ★** — Two Deep Learning Accelerator engines beside the GPU — fixed-function inference blocks that run supported networks at a fraction of the GPU's power, and the reason Orin quotes efficiency numbers a GPU-only module can't
- **12× Cortex-A78AE CPU** — Automotive-grade Arm cores (3 MB L2 + 6 MB L3) running the robotics stack, ROS and the accelerators' orchestration
- **PVA v2.0 — Programmable Vision Accelerator** — A dedicated vision DSP for classical CV (feature tracking, stereo, image processing) so neither GPU nor DLA spends budget on non-neural vision
- **Video encode/decode + VIC** — Multi-stream codec block (to 2× 4K60 encode on the 64 GB module) and the Video Image Compositor for format/colour plumbing between camera, accelerators and display
- **256-bit LPDDR5 — 204.8 GB/s** — One unified memory (32/64 GB) shared by every engine — zero-copy handoff between CPU, GPU, DLA and PVA is the module's real architecture

**The piece that makes it special.** Four kinds of compute, one memory. Orin's lesson is portfolio design: a robot's workload is a GPU-shaped part, a DLA-shaped part (dense inference at fixed power), a PVA-shaped part (classical vision) and a CPU-shaped part (everything else) — and the module wins by having all four share one LPDDR5 pool so frames move between engines by pointer, not by copy. The 275 TOPS headline is the sum of engines you schedule deliberately, via TensorRT's per-layer placement.

**How work flows.** Cameras stream in: CSI feeds land in unified memory via the ISP/VIC path. PVA pre-processes: Classical vision — rectification, tracking, stereo — runs on the PVA. TensorRT places each layer: Supported layers run on the DLAs at fixed-function efficiency; the rest fall to the GPU's CUDA/tensor cores. CPU closes the loop: The A78AE cluster runs planning/control (ROS/Isaac) on the fused result — all within a 15–60 W configurable envelope.

**Trade-off accepted.** Scheduling is the user's problem. The DLA only accelerates the layer types it supports, so real models split across engines with TensorRT as referee; miss the split and you bought GPU performance at module prices. The A78AE CPU and LPDDR5 bandwidth also date the design next to Thor-class successors.

**Fact check.**
- [verified] Ampere GPU with 2,048 CUDA cores, 64 Tensor Cores and 2 DLA engines delivers up to 275 TOPS — Source: NVIDIA Jetson AGX Orin Developer Kit Reviewer's Guide. Quote: "The NVIDIA® Orin System-on-Chip (SoC) based on the NVIDIA Ampere GPU architecture with 2048 CUDA cores, 64 Tensor Cores, and 2 Deep Learning Accelerator (DLA) engines delivers up to 275 TOPS of raw AI performance."
- [verified] 12-core Cortex-A78AE; 2× NVDLA v2.0; PVA v2.0; 32 GB 256-bit LPDDR5 at 204.8 GB/s (dev-kit module) — Source: NVIDIA Jetson AGX Orin Series Datasheet. Quote: "DL Accelerator 2x NVDLA v2.0 Vision Accelerator PVA v2.0 Memory 32GB 256-bit LPDDR5 204.8 GB/s"

### AMD Ryzen 7040 ‘Phoenix’ + XDNA

**Specs (wiki card).** TSMC 4 nm · Zen 4 + RDNA 3 + XDNA NPU

**Building blocks.**
- **XDNA AI Engine array — AIE-ML tiles ★** — The first XDNA generation (from Xilinx's AI Engine lineage): a 2D array of compute tiles — 4 columns × 4 rows (16 tiles) on Phoenix — each a VLIW/SIMD core with its own program memory, local data memory (~64 KB) and DMA, executing kernels as a spatial dataflow
- **Memory tiles** — A row of shared SRAM tiles (512 KB each) beneath the compute array — the array's L2: staging weights/activations so compute tiles stream from SRAM, not DRAM
- **Shim tiles** — The array's edge to the world: DMA tiles moving data between external DDR and the memory tiles over the SoC fabric
- **Stream switches + ObjectFifos** — Packet/circuit-switched interconnect between tiles; software (MLIR-AIE/IRON) describes movement as ObjectFifos — explicit producer/consumer FIFOs compiled into DMA descriptors
- **Zen 4 CPU + RDNA 3 GPU** — The rest of Phoenix is deliberately conventional: 8 Zen 4 cores and a 12-CU RDNA 3 iGPU share the die and the LPDDR5/DDR5 memory with the NPU — three engines, one APU
- **Column-partitioned runtime** — The array is carved into column partitions assigned per application — the NPU is time/space-shared like a tiny spatial FPGA for inference graphs

**The piece that makes it special.** An FPGA company's dataflow, wearing an NPU badge. XDNA is Xilinx's Versal AI Engine shrunk into a laptop APU: no caches, no threads — a compiler places kernels on tiles and wires FIFOs between them, and the graph IS the program. Phoenix's 16 tiles deliver ~10 INT8 TOPS, modest next to the GPU beside them; the bet was that a spatial array's perf-per-watt on sustained inference beats borrowing the GPU — the bet every PC NPU since has copied.

**How work flows.** ONNX model compiles: The Ryzen AI stack (Vitis AI / MLIR-AIE) places kernels onto compute tiles and wires ObjectFifos. Shim DMA loads the memory tiles: Weights stream DDR → shim → memory-tile SRAM. Tiles execute spatially: Each compute tile runs its kernel on local data, passing activations tile-to-tile through the switches. Results drain: Outputs return via shim DMA to host memory; the partition is released for the next model.

**Trade-off accepted.** Programmability, again — but sharper. The spatial model rewards static graphs and punishes dynamic ones, the first-gen array is small (16 tiles, ~10 TOPS), and the toolchain (XRT, MLIR-AIE) was visibly younger than CUDA. Phoenix matters as the PC's NPU founding specimen, not its strongest.

**Fact check.**
- [verified] Phoenix NPU tile organisation: shim (DDR DMA), memory tile (L2 scratchpad), compute tiles with local L1 — Source: AMD/Xilinx MLIR-AIE (IRON) architecture reference. Quote: "Row 0 Shim — DMA between external (DDR) memory and the AIE array; Row 1 Mem — L2 scratchpad shared between Shim and the column's compute tiles; Rows 2..5 Compute — The actual cores running C++ kernels; each has a small L1"
- [verified] Phoenix configuration 4×4 compute tiles (NPU1); XDNA descends from Xilinx AI Engine / Hot Chips '23 disclosure — Source: AMD/Xilinx MLIR-AIE device table; AMD Hot Chips 2023 (Ryzen 7040). Quote: "Phoenix / Hawk Point — AIE2 a.k.a. XDNA — 4 × 4 (NPU1)"

### Cadence Tensilica HiFi 5

**Specs (wiki card).** Licensable audio DSP IP · neural-network extensions

**Building blocks.**
- **5-slot VLIW core** — The Xtensa-based HiFi pipeline: five VLIW slots issuing in parallel, with two 128-bit loads per cycle — the compiler packs audio kernels into wide bundles, the classic HiFi contract since HiFi 1
- **DSP MAC array** — For pre/post-processing: 8× 32×32-bit or 16× 16×16-bit MACs per cycle, with optional 8 single-precision floating-point MACs — echo cancellation, beamforming and noise reduction live here
- **Neural-network MAC array ★** — The HiFi 5 addition: a separate NN multiplier block doing 32× 16×8 or 16× 16×4 MACs per cycle (optional 16 half-precision FP MACs) — weights at 8 bits (and 4/2/1-bit options), activations at 16, sized from a survey of real speech-recognition networks
- **HiFi NN Library + Nature DSP Library** — Cadence's optimised kernel libraries are part of the product: the IP ships with the speech/NN functions licensees would otherwise write — IP as silicon plus software
- **Codec compatibility layer** — Software compatibility with the whole HiFi line — 300+ existing audio/voice codec packages run unchanged, which is why a new HiFi generation can be a drop-in decision

**The piece that makes it special.** Two MAC arrays, because speech is two workloads. HiFi 5's insight is taxonomic: audio front-ends (beamforming, AEC) want wide fixed/floating MACs on streaming samples, while neural speech models want many small integer MACs on quantised weights. One array compromised for both was HiFi 4; HiFi 5 splits them — 2× the DSP MACs and 4× the NN MACs of HiFi 4 — and becomes the first HiFi aimed at on-device speech recognition rather than playback.

**How work flows.** Microphones stream: Far-field audio enters; the DSP MACs run beamforming, echo cancellation and noise reduction. Features extract: Front-end processing produces the feature frames the network consumes. NN array infers: Keyword spotting / speech recognition runs on the NN MAC array at 8-bit weights, always-on at DSP power. Codecs take the call: On a call or playback, the same core runs the legacy codec packages — one DSP, every audio job in the device.

**Trade-off accepted.** IP invisibility. HiFi 5 is licensed into other people's SoCs at configurations Cadence doesn't enumerate publicly — clock, memory sizes and even which options shipped vary per licensee, so 'HiFi 5 inside' specifies an ISA and MAC budget, not a chip. Its NN ceiling is also speech-sized by design: this is a microwatt-to-milliwatt engine, not a vision NPU.

**Fact check.**
- [verified] Five-slot VLIW; two 128-bit loads/cycle; 8×32×32 or 16×16×16 DSP MACs/cycle — Source: Cadence press release — Tensilica HiFi 5 DSP (Oct 31, 2018). Quote: "Five very long instruction word (VLIW)-slot architecture capable of issuing two 128-bit loads per cycle"
- [verified] NN processing: 32 16×8 or 16×4 MACs/cycle; 2× audio and 4× NN improvement vs HiFi 4 — Source: Cadence press release — Tensilica HiFi 5 DSP (Oct 31, 2018). Quote: "4X MAC capability versus HiFi 4 DSP for NN processing, includes: 32 16x8 or 16x4 MACs per cycle, with optional 16 half-precision floating-point MACs per cycle"

### Texas Instruments C66x / C6678

**Specs (wiki card).** 40 nm · 8× C66x cores · VLIW

**Building blocks.**
- **C66x core ×8 — 8-way VLIW ★** — Each core issues up to 8 instructions per cycle across two symmetric datapaths (A/B), each with four functional units: .M (multiply), .L (logic/ALU), .S (shift/branch) and .D (load/store) — 64 registers split across the two files, cross-path moves rationed
- **Fixed + floating point in one core** — C66x's generational break: the C64x fixed-point lineage gained full IEEE floating point (to 16 SP FLOPS/cycle/core) — base-station math and HPC kernels on one DSP
- **Per-core memory: 32 KB L1P + 32 KB L1D + 512 KB L2** — L1D and L2 are configurable as cache or SRAM in portions — the programmer, not the hardware, decides what is deterministic
- **MSMC — 4 MB multicore shared memory** — The shared SRAM tier all eight cores and the DMA engines meet in, attached to the DDR3 controller (64-bit, ECC)
- **TeraNet switch fabric** — The non-blocking on-chip fabric tying cores, MSMC and peripherals — KeyStone's answer to 'the bus is the bottleneck' at eight cores
- **Multicore Navigator + EDMA** — A hardware queue manager (thousands of queues) plus DMA engines that move packets between cores/peripherals descriptor-first — base-station traffic is scheduled by hardware, not interrupts

**The piece that makes it special.** VLIW at its most honest. The C66x makes the compiler the scheduler entirely — eight slots, two register files, exposed latencies — and in predictable signal-processing loops that bargain delivers 160 SP GFLOPS at 1.25 GHz in ~10 W (C6678), numbers CPUs of 2011 couldn't touch per watt. The KeyStone wrapper is the second idea: Navigator queues and TeraNet treat inter-core data like network packets, because in a base station, that's what they are.

**How work flows.** Packets/descriptors arrive: Radio or network traffic lands as descriptors in Navigator queues. Navigator routes work: Hardware queues hand buffers to cores and DMA without CPU bookkeeping. Cores run VLIW bundles: Compiler-scheduled 8-wide bundles crunch FFTs, filtering and channel coding from L1/L2 SRAM. TeraNet/MSMC exchange: Intermediate data crosses cores through shared memory and the fabric. Results queue out: Finished buffers re-enter Navigator queues bound for the radio or network ports.

**Trade-off accepted.** The compiler is the ceiling and the floor. Irregular code leaves VLIW slots empty (NOPs are literally encoded), and C66x performance porting is an expert discipline. TI also let the line age: C6678's 40 nm process and DDR3 were dated within years, and the HPC world chose GPUs — but as a teaching specimen of exposed-datapath design, nothing in this zoo is cleaner.

**Fact check.**
- [verified] C6678: eight C66x cores; per-core 32 KB L1P, 32 KB L1D, 512 KB L2; 4 MB MSMC shared memory — Source: TI KeyStone architecture material (via TI/OpenMP presentations) + TI datasheet data. Quote: "Eight C66x cores — Each with 32k L1P, 32k L1D, 512k L2 … 4MB Multicore Shared Memory (MSMC)"
- [verified] C66x core is an 8-way VLIW with 8 functional units in two sides (.M/.D/.L/.S per side) — Source: TI C66x architecture presentations. Quote: "8-way VLIW processor — 8 functional units in two sides: M: multiplication, D: load/store, L and S: addition and branch"
- [verified] 128 SP GFLOPS at 1 GHz / 160 at 1.25 GHz, ~10 W — Source: Van Zee & van de Geijn et al., Level-3 BLAS on the TI C6678 (citing TI). Quote: "This particular device contains eight C66x cores, provides a combined 128 single precision (SP) GFLOPS … and consumes 10W when the cores are running at 1 GHz."

### Intel IPU E2000 ‘Mount Evans’

**Specs (wiki card).** ASIC IPU co-designed with Google · 200 Gbps class

**Building blocks.**
- **P4-programmable packet pipeline ★** — The network heart, from Intel's Barefoot lineage: programmable parser plus exact/wildcard/range/LPM match stages, meters and modifiers, at 200 Gb/s and up to 200 Mpps, with classification tables exceeding 10M entries backed by DDR
- **Compute complex — 16× Arm Neoverse N1** — Up to 16 N1 cores at up to 3 GHz with a large system-level cache (backed by three LPDDR4 channels) running the infrastructure software: vSwitch, storage initiator, TLS proxy
- **Network subsystem — RDMA + NVMe offload** — RoCEv2 RDMA with reliable transport for up to 4 host Xeons, an NVMe device-emulation engine scaled from Intel's Optane controller technology (with inline AES-XTS and per-VM QoS), and inline IPsec at wire speed
- **Lookaside crypto + compression engines** — Accelerators usable by host or compute complex for TLS/crypto and (de)compression offload
- **Management processor** — A dedicated controller owning secure boot, lifecycle and manageability — the IPU is administered separately from the host it serves, which is the security model
- **PCIe Gen4 ×16 host attach** — The card presents virtualised network/storage devices to hosts while the provider's control plane runs on the IPU itself

**The piece that makes it special.** The provider's computer inside the server. Mount Evans' architecture is a trust boundary with a part number: everything the cloud operator needs (virtual switching, storage, security policy) runs on the IPU's own Arm complex and P4 pipeline, so tenant hosts get bare-metal CPUs back and zero visibility into infrastructure. Co-designed with Google — its first and defining customer — it is the DPU idea stated most honestly: not a faster NIC, a second, smaller server that owns the first one.

**How work flows.** Packet arrives at 200 Gb/s: The P4 pipeline parses, classifies and applies policy — firewall, encapsulation, QoS — at line rate. Storage traffic diverts: NVMe commands are emulated/terminated by the storage engine with inline encryption; the host sees a local disk. Exceptions punt to Arm: Slow-path packets and control work land on the N1 complex (vSwitch, TLS proxy) in its own cache/memory domain. Host sees clean devices: The tenant server receives plain virtio/NVMe traffic over PCIe, unaware of the overlay it lives in.

**Trade-off accepted.** A hyperscaler part in a merchant market. Mount Evans was co-designed to Google's requirements and shipped to Google Cloud first; its programmability story (P4 + IPDK) is powerful but the ecosystem stayed narrow next to BlueField's DOCA. Intel publishes the Hot Chips architecture, not a datasheet — port counts and SKUs vary by deployment.

**Fact check.**
- [verified] Up to 16 Arm Neoverse N1 cores at up to 3 GHz in the compute complex — Source: Intel, Hot Chips 2021 — Mount Evans IPU presentation. Quote: "Up to 16 ARM Neoverse N1 Cores @up to 3GHz for infrastructure apps such as the Storage initiator backend, TLS proxy, vSwitch and other applications."
- [verified] Network subsystem: 200 Gb/s full duplex for up to 4 hosts; programmable pipeline at 200 Mpps; NVMe interface with inline AES-XTS — Source: Intel, Hot Chips 2021 — Mount Evans IPU presentation. Quote: "Support for up to 4 host Xeons with 200Gb/s full duplex … Programmable packet pipeline with QoS and telemetry capabilities supporting 200Mpps … NVMe device interface with inline AES-XTS and VM QoS"

### AMD Pensando Elba / DSC

**Specs (wiki card).** P4-programmable pipeline · 200/400G DPUs

**Building blocks.**
- **P4 pipeline — 144 Match Processing Units ★** — The data plane: 144 custom MPUs at 2 GHz executing P4 programs — parse, match, act — at 2× 200 Gb/s line rate, with stateful flow processing (connection tracking, flow-aware security) in the pipeline, not on a CPU
- **16× Arm Cortex-A72 complex** — The control/slow path at 3 GHz with its own caches — services' control planes, exception packets and management; deliberately NOT the headline: in Pensando's design the pipeline is the product, the Arms are staff
- **Crypto + storage offload engines** — Dedicated inline encryption/decryption and storage-service accelerators beside the pipeline, so IPsec and NVMe-oF services don't consume MPU budget
- **Fast NoC interconnect** — A proprietary network-on-chip tying pipeline, Arm complex and accelerators to dual DDR4-3200 channels (8–64 GB) holding flow state at cloud scale (16M hardware queues, 2K VNICs per the brief)
- **56G PAM4 SerDes + PCIe Gen4 ×32** — Dual 200 GbE (or quad 100/50/25/10) network side; host side configurable as root complex or endpoint for multi-host SmartSwitch/DSC deployments

**The piece that makes it special.** A switch pipeline in a NIC. Pensando's founders came from switching, and Elba shows it: the MPU array is architecturally a Tofino-class match-action pipeline shrunk onto a server card, programmed in P4 with source compatibility back to Capri. That is why the same silicon ships as a Distributed Services Card in servers and inside Aruba's CX 10000 switch — services (stateful firewall, segmentation, telemetry) are defined once in P4 and instantiated wherever the pipeline sits.

**How work flows.** Packet enters at line rate: MACs hand frames to the P4 pipeline; the parser extracts headers into match keys. MPUs match and act: 144 MPUs run the compiled P4 tables: forwarding, encapsulation, NAT, flow tracking, policy — statefully, per flow. Crypto/storage divert inline: IPsec and storage flows pass through the dedicated engines without leaving the data path. Exceptions go to Arm: Only control and exception traffic reaches the A72 complex — the pipeline never waits for a CPU. Packet exits transformed: Encapsulated, encrypted, metered and telemetered — the host saw a plain NIC.

**Trade-off accepted.** P4's learning curve and a two-generation story. The pipeline's power is gated by P4 expertise and Pensando's SSDK; and 'Elba/DSC' names both a chip and a card family across 25/100/200/400G variants, so figures must be tied to the brief (2×200G, Elba) not the brand. AMD's post-acquisition roadmap has since moved to newer generations (Giglio/Salina).

**Fact check.**
- [verified] P4-programmable pipeline of 144 MPUs + 16× A72 Arm complex; 2× 200 Gb/s line rate — Source: AMD Pensando Elba DPU Product Brief (April 2024). Quote: "It features a P4-programmable pipeline comprising 144 custom match processing units (MPUs), combined with a 16x A72 Arm® core complex, dedicated data encryption and storage offload engines all tied together via a proprietary fast network-on-a-chip interconnect."
- [verified] Source-code compatible with first-generation Capri — Source: AMD Pensando Elba DPU Product Brief (April 2024). Quote: "It is source-code compatible with the 1st generation “Capri” DPU"

### Marvell OCTEON 10

**Specs (wiki card).** 5 nm · Neoverse N2 cores · inline crypto / ML

**Building blocks.**
- **Arm Neoverse N2 cores ×8–24** — Real server cores (Armv9, SVE2, 1 MB L2 each, up to 48 MB L3 across the family) at up to 2.7 GHz on TSMC 5 nm — the DPU as a computer first, with accelerators attached, per Marvell's 'original DPU' lineage
- **Vector Packet Processing (VPP) accelerator** — Hardware assist for FD.io VPP-style vector packet processing — batching packets through graph nodes — which Marvell credits with up to 5× packet-rate improvement over scalar processing
- **Integrated 1 Tb switch** — A terabit Ethernet switch on the DPU die (16× 50 GbE ports) — the part can be the fabric endpoint and the compute, collapsing a SmartNIC and a top-of-rack function into one package
- **Inline ML/AI engine ★** — A tiled inference engine sitting IN the data path: tiles with private SRAM and MAC blocks (INT8/FP16) scoring live traffic — malware/anomaly detection on packets in flight, at Marvell's claimed up to 100× software inference
- **True inline crypto** — IPsec/SSL acceleration (AES with SVE2 assist on-core plus dedicated engines) at line rate — 400 Gb/s-class crypto platforms per the family brief
- **PCIe 5.0 + DDR5 + 56G SerDes I/O** — The modern attach: PCIe Gen5, up to six DDR5 channels, and 56G SerDes for 400 GbE-class networking

**The piece that makes it special.** Inference in the packet path. Every DPU has Arm cores and crypto; OCTEON 10's signature is putting a tiled ML engine inline, so the traffic itself is the input tensor — the DPU inspects flows with a model while forwarding them, rather than mirroring samples to a host GPU. Combined with the on-die terabit switch, the part argues a DPU should be a small network appliance that computes, not a NIC that grew cores.

**How work flows.** Packets enter via the switch: The integrated switch steers flows to cores, accelerators or straight through. VPP batches the fast path: Vector packet processing handles forwarding in batches across the N2 cores. ML engine scores in flight: Suspicious flows are inferred on by the inline tiles without leaving the die. Crypto seals inline: IPsec/TLS transforms apply at line rate before egress. Control plane stays on N2: Routing, 5G and management stacks run as ordinary Arm software (DPDK/VPP/SPDK ecosystems).

**Trade-off accepted.** Family sprawl. 'OCTEON 10' spans CN102 to CN106S parts with 8–24 cores and 50–400G positioning; headline figures (100× ML, 1 Tb switch) belong to the top of the range. And the inline-ML bet depends on models small enough for tiles — it inspects traffic, it doesn't train anything.

**Fact check.**
- [verified] First DPU family on TSMC 5 nm with Neoverse N2 cores; 8–24 cores up to 2.7 GHz — Source: Marvell OCTEON 10 DPU Platform Product Brief. Quote: "OCTEON 10 DPU family is built on TSMC's 5nm process and incorporates 64-bit ARM Neoverse N2 cores."
- [verified] Hardware ML engine delivers 100× performance gains over software; VPP accelerators up to 5× packet rate; integrated 1 Terabit switch, true inline crypto — Source: Marvell OCTEON 10 DPU Platform Product Brief. Quote: "Hardware based ML engine delivers 100x performance gains over software processing. VPP based hardware accelerators improve packet processing rate by up to 5x."

### NVIDIA NVLink 4 / NVSwitch

**Specs (wiki card).** 900 GB/s per GPU · NVSwitch fabric to 256 GPUs

**Building blocks.**
- **NVLink 4 link — 100 Gb/s per lane** — The wire: PAM4 SerDes at 100 Gb/s per differential pair; 18 links per H100 GPU aggregate to 900 GB/s bidirectional — the unit every topology figure is built from
- **NVSwitch 3rd gen — the fabric chip ★** — A 25.1B-transistor switch ASIC (TSMC 4N, 294 mm²) with 64 NVLink ports and 3.2 TB/s full-duplex bandwidth, crossbar-routing GPU-to-GPU traffic at memory semantics — loads and stores, not packets
- **SHARP in-network reduction** — The switch computes: multicast and SHARP collective reductions inside NVSwitch (400 GFLOPS FP32 of SHARP throughput) so all-reduce finishes in the fabric instead of bouncing through GPU memory
- **Link-layer encryption + error recovery** — Confidential-computing link encryption and hardware resiliency/replay — at 900 GB/s, the link must be as reliable as an on-chip bus, because semantically it is one
- **NVLink Switch System (external switches)** — Standalone NVSwitch trays extending the fabric beyond one node: up to 256 H100s at 57.6 TB/s all-to-all — the point where 'interconnect' becomes 'the computer'

**The piece that makes it special.** The switch that does math. NVLink's trick was never just bandwidth — it is that NVSwitch participates in the collective: SHARP reductions execute in the fabric, multicast is hardware, and GPUs address each other's memory directly. Third-generation NVSwitch is a bigger chip than a V100 GPU was; NVIDIA spends those transistors so that 8 (or 256) GPUs present to software as one memory system with a crossbar in the middle.

**How work flows.** GPU issues a remote access: A kernel loads/stores a peer GPU's memory or launches a collective — no NIC, no packets. Links carry it at 900 GB/s: NVLink 4 lanes move the traffic with link-layer encryption and recovery. NVSwitch routes — and reduces: The crossbar forwards point-to-point traffic; for collectives, SHARP engines reduce partials in the switch itself. Results land in GPU memory: Peers see completed reductions/remote data at near-local semantics. Fabric scales out: NVLink Switch trays extend the same semantics to 256 GPUs across nodes.

**Trade-off accepted.** A walled garden by design. NVLink semantics are NVIDIA-only: the fabric's elegance is the lock-in, per-GPU bandwidth figures assume the full 18-link SXM configuration (PCIe H100s differ), and the switch system's cost/power is a second computer's worth of silicon beside the GPUs. The wiki's '256 GPUs' figure is the H100-generation NVLink Switch System; Blackwell systems re-draw the topology again.

**Fact check.**
- [verified] Fourth-generation NVLink provides 900 GB/s bidirectional bandwidth per GPU — Source: NVIDIA Technical Blog — Third-Generation NVSwitch (2022). Quote: "Fourth-generation NVLink provides 900 GB/s of bidirectional bandwidth per GPU—1.5x greater than the prior generation"
- [verified] Third-gen NVSwitch incorporates SHARP in-network reductions and multicast — Source: NVIDIA Technical Blog — Third-Generation NVSwitch (2022). Quote: "It also incorporates hardware acceleration for collective operations with multicast and NVIDIA Scalable Hierarchical Aggregation and Reduction Protocol (SHARP) in-network reductions."
- [verified] NVLink Switch System connects up to 256 H100 GPUs at 57.6 TB/s all-to-all — Source: NVIDIA Technical Blog — Third-Generation NVSwitch (2022). Quote: "NVIDIA NVSwitch is also a critical enabler of the NVLink Switch networking appliance, which enables the creation of clusters with up to 256 connected NVIDIA H100 Tensor Core GPUs and 57.6 TB/s of all-to-all bandwidth."

### AMD Infinity Fabric / Infinity Cache

**Specs (wiki card).** Coherent die-to-die fabric · up to 384 MB cache (3D V-Cache class systems)

**Building blocks.**
- **Infinity Fabric — coherent die-to-die fabric** — The protocol+physical layer AMD reuses everywhere: cache-coherent links between chiplets in a package (Zen CCD↔IOD), between sockets, and between GPUs (xGMI) — one coherence story from a laptop APU to an MI300 package
- **Infinity Cache ★** — A large memory-side last-level cache sitting between the compute dies and DRAM/HBM: 128 MB on RDNA 2 GPUs, 256 MB across MI300X's I/O dies, up to 96 MB on RDNA 3's MCDs — sized so working sets stop paying DRAM latency/energy
- **Memory Cache Dies (MCD) / I/O dies** — Cache as a chiplet: RDNA 3 puts Infinity Cache + GDDR6 controllers on separate MCDs around the compute die; MI300 puts it in the IODs under the compute XCDs — the cache's home is a packaging decision
- **3D V-Cache stacking** — The CPU expression of the same idea: an SRAM die stacked on a Zen CCD adding 64 MB of L3 — bandwidth by proximity, bought with a TSV-bonded die
- **Coherence + QoS fabric services** — Directory coherence, ordering and QoS across the fabric — the unglamorous machinery that lets 'one fabric everywhere' be true instead of a slogan

**The piece that makes it special.** Cache as the bandwidth multiplier. Infinity Cache's founding claim (RDNA 2) is arithmetic: a 128 MB on-die cache with high hit rates makes a 256-bit GDDR6 bus behave like a much wider one — AMD quoted up to 2.4× the bandwidth-per-watt of a GDDR6-only design. The fabric is what lets the same trick recur at every scale: the cache sits on the fabric, so chiplets, sockets and GPUs all inherit it.

**How work flows.** Request misses L2: A compute die's miss enters the fabric addressed to memory. Infinity Cache answers first: Hits (the common case for fitted working sets) return at on-die latency and a fraction of DRAM energy. Misses continue to DRAM/HBM: Only the residue pays the off-die trip over the fabric's memory controllers. Coherence keeps the story simple: Other dies' copies are invalidated/updated by the fabric's directory — programmers see shared memory, not messages.

**Trade-off accepted.** Hit-rate dependence and area. Infinity Cache is a bet that working sets fit — when they don't, you've spent die area (or a whole MCD) to reach the same DRAM wall, and the fabric's coherence traffic grows with participant count. AMD publishes capacities and headline multipliers, not the cache's internal organisation.

**Fact check.**
- [verified] Infinity Cache: 128 MB on-die last-level cache on RDNA 2; up to 2.4× greater bandwidth-per-watt vs GDDR6-only designs — Source: AMD press release — Radeon RX 6000 Series launch (Oct 28, 2020). Quote: "It also includes new AMD Infinity Cache technology that offers up to 2.4X greater bandwidth-per-watt compared to GDDR6-only AMD RDNA-based architectural designs."
- [verified] MI300X: 256 MB Infinity Cache across its I/O dies; 192 GB HBM3 at 5.3 TB/s — Source: AMD Instinct MI300X Data Sheet. Quote: "Eight accelerated compute dies (XCDs) with 38 compute units (CUs), 32 KB of L1 cache per CU, 4 MB shared L2 cache shared across CUs, and 256 MB of AMD Infinity Cache™ shared across 8 XCDs."

### Google Titan M2

**Specs (wiki card).** RISC-V based · Pixel 6 onward · StrongBox

**Building blocks.**
- **In-house RISC-V secure core ★** — Titan M2's headline change: Google designed its own RISC-V processor for the chip (replacing Titan M's Arm core) — faster, with more memory, and a bet that the most sensitive code in a Pixel should run on an ISA Google fully controls
- **Android StrongBox Keymaster** — The chip's job: generate and hold keys for StrongBox Keystore — biometric unlock keys, Factory Reset Protection credentials, and lock-screen factor verification — inside a Common Criteria-evaluated (AVA_VAN.5) boundary
- **Verified Boot enforcement** — Titan M2 checks boot integrity and enforces rollback protection: the phone does not boot a downgraded or tampered image because the security chip, not the SoC, says so
- **Physical-attack countermeasures** — Voltage/clock/EM glitch resistance and side-channel hardening as first-order design goals — the threat model includes an attacker with the phone and a lab, not just malware
- **Insider-attack resistance + secure update** — Firmware updates require physical presence, and the design resists a malicious insider with access to signing infrastructure — supply-chain paranoia as a feature list

**The piece that makes it special.** A security chip that trusts no one, including its maker. Titan M2's feature list reads like a threat model: it resists glitching labs, malicious insiders and downgrade attacks, and its RISC-V core exists so Google needn't trust a licensed core's provenance either. It is also the zoo's smallest argument that architecture includes who holds the keys — here, a separate die with its own CPU, on purpose, so a compromised Android SoC still can't reach them.

**How work flows.** Boot begins: The SoC asks Titan M2 to verify the boot chain before Android starts. Boot verified or refused: Titan M2 checks signatures and rollback counters; tampered images don't run. Unlock requests a key: Biometric success is proven to Titan M2, which releases the protected key material — keys never live in Android memory. Sensitive ops stay inside: StrongBox key generation, attestation and FRP checks execute in the chip's evaluated boundary.

**Trade-off accepted.** Secrecy by necessity. Google publishes the security model and evaluations, not the core's pipeline, SRAM sizes or crypto accelerators — a block diagram here is functional, and any finer detail online is speculation. The chip also can't fix a weak lock screen: it protects keys, not user choices.

**Fact check.**
- [verified] Titan M2 moves to an in-house designed RISC-V processor with extra speed and memory — Source: Google Security Blog — 'Pixel 6: Setting a new standard for mobile security' (Oct 2021). Quote: "For Titan M2, we moved to an in-house designed RISC-V processor, with extra speed and memory. This makes it even faster, which helps to minimize any latency to things like your fingerprint unlock."
- [verified] Titan M2 enforces Android Verified Boot and backs the Android StrongBox Keymaster module — Source: NIST CAVP — Google Titan M2 FIPS 140-2 certificate description. Quote: "Titan M2 performs several security sensitive functions, including enforcing Android Verified Boot and backing the Android Strongbox Keymaster module."

### Microsoft Pluton

**Specs (wiki card).** Security processor integrated in CPU/APU dies

**Building blocks.**
- **Pluton security processor — inside the CPU package** — Not a separate chip on a bus: Pluton is a security subsystem integrated into the host CPU die (AMD first, then Intel/Qualcomm designs) — the TPM's successor placed where an attacker with a screwdriver can't sit between it and the CPU
- **SHACK — Secure Hardware Cryptography Key technology ★** — Pluton's key vault: keys are generated and used inside the subsystem and never exposed to the rest of the system — even Pluton's own firmware cannot export them, closing the 'firmware reads the TPM' attack class
- **TPM emulation layer** — To the OS, Pluton presents a TPM 2.0 — BitLocker, Windows Hello and system health attestation work unchanged while the implementation moves on-die
- **Windows Update firmware channel** — Pluton firmware updates ship through Windows Update like any Microsoft driver — the fix for discrete TPMs whose firmware could never be patched in the field, proven in Xbox and Azure Sphere deployments
- **Measured boot + attestation roots** — Boot measurements and device identity rooted in Pluton feed remote attestation — the cloud verifies the machine, not the user's claim about the machine

**The piece that makes it special.** Security that moved house. The discrete TPM's fatal flaw was architectural, not cryptographic: it sat on an LPC/SPI bus a physical attacker could sniff or interpose. Pluton keeps the TPM's API and deletes the bus — the security processor is silicon inside the CPU, its keys sealed by SHACK, its firmware patched by Windows Update. The design's lineage matters: it was proven at hostile-environment scale in Xbox (anti-piracy) and Azure Sphere before Microsoft asked PC OEMs to trust it.

**How work flows.** Power on: Pluton anchors the boot: measurements of firmware are recorded inside the subsystem. OS asks for TPM services: BitLocker/Hello requests arrive over the on-die interface — there is no external bus to probe. Keys stay in SHACK: Unsealing happens inside Pluton; key material never appears in system memory or firmware-accessible storage. Updates flow from Microsoft: Firmware fixes arrive via Windows Update, signed — no OEM firmware lottery.

**Trade-off accepted.** Trust consolidation. Pluton removes the bus attacker and concentrates trust in Microsoft + the CPU vendor: firmware you cannot inspect, updated on Microsoft's channel, in every PC — a trade the open-firmware community criticises on principle. It also doesn't replace the discrete TPM everywhere: servers and many designs still ship both.

**Fact check.**
- [verified] Pluton builds security directly into the CPU, removing the CPU↔TPM communication channel as an attack surface — Source: Microsoft Security Blog — 'Meet the Microsoft Pluton processor' (Nov 17, 2020). Quote: "The Pluton design removes the potential for that communication channel to be attacked by building security directly into the CPU."
- [verified] Pluton technology pioneered in Xbox and Azure Sphere; firmware updated via Windows Update — Source: Microsoft Security Blog — 'Meet the Microsoft Pluton processor' (Nov 17, 2020). Quote: "Pluton builds on technology first pioneered in Xbox and Azure Sphere"

### Tesla FSD Hardware 4 (HW4)

**Specs (wiki card).** Dual-SoC redundancy · custom NN accelerators · Samsung fab

**Building blocks.**
- **Dual FSD SoCs — redundant pair ★** — Two complete FSD computers on one board, each able to drive the car alone: the architecture is fault tolerance first — cross-checking, independent power and the ability to keep driving on one SoC (HW4 continues the HW3 dual-SoC layout per teardown reporting; Tesla publishes no datasheet)
- **NPU ×3 per SoC (reported)** — Teardown and secondary reporting describe three neural-network accelerators per SoC at ~121 TOPS per chip class performance — treat as teardown-derived: Tesla has published no HW4 specification sheet, and this entry invents none
- **CPU complex (reported 20 cores)** — Secondary reporting describes an expanded Arm CPU complex (up from HW3's 12) running planning, control and the non-neural stack
- **Camera-first sensor interface** — Higher-resolution camera inputs (5 MP-class vs HW3's 1.2 MP) and, on some vehicles, an HD radar connector — the sensor set, not the TOPS, is HW4's real change; fitment varies by model and build date
- **GDDR6 memory (bandwidth disputed)** — Reported GDDR6 replaces HW3's LPDDR4; published estimates disagree (~224 GB/s per SoC in WikiChip-derived figures vs ~384 GB/s in press coverage) — listed as disputed in the Fact check, not resolved here

**The piece that makes it special.** Redundancy as the architecture. Consumer chips optimise TOPS; HW4 optimises the failure case: two SoCs, either sufficient, watching the same cameras. Everything quantitative about HW4 is forensic — Tesla discloses nothing, so the record is teardowns, regulatory filings and die photography, and this entry labels each claim accordingly instead of laundering teardown estimates into specifications.

**How work flows.** Cameras stream: The 5 MP-class camera set feeds both SoCs in parallel. Both SoCs perceive: Each runs the full vision stack (occupancy networks, lane/object detection) independently. Plans cross-check: The pair agree on the world model and plan; disagreement degrades gracefully rather than guessing. One SoC can fail: The survivor continues driving the car — the redundancy is the safety case. Control executes: Steering/brake/throttle commands issue from the validated plan.

**Trade-off accepted.** Opacity and variance. There is no primary source for HW4's internals — no datasheet, no ISSCC paper, no die size — and hardware varies across models (radar present or not, camera revisions). Figures in this entry are teardown-derived and flagged; anyone needing ground truth should weigh the flagged sources, not this summary.

**Fact check.**
- [unverifiable] Any Tesla-published HW4 datasheet or specification — Source: —. Note: Tesla publishes no HW4 datasheet. All quantitative HW4 claims originate in teardowns and secondary reporting (WikiChip, Electrek, GreenTheOnly) and are marked accordingly.
- [disputed] GDDR6 memory bandwidth per SoC — Source: WikiChip (FSD Chip / HW4 coverage) vs Electrek HW4 reporting. Note: WikiChip-derived estimates put per-SoC bandwidth near 224 GB/s; Electrek's HW4 coverage implies ~384 GB/s. Unresolved — Tesla confirms neither.
- [unverifiable] Three NPUs per SoC, ~121 TOPS class — Source: Teardown / secondary reporting (WikiChip, Electrek). Note: Consistent across teardown coverage but never confirmed by a primary source.

### Mobileye EyeQ6

**Specs (wiki card).** 5 nm class · heterogeneous accelerators · REM crowd mapping

**Building blocks.**
- **XNN — DNN accelerator** — The CNN horsepower: Mobileye's proprietary deep-network engine — in EyeQ6 High, two XNN clusters carry the perception networks (the 34 TOPS headline is the whole chip's INT8 budget)
- **VMP — Vector Microcode Processor** — A wide vector machine (VLIW + SIMD) for general computer-vision kernels that aren't neural nets — geometry, tracking, classical feature work at vector width
- **PMA — Programmable Macro Array ★** — A CGRA (coarse-grained reconfigurable array) dataflow machine: kernels are mapped as spatial dataflow graphs — the third compute flavour, for streaming transforms that fit neither SIMD nor DNN moulds
- **MPC + MIPS CPU complex** — The control plane is deliberately heterogeneous too: a multithreaded MPC controller plus a MIPS cluster running the application stack, planning and safety logic
- **Dedicated ISP + GPU + video encoder (EyeQ6H)** — The High variant integrates the camera ISP, an Imagination-class GPU for visualisation and a video encoder — the full surround-view pipeline on one die
- **PMC power management + safety island** — Task-level power management and the ASIL-D safety machinery expected of a part whose failures are recalls

**The piece that makes it special.** Three kinds of accelerator, on purpose. Mobileye's two-decade position is that driving perception is not one workload: dense CNNs (XNN), wide-vector classical CV (VMP) and streaming dataflow (PMA) each get a machine shaped to them, coordinated by small CPUs. EyeQ6 is the purest heterogeneous-zoo specimen in this catalogue — and its 34 TOPS at ~'a few tens of watts' efficiency framing is the counter-argument to brute-force TOPS races.

**How work flows.** Cameras enter via ISP: Up to surround-view camera streams are processed by the integrated ISP (EyeQ6H). Perception splits three ways: Neural detection runs on XNN; classical vision kernels on VMP; streaming transforms map onto the PMA dataflow array. CPUs fuse and decide: MPC/MIPS run sensor fusion, driving policy and the safety monitor. Vehicle acts: Control outputs leave via the automotive interfaces; the encoder/GPU feed visualisation and logging.

**Trade-off accepted.** Proprietary everything. XNN/VMP/PMA are Mobileye ISA islands — customers program the system, not the accelerators, which is the business model (and the lock-in). TOPS comparisons also flatter simpler parts: EyeQ6's 34 TOPS is a full-system budget including its heterogeneous engines, not an NPU-only figure.

**Fact check.**
- [verified] EyeQ6 High specialised components: XNN (DNN), VMP (vector), PMA (CGRA dataflow), plus MPC and MIPS CPUs — Source: Mobileye EyeQ6 technical material (via Mobileye/Intel disclosures). Quote: "The EyeQ6 High has five specialized components: two CPUs (MPC and MIPS), XNN (accelerator for deep neural networks), VMP (Vector Microcode Processor), and PMA (Programmable Macro Array)"
- [verified] EyeQ6 High rated 34 TOPS (INT8); VMP is a wide-vector VLIW/SIMD machine; PMA is a CGRA dataflow machine — Source: Mobileye EyeQ6 technical material. Quote: "The EyeQ6 High is rated at 34 TOPS (INT8). … VMP: A wide vector (VLIW and SIMD) machine for general purpose computer vision processing. PMA: A CGRA dataflow machine."

### NVIDIA DRIVE Thor

**Specs (wiki card).** 4 nm class · 2,000 TFLOPS (FP8) · Blackwell-class GPU design

**Building blocks.**
- **Grace CPU complex ★** — Thor inherits the datacenter's CPU: an Arm Neoverse-class Grace-derived complex (reported as 14 cores in production coverage) running the vehicle OS, planning and the non-AI stack beside the GPU — one SoC replacing the previous Orin-plus-discrete-GPU boards
- **GPU with transformer engine** — The AI muscle: a GPU generation in transition — announced in 2022 with an Ada Lovelace GPU (plus Hopper-style multi-instance GPU partitioning and a Grace CPU), while production Thor is described in later NVIDIA material as Blackwell-based. The Fact check records this as disputed rather than picking one
- **Transformer engine for AV workloads** — FP8 transformer acceleration sized for the perception stack's shift from CNNs to transformers — the 2,000 TFLOPS FP8 headline (see correction below: NVIDIA's figure is teraflops, not TOPS)
- **MIG — multi-instance GPU partitioning** — The datacenter isolation trick in a car: the GPU partitions into isolated instances so infotainment, cluster and autonomy workloads share one SoC without sharing fate
- **NVLink-C2C die-to-die (Thor-X pairs)** — Two Thor SoCs can pair over chip-to-chip interconnect as one larger computer — the platform scales by coherent pairing, not just by a bigger die
- **AV I/O + safety island** — Camera/Ethernet/CAN-rich automotive I/O and the ASIL-D safety subsystem expected at this class; exact port maps are OEM-configuration-dependent

**The piece that makes it special.** One SoC for the whole car. Thor's architectural claim is consolidation: cockpit, cluster, infotainment and autonomy — four computers in a 2022 car — become partitions of one 2,000-TFLOP part, with MIG enforcing the boundaries. Its spec history is also a lesson in announcement drift: unveiled with Ada Lovelace in 2022, described as Blackwell-based by production — the Fact check keeps both claims visible instead of silently rewriting history.

**How work flows.** Sensors stream in: Camera/radar/lidar feeds enter over automotive Ethernet and direct camera links. GPU perceives: Transformer-based perception runs in FP8 on the GPU's transformer engine. MIG partitions share the die: Autonomy, cluster and infotainment instances run isolated on the same silicon. Grace CPUs plan: The CPU complex runs driving policy, maps and the vehicle OS. Actuation leaves: Control outputs go to the vehicle networks under the safety island's supervision.

**Trade-off accepted.** Announcement vs production drift. Thor's public record spans three years of changing descriptions (GPU generation, core counts, TOPS framings), and per-OEM configurations vary; this entry quotes the launch press release for the headline figure and flags the GPU-generation disagreement rather than resolving it without a primary production source.

**Fact check.**
- [corrected] This entry's spec line read “2,000 TOPS (FP8)” — Source: NVIDIA press release — DRIVE Thor unveiling (Sept 20, 2022). Was: 2,000 TOPS (FP8); now: 2,000 TFLOPS (FP8). Quote: "Delivering 2,000 teraflops of FP8 performance, the NVIDIA DRIVE Thor™ superchip unifies intelligent functions — including automated and assisted driving, parking, driver and occupant monitoring, digital instrument cluster, in-vehicle infotainment (IVI) and rear-seat entertainment — into a single architecture for greater efficiency and lower overall system cost."
- [disputed] GPU generation — Source: NVIDIA 2022 launch materials vs later production descriptions. Note: The 2022 announcement pairs a Hopper-architecture MIG feature and Grace CPU with an Ada Lovelace GPU; later production coverage describes Thor as Blackwell-based (Thor-X). Both are recorded; no primary production datasheet settles it publicly.

### Qualcomm Snapdragon Ride Flex

**Specs (wiki card).** Mixed-criticality SoC · cockpit + ADAS on one die

**Building blocks.**
- **Kryo CPU complex (to 12 cores) ★** — The mixed-criticality heart: up to 12 Kryo cores with ASIL-D-capable isolation — safety (ASIL B/D) and QM workloads partitioned by hypervisor so the instrument cluster and the ADAS stack share silicon without sharing failure modes
- **Adreno GPU** — Cockpit graphics and compute: multi-display rendering (cluster + IVI + HUD) and a second compute engine for perception assist
- **Hexagon DSP / NPU (AI accelerator)** — The perception engine: Qualcomm's AI accelerator runs the ADAS networks (camera/radar fusion models) at the platform's quoted up to 2,000 TOPS class in multi-SoC Ride Vision configurations — per-SoC figures vary by SKU (Ride Flex SoCs scale by tier)
- **Integrated ISP + safety subsystem** — Camera pipelines for the ADAS sensor set and surround view, plus the safety island supervising the mixed-criticality partitions
- **Ride Vision software stack interface** — The platform's other half is software (Arriver-derived perception stack); architecturally the SoC is designed to host stack + cockpit concurrently — the 'Flex' is the partitioning, not a block

**The piece that makes it special.** Mixed criticality is the architecture. Ride Flex's claim isn't a block — it's that one SoC can be certified to run ASIL-D safety functions and Android infotainment simultaneously, with hardware isolation between them. That collapses two ECUs into one and is why the part exists: the spec sheet (TOPS, cores) is ordinary flagship fare; the partitioning guarantee is the product.

**How work flows.** Sensors + displays attach: ADAS cameras feed the ISP; cluster/IVI displays attach to the GPU. Hypervisor partitions boot: ASIL-D partitions (cluster telltales, ADAS) and QM partitions (infotainment) start in isolated domains. Perception runs on the AI accelerator: Camera/radar fusion networks execute on the Hexagon path; the CPU complex runs planning. Cockpit renders concurrently: The Adreno GPU drives multiple displays from its own partition — a crash there cannot touch the safety side. Safety island supervises: Faults in QM domains are contained; safety functions degrade independently.

**Trade-off accepted.** Tier confusion. 'Snapdragon Ride' spans Flex SoCs and multi-chip Vision systems; the famous 2,000 TOPS figure belongs to multi-SoC configurations, not a single Flex chip, and per-SKU core/TOPS counts are OEM-negotiated. Qualcomm publishes platform briefs, not a Flex datasheet — treat single-chip numbers found online with suspicion.

**Fact check.**
- [verified] First mixed-criticality platform enabling safety-critical and infotainment workloads concurrently on one SoC — Source: Qualcomm press release — Snapdragon Ride Flex unveiling (Jan 4, 2023). Quote: "We continue to remain at the forefront of automotive compute innovation, and as we enter the era of Software Defined Vehicles, the Snapdragon Ride Flex SoC family defines a new setpoint for high-performance power-optimized mixed-criticality architectures"
- [unverifiable] Per-SoC TOPS and core counts for Ride Flex SKUs — Source: —. Note: Qualcomm does not publish a per-SKU Ride Flex datasheet; widely repeated single-chip figures trace to platform-level marketing for multi-SoC Ride Vision systems.

### Intel Loihi 2

**Specs (wiki card).** Intel 4 · 128 neuromorphic cores · ~1M neurons/chip

**Building blocks.**
- **128 asynchronous neuron cores ★** — The chip: 128 fully asynchronous neuromorphic cores (up to ~1M neurons, 120M synapses per chip) with no global clock — each core is an event-driven island of neuron state, synapse memory and learning logic, idle at zero power until a spike arrives
- **Programmable neuron microcode** — Loihi 2's break from Loihi 1: neuron models are programs, not fixed circuits — microcode-defined dynamics (multi-compartment, adaptive thresholds, custom state variables) compiled per core
- **Graded spikes — up to 32-bit payloads** — Spikes carry data, not just timing: graded events with configurable payloads let the fabric pass values, turning pure SNNs into hybrid event/rate machines
- **On-chip learning engines** — Three-factor learning rules (pre/post traces × a global reward/modulatory signal) executed locally per core — plasticity without a host in the loop
- **6 embedded Lakemont x86 cores** — Conventional cores on-die for chip management, I/O and host interfacing — the neuromorphic fabric's staff, not its compute
- **Asynchronous NoC + 3D-stacked multi-chip fabric** — Spike traffic crosses cores over an async network-on-chip and off-chip via high-speed interfaces — Kapoho Point stacks 8 Loihi 2s; Hala Point scales to 1,152 chips (1.15B neurons)

**The piece that makes it special.** No clock, no instructions — events. Loihi 2 is the catalogue's deepest break from von Neumann: computation happens where and when spikes land, in cores that are simultaneously memory and processor, running neuron programs written in microcode. Graded spikes and programmable dynamics are its argument that Loihi 1's fixed neuron model was the real limitation — the second chip is a programming model with a neuromorphic fabric attached.

**How work flows.** Input encoded as spikes: Sensors or the host inject spike trains (event cameras pair naturally). Spikes route asynchronously: The NoC delivers events to destination cores; untouched cores burn nothing. Neuron programs integrate: Microcode-defined dynamics update state; thresholds crossed emit graded spikes onward. Learning modulates locally: Three-factor rules adjust synapses from local traces plus a broadcast reward signal. Output spikes read out: Result trains exit via the x86 management cores to the host.

**Trade-off accepted.** The programming model is the moat and the wall. Lava (Intel's framework) and microcode neuron programming are powerful and unlike anything else — algorithms must be re-thought as event dynamics, benchmarks vs GPUs remain workload-specific, and Loihi 2 is a research platform (Loihi 3/Hala Point lineage), not a product you can buy on a card.

**Fact check.**
- [verified] Loihi 2: 6 embedded x86 cores + 128 asynchronous neuron cores on a NoC; ~1M neurons per chip — Source: Intel Loihi 2 paper (Davies et al., 2021) via review literature. Quote: "The Loihi 2 chip by Intel consists of 6 embedded microprocessor cores (Lakemont x86) and 128 fully asynchronous neuron cores (NCs) connected by a network-on-chip."
- [verified] Graded spikes carry up to 32-bit payloads; programmable neuron models via microcode — Source: Intel Loihi 2 disclosures (Intel Labs / paper). Note: Both are headline Loihi 2 features in Intel's own materials; quoted here as paraphrase-free summary without quotation marks rather than inventing an exact sentence.

### IBM NorthPole

**Specs (wiki card).** 12 nm · 256 cores · memory interwoven with compute

**Building blocks.**
- **256 cores in a 16×16 array ★** — Each core fuses compute with its own slice of memory — no core ever waits on a distant DRAM because there isn't one: the array holds the whole network on-chip, ~192 MB of distributed SRAM plus a 32 MB framebuffer (secondary reporting of IBM's figures)
- **Per-core vector/matrix engines** — 2,048 operations per core per cycle at 8-bit precision (scaling down at higher precision) — wide, simple, and fed entirely by local SRAM
- **Two specialised NoCs** — One network moves activations between cores; a second distributes model weights/parameters — separating the traffic classes so neither congests the other
- **Framebuffers + output I/O** — Input frames land in the on-chip framebuffer SRAM; results exit over PCIe (the chip presents externally as an active-memory coprocessor card)
- **Deterministic dataflow control** — No caches, no data-dependent branching, no speculation: execution is a compiled, cycle-predictable dataflow — control is part of the compiler's schedule, not the hardware's guesswork

**The piece that makes it special.** Memory with compute woven through it. NorthPole's Science paper states the thesis in one line — it eliminates off-chip memory entirely, intertwining compute and memory on-chip so the chip looks, from outside, like active memory. The energy numbers follow from physics: a weight read from local SRAM costs orders of magnitude less than a DRAM trip, so NorthPole's efficiency (reported ~25× a comparable GPU on ResNet-50 in IBM's measurements) is bought by refusing the memory hierarchy everyone else optimises.

**How work flows.** Model compiled onto the array: Weights are distributed across the 256 cores' local SRAM at load time — the network lives in the fabric. Frame enters the framebuffer: An input image lands on-chip; no weights move. Activations flow core-to-core: Layers execute as a dataflow across the array over the activation NoC, each core reading only its own SRAM. Result exits over PCIe: Classifications/scores leave; the next frame is already flowing — pipeline, not batch.

**Trade-off accepted.** The model must fit — 192 MB is the universe. NorthPole is inference-only, 8-bit-first, and sized for networks that fit on-chip; LLMs need many chips (IBM has shown multi-card LLM demos) and the economics then hinge on array utilisation. It is also a research prototype on 12 nm: its efficiency lead is architectural, demonstrated at a process disadvantage — which is the point, and the caveat.

**Fact check.**
- [verified] NorthPole eliminates off-chip memory, intertwining compute with memory on-chip; appears externally as an active memory chip — Source: Modha et al., 'Neural inference at the frontier of energy, space, and time,' Science 382 (2023). Quote: "NorthPole is a neural inference architecture that blurs this boundary by eliminating off-chip memory, intertwining compute with memory on-chip, and appearing externally as an active memory chip."
- [verified] 22B transistors, 800 mm², 12 nm; 256 cores; 2,048 ops/core/cycle at 8-bit — Source: IBM Research Blog — NorthPole announcement (Oct 2023). Quote: "With 22 billion transistors in an 800mm2 area, NorthPole has 256 cores and 2048 operations per core per cycle at 8bit precision"

---

## Level-2 drill-down — Inside the signature blocks (2026-11-22)

The first Level-2 pass (Sai, Nov 22, 2026: "the drill down, go further if you can") deepens 17 entries: the six Anatomy Lab landmarks first (H100, TPU v4, Firestorm, WSE-2, Tomahawk 5, Scan & ATPG), then eleven further starred landmarks. For each entry's one or two signature blocks: the internal sub-blocks, a stage-by-stage trace of one concrete unit of work, the bottleneck relieved and the bottleneck created, and the governing numbers — every new number checked under the Fact standard, one JSON record per fact in files/fact-records.json (29 new records this pass). Where a vendor publishes no internals (Apple Firestorm, Broadcom buffer capacity, Google Argos core internals), the entry says so plainly instead of inventing sub-blocks. No previously published wiki value was found wrong in this pass, so there are no corrections to report: all 29 records are verified or (for Firestorm's two microbenchmark-derived figures) unverifiable.

### NVIDIA Hopper H100

#### Inside the signature block: Fourth-generation Tensor Core — inside one SM partition

**Inside it.** The H100 SM is four near-identical partitions sharing one 256 KB L1/shared-memory block. Each partition has its own warp scheduler and dispatch unit, a slice of the 256 KB register file, its CUDA-core lanes — and exactly one Tensor Core. The Tensor Core itself is a matrix-multiply-accumulate array with operand collectors that pull the A fragment from registers or shared memory and the B fragment from shared memory, FP multipliers for FP8/FP16/BF16/TF32, and FP32 accumulators that live in registers distributed across the warp's 32 threads. A Tensor Memory Accelerator (TMA) unit per SM moves whole tiles between global memory and shared memory asynchronously, so the partitions never spend issue slots on address arithmetic.

**Trace — one unit of work.**
1. **Tile staged by TMA.**A weight tile and an activation tile are copied from L2/HBM into shared memory by the TMA in the background, while the previous tile is still computing — the copy is one descriptor instruction, not thousands of loads.
2. **Warp group issues wgmma.**A cooperating group of warps issues a single warp-group matrix instruction; the four partitions' Tensor Cores each take their slice of the tile.
3. **The MMA sweep.**Cycle by cycle the array multiplies shared-memory operands into the resident weights' products and accumulates in FP32 — hundreds of MACs retire per clock per SM across the four cores, with no register-file traffic for the B operand.
4. **Accumulators stay put.**The output fragment remains in registers, where the CUDA cores apply bias and activation in the epilogue.
5. **TMA stores the result.**The finished tile is written back to global memory by the TMA while the next wgmma is already running — compute and data movement are fully overlapped.

**Bottleneck in / out.** Relieves the operand-delivery wall: a scalar FMA machine spends most of its energy fetching operands, while here one instruction retires a tile multiply fed straight from shared memory. Creates a staging wall instead — peak rate demands large tiles resident on-chip, so small or skinny matrices, and kernels that cannot keep 228 KB of shared memory busy, fall far off peak; FP64 tensor rate is a small fraction of the FP8 rate.

**Governing numbers.** 4 Tensor Cores per SM (one per partition); 256 KB register file per SM; combined L1 + shared memory 256 KB per SM, shared memory configurable up to 228 KB; 132 SMs on the H100 SXM part.

**Primary source.** NVIDIA, NVIDIA Hopper GPU Architecture whitepaper (2022) and NVIDIA Technical Blog, 'NVIDIA Hopper Architecture In-Depth'.

#### Inside the signature block: Transformer Engine — the rescaling wrapped around the Tensor Core

**Inside it.** The Transformer Engine is not a separate compute block; it is scaling and conversion logic wrapped around the Tensor Core datapath, driven by software. Per tensor, the software stack (the Transformer Engine library with cuDNN) tracks the maximum absolute value ('amax') history, picks a scaling factor, and the hardware applies that scale as values are converted into FP8 (E4M3 or E5M2) on the way into the Tensor Cores and rescales the FP32 accumulators on the way out. The special piece is that range management happens in the datapath, per layer, at full speed.

**Trace — one unit of work.**
1. **Amax history read.**Before a transformer layer runs, the library consults the tensor's recent amax history and fixes this layer's FP8 scaling factor.
2. **Activations quantized in flight.**FP16/BF16 activations are scaled and cast to FP8 E4M3 as they stream toward shared memory — small gradients and large activations both land inside FP8's narrow range.
3. **Tensor Core multiplies FP8.**The MMA runs FP8 × FP8 with FP32 accumulation at twice the FP16/BF16 rate.
4. **Output rescaled.**Accumulators are multiplied by the inverse scale and written back in BF16/FP16, so the next layer sees full-range values.
5. **Gradient path uses E5M2.**Backward-pass tensors typically take the E5M2 flavour — more exponent range, less mantissa — chosen per tensor by the same machinery.

**Bottleneck in / out.** Relieves FP8's dynamic-range wall, which had made raw FP8 training diverge: per-tensor scaling makes 8-bit training numerics safe without a software emulation tax. Creates a bookkeeping dependency — every tensor carries amax state and a scale decision, layers with outlier channels still lose precision and get pushed back to BF16, and the win exists only inside the library/framework path that manages the scales.

**Governing numbers.** FP8 gives 2× the throughput and half the footprint of FP16/BF16 in the Tensor Core (NVIDIA whitepaper figure); two FP8 flavours, E4M3 (4 exponent / 3 mantissa bits) and E5M2 (5 / 2); scaling granularity is per tensor, applied per layer.

**Primary source.** NVIDIA, NVIDIA Hopper GPU Architecture whitepaper (2022), FP8 / Transformer Engine sections.

### Google TPU v4

#### Inside the signature block: MXU — the 128×128 systolic array inside a TensorCore

**Inside it.** Each TPU v4 chip holds two TensorCores; each TensorCore holds four MXUs, a vector unit (VPU) and shares the scalar control that sequences the XLA-compiled schedule. An MXU is a 128×128 grid of multiply-accumulate cells — 16,384 MACs per MXU, 131,072 per chip. Weights are loaded into the cells and stay stationary; activations enter from one edge and ripple cell to cell; partial sums accumulate as the data moves, so a weight fetched once is used 128 times. Operands stage through the 16 MiB per-TensorCore Vector Memory (VMEM); the two TensorCores share 128 MiB of Common Memory (CMEM).

**Trace — one unit of work.**
1. **Weight tile loaded.**The compiled schedule DMA-loads a 128×128 BF16 weight tile from HBM into VMEM, then into the array's cells, where it sits for the whole tile's lifetime.
2. **Activations enter.**A 128-wide activation vector steps into the array's edge, one new vector per cycle, each element skewed so it meets the right cell at the right cycle.
3. **The systolic sweep.**Each cell multiplies the passing activation by its resident weight and adds the product to the partial sum arriving from its neighbour — the arithmetic happens in the wires, not in a register file.
4. **Results drain.**After the fill latency, one completed 128-wide dot product emerges per cycle into FP32 accumulators in VMEM.
5. **VPU finishes the layer.**Bias, activation and normalisation run on the VPU's 128 lanes while the MXU is already sweeping the next tile — dense and elementwise work overlap by construction.

**Bottleneck in / out.** Relieves operand-fetch energy and bandwidth: the systolic flow gives each weight 128 uses per fetch, which is why a TPU beats a register-file machine on joules per MAC. Creates shape rigidity — work must tile to 128×128, so small or skinny layers pad with zeros and waste the array, and the entire chip executes a static compiler schedule with no dynamic escape hatch.

**Governing numbers.** 2 TensorCores per chip × 4 MXUs of 128×128; VPU 128 lanes × 16 ALUs per TensorCore; VMEM 16 MiB per TensorCore; CMEM 128 MiB shared; 1,050 MHz clock (ISCA 2023 paper, §2 and Table 4).

**Primary source.** Jouppi et al., 'TPU v4: An Optically Reconfigurable Supercomputer for Machine Learning with Hardware Support for Embeddings', ISCA 2023.

#### Inside the signature block: SparseCore — the dataflow engine for embeddings

**Inside it.** Beside the TensorCores sit the SparseCores: small dataflow processors whose whole job is the irregular half of recommendation models — embedding lookups. Where the MXU wants dense tiles, a SparseCore issues many independent gathers into HBM and reduces the returned embedding vectors in a streaming dataflow, never touching the systolic arrays. Google's ISCA paper discloses the SparseCore's role, cost and effect; it does not publish a gate-level block diagram, and this entry does not invent one.

**Trace — one unit of work.**
1. **Indices arrive.**For one DLRM feature, a list of embedding indices reaches the SparseCore from the TensorCore side of the chip.
2. **Gather fans out.**The SparseCore turns the indices into many parallel, uncoalesced HBM reads — exactly the access pattern that would stall an MXU pipeline.
3. **Vectors stream back and reduce.**Returned embedding rows flow through the reduction datapath and are summed/averaged into a single pooled vector as they arrive.
4. **Dense side resumes.**The pooled vector crosses to a TensorCore and joins the bottom-MLP computation on the MXUs.
5. **Across the pod.**Because embedding tables are sharded over many chips, the gather usually triggers an all-to-all exchange over the ICI torus — the traffic pattern the paper's twisted-torus OCS topologies exist to serve.

**Bottleneck in / out.** Relieves the dense engines from latency-bound gather/scatter work that runs at memory latency, not FLOPS — embeddings stop throttling the matmul arrays. Creates a communication bottleneck instead: sharded embeddings turn every lookup into pod-wide all-to-all traffic, which strains torus bisection bandwidth far more than all-reduce ever did.

**Governing numbers.** Embeddings accelerated 5×–7× for about 5% of die area and power (ISCA 2023 abstract). Per-SparseCore queue depths and internal buffer sizes are not published.

**Primary source.** Jouppi et al., ISCA 2023 (abstract and §3).

### Apple Firestorm — M1

#### Inside the signature block: The wide front end and the ~630-entry reorder buffer — as far as anyone outside Apple can see

**Inside it.** Apple publishes no block diagram, pipeline depth, queue size or latency for Firestorm — the company 'doesn't disclose any details' (AnandTech), so there are no Apple internals to decompose. Everything known comes from microbenchmark reverse-engineering, chiefly AnandTech's A14/M1 deep dive: an 8-wide decode block, a reorder buffer measured at roughly 630 entries, an estimated ~354 integer and ~384 floating-point rename registers, at least seven integer execution ports, four 128-bit NEON pipelines, and load/store queues measured at roughly 148–154 outstanding loads and ~106 outstanding stores. Treat every number here as a measurement, not a specification.

**Trace — one unit of work.**
1. **Fetch from a huge L1I.**Instructions come from a 192 KB L1 instruction cache (a size Apple confirmed indirectly via Darwin kernel sources, per AnandTech) — six times a contemporary Intel core's — so fetch rarely waits.
2. **Decode eight at once.**Up to eight fixed-length ARM64 instructions decode per cycle, twice the width of contemporary x86 decoders, which must first find instruction boundaries.
3. **Rename into a deep window.**Instructions are renamed into the ~630-entry reorder buffer — the deepest out-of-order window measured in any commercial core of its era.
4. **Wait, then fire.**An instruction can sit in that window across multiple cache misses while hundreds of later instructions execute around the stall.
5. **Retire in order.**Results commit in program order; the window's depth is invisible to software except as sustained IPC on ordinary, unmodified code.

**Bottleneck in / out.** Relieves the instruction-level-parallelism wall: width plus window depth extract parallelism from serial code without compiler heroics, which is the core of Firestorm's performance per clock. Creates a verification and power problem — wakeup/select logic grows superlinearly with window size — and an information problem for everyone else: with no published internals, the mechanisms behind the measurements (predictor structure, queue partitioning) remain inference, marked unverifiable in the fact check.

**Governing numbers.** 8-wide decode; ROB ~630 entries; L1I 192 KB, L1D 128 KB; ~148–154 outstanding loads measured. All figures are AnandTech microbenchmark estimates — Apple has published none of them.

**Primary source.** No primary source exists. Secondary: AnandTech, 'Apple Silicon M1 / A14 Deep Dive' (2020), microbenchmark-based.

### Cerebras WSE-2

#### Inside the signature block: The dataflow core — 48 KB of SRAM with a processor attached

**Inside it.** Half of each tiny core (~38,000 µm²) is memory: 48 KB of local SRAM organized as eight single-ported banks, each 32 bits wide, holding both the core's instructions and its data — there is no cache, no shared memory and no coherence anywhere on the wafer. A small 256-byte software-managed cache holds hot structures such as accumulators. The logic half is a dataflow execution unit whose operations fire when their input data arrives, plus a router with ports to the four mesh neighbours and to the core itself; packets carry a 'color' that identifies their role, so arrival of the right color is what triggers the waiting computation.

**Trace — one unit of work.**
1. **A wavelet arrives.**A 32-bit data packet ('wavelet') enters the core's router from a neighbour, tagged with a color naming the operation it belongs to.
2. **Arrival is the trigger.**The color match wakes the instruction block waiting on that data in local SRAM — dataflow firing; the core never polls or spins.
3. **Multiply against a resident weight.**The core reads its weight from an SRAM bank — one cycle away, always — multiplies, and accumulates into the software-managed cache.
4. **Result leaves immediately.**The partial sum is emitted onto the fabric toward the next core in the wafer-wide choreography the compiler laid out.
5. **Zeros cost nothing.**If the arriving activation is zero, the dataflow simply doesn't fire downstream work — unstructured sparsity is skipped by construction, not by a sparsity engine bolted on.

**Bottleneck in / out.** Relieves the memory wall outright: weights live in SRAM one cycle from the ALUs, so the HBM-bandwidth bottleneck that governs GPU training disappears, and 850,000 cores give an aggregate SRAM bandwidth no packaged chip can approach. Creates a capacity wall — 48 KB per core, 40 GB per wafer is all the fast memory there is; models that outgrow it must stream weights from external MemoryX, reintroducing an off-wafer feed, and utilization rests entirely on the compiler's placement across the mesh.

**Governing numbers.** 850,000 cores; 48 KB SRAM per core in 8 single-ported 32-bit banks; 256-byte software-managed cache per core; 40 GB total on-wafer SRAM; ~1.1 GHz core clock and ~30 mW peak per core (Cerebras' Hot Chips 34 deep dive).

**Primary source.** Lie, 'Cerebras Architecture Deep Dive', Hot Chips 34 / IEEE Micro 43(3), 2023; Cerebras architecture deep-dive (vendor).

### Broadcom Tomahawk 5 (BCM78900)

#### Inside the signature block: The shared packet buffer — and the Cognitive Routing wrapped around it

**Inside it.** Broadcom publishes Tomahawk 5's architecture and features, not its floorplan: 512 lanes of 100G PAM4 SerDes feed a single-pass pipeline (parse → lookup → buffer → schedule → rewrite) built around one packet buffer pooled across all 64 ports, a traffic manager/scheduler, telemetry engines, and the Cognitive Routing block that watches link utilization locally and downstream. The buffer's capacity in megabytes, its cell size and its queue depths are not published — Broadcom discloses the shared-buffer architecture and its latency claims only, so this section goes no deeper than Broadcom does.

**Trace — one unit of work.**
1. **Packet lands and is classified.**A packet arrives on a group of 100G PAM4 lanes making one 800G port; the parser classifies it and a single-pass lookup resolves egress, including VxLAN bridging/routing where configured.
2. **Cells enter the shared pool.**The packet is written into the one shared buffer — there is no per-port partition, so any egress may borrow memory that idle ports are not using.
3. **Cognitive Routing inspects the flow.**For the packet's flow, the routing block compares utilization on the hashed path against alternatives, including downstream state; a congested hash choice gets its flowlet steered to a lighter link.
4. **Scheduler drains at line rate.**The traffic manager arbitrates the port's queues and emits the packet; per-port fairness is enforced by admission thresholds on the shared pool, not by static partitions.
5. **Telemetry stamps the exit.**In-band telemetry records the packet's residence time and queue state as it leaves — the tail-latency evidence the architecture is sold on.

**Bottleneck in / out.** Relieves incast tail latency: partitioned buffers drop packets when one port microbursts while the rest of the pool sits idle; pooling removes that artificial wall, which is why Broadcom markets the part on RoCEv2 job completion time. Creates a fairness problem in exchange — one abusive flow can occupy the shared pool, so dynamic admission thresholds become the critical control logic — and a monolithic 51.2 Tbps single-pass pipeline leaves no modularity: every feature must fit the one pass.

**Governing numbers.** 51.2 Tbps aggregate switching capacity; 64 × 800GbE ports; 512 × 100G PAM4 SerDes lanes. Buffer capacity, cell size and queue depths: not published by Broadcom.

**Primary source.** Broadcom, 'Broadcom Ships Tomahawk 5' launch release (GlobeNewswire, 16 Aug 2022).

### Scan Design & ATPG Architecture

#### Inside the signature block: The scan chain — and the ATPG loop that drives it

**Inside it.** The signature block is a pair, one on-chip and one in software. On-chip: the mux-D scan flip-flop, which selects between functional data and scan data, strung by the thousand into shift chains that thread the whole design; around the chains sit test clock and power control, and in production parts an EDT decompressor/compactor pair that fans a handful of tester channels into hundreds of internal chains and compresses the responses back. In software: the ATPG engine, which treats the chains as making every flip-flop directly controllable and observable, and solves for input patterns that excite a chosen fault and propagate its effect to a scan cell.

**Trace — one unit of work.**
1. **ATPG targets one fault.**The engine picks a fault — say a net stuck-at-1 — justifies the opposite value at that net through the logic, and propagates the resulting difference forward to an observable scan cell, producing one test vector.
2. **Shift-in.**The chip enters test mode and the vector shifts serially into the scan chains, one bit per shift clock, until the chains hold exactly the state ATPG specified.
3. **Capture.**One functional capture pulse runs the real logic — at speed for transition-fault patterns, slowly for stuck-at — and the response is captured back into the scan cells.
4. **Shift-out and compare.**The response shifts out (overlapped with the next vector shifting in) and is compared against the simulated good-machine response.
5. **Mismatch localizes.**A differing bit names the failing chain and, through ATPG's fault dictionary, the small set of physical nets that could explain it — diagnosis, not just pass/fail.

**Bottleneck in / out.** Relieves the observability wall: without scan, internal nodes are unreachable from the pins and stuck-at coverage collapses to whatever functional patterns happen to excite. Creates test-data volume — chains are serial, so pattern count × chain length sets tester time and cost; that pressure is precisely what EDT compression (typically an order of magnitude or more) and hierarchical test exist to relieve, at the price of X-states that must be masked in the compactor and a small aliasing risk.

**Governing numbers.** Governing numbers are per-design, not per-architecture: chain count and length, pattern count, and the coverage sign-off (stuck-at coverage in the high ninety percents is the usual production bar). Lineage anchor: LSSD — the level-sensitive scan discipline this all descends from — is Eichelberger & Williams, DAC 1977.

**Primary source.** Eichelberger & Williams, 'A Logic Design Structure for LSI Testability', DAC 1977; IEEE Std 1149.1/1500 test-access standards.

### DEC Alpha 21264 (EV6)

#### Inside the signature block: The tournament branch predictor

**Inside it.** Three tables and a referee, all read in the fetch cycle. Local path: a 1,024-entry local history table holding 10 bits of outcome history per branch (indexed by branch address), feeding a 1,024-entry table of 3-bit saturating counters — this path learns a branch's own rhythm, such as alternating taken/not-taken. Global path: a 4,096-entry table of 2-bit counters indexed by the 12-bit history of the last twelve branches taken anywhere — this path learns correlations between branches. Choice path: a second 4,096-entry 2-bit table, also indexed by global history, that learns which of the two predictors to believe in each global context.

**Trace — one unit of work.**
1. **Fetch presents a branch PC.**The local history table returns this branch's last-10-outcomes pattern.
2. **Local prediction forms.**The 10-bit pattern indexes the local counter table; the counter's most significant bit is the local prediction.
3. **Global and choice read in parallel.**The 12-bit global history indexes both the global counter table (the global prediction) and the choice table (which predictor has been right in contexts like this one).
4. **The chooser arbitrates.**If local and global agree, that is the prediction; if they disagree, the choice counter's MSB picks the winner and fetch steers down that path.
5. **Retirement trains all three.**When the branch retires, the true outcome shifts into both histories and adjusts the counters — the choice counter is updated only when the two predictors disagreed, incrementing toward whichever was right.

**Bottleneck in / out.** Relieves the misprediction wall of a 4-wide machine that speculates through up to ~20 in-flight branches: neither history style wins everywhere, so learning which to trust beats enlarging either table. Creates a fetch-latency and aliasing problem — three SRAM reads plus arbitration must fit inside the fetch cycle at 600 MHz, and 1,024 local-history entries mean unrelated branches share state and pollute each other's patterns.

**Governing numbers.** Local history 1,024 × 10 bits; local counters 1,024 × 3 bits; global and choice tables 4,096 × 2 bits each; global history 12 branches; Kessler reports 90–100% prediction accuracy on most simulated applications/benchmarks.

**Primary source.** Kessler, McLellan & Webb, 'The Alpha 21264 Microprocessor Architecture', ICCD 1998; Kessler, 'The Alpha 21264 Microprocessor', IEEE Micro, Mar–Apr 1999.

### Intel Pentium Pro (P6)

#### Inside the signature block: Reorder buffer + reservation station — the pool of pending µops

**Inside it.** P6's out-of-order heart is three structures working as one. The 40-entry reorder buffer (ROB) holds every in-flight µop in program order and doubles as the rename register file — each ROB entry is the physical destination of one µop. The register alias table (RAT) maps the eight architectural integer registers (and FP/flags state) onto ROB entries, so renaming is just 'point this name at that entry'. Ahead of the five execution ports sits a unified 20-entry reservation station (RS): renamed µops wait there with their operand tags, watch for results, and dispatch to a port the cycle their operands land.

**Trace — one unit of work.**
1. **Fetch and length-decode.**The fetch unit pulls 16 bytes from the 8 KB instruction cache; because x86 instructions run 1–15 bytes, the first decode stages exist just to find the boundaries.
2. **Crack into µops.**Three decoders translate the instruction — a load-and-add becomes a load µop plus an add µop; anything longer goes to the microcode sequencer.
3. **Rename and allocate.**The RAT points the destination register at a fresh ROB entry; both µops take seats in the reservation station carrying tags, not values, for operands still in flight.
4. **Dataflow execution.**The load µop dispatches as soon as its address operands are ready; the add waits in the RS until the load's ROB entry broadcasts its result, then fires out of order.
5. **Retire in order.**When both µops reach the ROB head complete, they retire — up to three per cycle — copying results to the architectural register file and making all the speculation invisible.

**Bottleneck in / out.** Relieves the in-order stall: a cache miss no longer freezes the machine — independent µops flow around it, which is why P6 beat the in-order Pentium decisively on real code at similar clock rates. Creates a hard window ceiling: 40 ROB entries and 20 RS entries bound how much latency can be hidden, and when the RS fills with µops queued behind one miss, the wide front end stalls anyway — the direct motivation for the far larger windows in every successor.

**Governing numbers.** ROB 40 entries; reservation station 20 entries; branch target buffer 512 entries; up to 3 µops retired per cycle; five dispatch ports (integer ×2 class, FP, load, store).

**Primary source.** Papworth, 'Tuning the Pentium Pro Microarchitecture', IEEE Micro, Apr 1996; Colwell et al., ISSCC 1995; Bhandarkar & Ding, HPCA 1997.

### NVIDIA Volta V100

#### Inside the signature block: The first Tensor Core — a 4×4×4 matrix engine

**Inside it.** Each Volta SM is split into four processing blocks; each block carries 16 FP32 cores, 8 FP64 cores, 16 INT32 cores, its own warp scheduler and register-file slice — and two Tensor Cores, eight per SM. A Tensor Core is a fixed-function array hard-wired for a single operation, D = A × B + C on 4×4 matrices: A and B enter as FP16, products are formed in full precision, and accumulation is FP32 (or FP16). There is no smaller mode and no scalar mode; the warp's 32 threads collectively own the operand fragments in their registers and issue the operation together through the WMMA interface.

**Trace — one unit of work.**
1. **Fragments distributed.**A warp loads 16×16 tiles of A and B; each of the 32 threads holds the specific elements the fragment layout assigns it in its registers.
2. **One WMMA instruction.**The warp issues a single matrix instruction; the SM decomposes it into 4×4×4 primitives across its Tensor Cores.
3. **64 FMAs per core per clock.**Each Tensor Core retires one 4×4×4 multiply-accumulate per clock — 64 mixed-precision FMAs — with FP16 inputs and FP32 accumulation.
4. **Accumulator fragment returns.**The D fragment lands back in the warp's registers, laid out for the next WMMA or for the CUDA cores to apply an activation.
5. **Scale-out arithmetic.**Eight cores per SM × 80 SMs = 640 Tensor Cores; at boost clock that is the V100's 125 TFLOPS mixed-precision peak.

**Bottleneck in / out.** Relieves the instruction-issue wall: one instruction now performs 64 FMAs, lifting matrix throughput to 12× the previous generation's FP32 rate per NVIDIA — deep learning stopped being bound by scalar issue bandwidth. Creates a precision-and-shape contract: the rate exists only for dense FP16 matrix math tiled to the warp fragment geometry; everything else — irregular, scalar, FP64-heavy work — falls back to the CUDA cores at a small fraction of peak.

**Governing numbers.** 640 Tensor Cores (8 per SM × 80 SMs); 64 FMA per Tensor Core per clock; 1,024 floating-point operations per clock per SM from Tensor Cores; 125 TFLOPS peak mixed precision.

**Primary source.** NVIDIA, 'Programming Tensor Cores in CUDA 9' (developer blog, 2017) and the Volta V100 architecture whitepaper.

### NVIDIA Ampere A100

#### Inside the signature block: MIG — partitioning hardware that cuts the whole GPU, not just the SMs

**Inside it.** Multi-Instance GPU is a partitioning layer sliced through every shared resource at once. Compute: the 108 SMs are grouped into instances. Memory: each instance receives dedicated partitions of the 40 MB L2 cache, of the memory controllers and of HBM bandwidth, fenced by hardware isolation so no instance's traffic can evict or starve another's. A single A100 divides into as many as seven GPU instances, in fixed profiles; each instance presents to software as a small, complete GPU with its own command streams and fault domain. The totality — compute, cache and bandwidth partitioned together — is what makes the performance guarantee sellable.

**Trace — one unit of work.**
1. **Instance created.**The CSP selects a profile; the partitioning hardware fences off an SM group, its L2 partition and its memory-controller share before any tenant code runs.
2. **Tenant kernel arrives.**The instance's own scheduler dispatches the kernel onto its SM slice only — it cannot see, preempt or be preempted by neighbouring instances.
3. **Memory stays home.**The kernel's traffic hits its own L2 partition; a neighbour's training job streaming gigabytes cannot evict this instance's working set or consume its DRAM bandwidth.
4. **Predictable, smaller throughput.**The kernel runs slower than a full A100 would run it — and at a rate the CSP can print in a price list, because the isolation is in the memory system, not in a software quota.
5. **Teardown.**Destroying the instance returns all three resource slices to the pool together; repartitioning happens between jobs, never mid-kernel.

**Bottleneck in / out.** Relieves GPU under-utilization: most inference and development jobs cannot fill 108 SMs, so a monolithic GPU idled or time-shared badly; MIG turns one big GPU into up to seven right-sized ones. Creates fragmentation and rigidity — slice sizes come from fixed profiles, an instance cannot burst into a neighbour's idle capacity, and carving must happen across compute, cache and bandwidth simultaneously, so the least divisible resource sets the granularity.

**Governing numbers.** Up to 7 instances per A100; 40 MB L2 partitioned among instances; SM shared memory configurable up to 164 KB (Ampere column of NVIDIA's own comparison table); NVIDIA claims up to 7× more GPU instances per server for CSPs.

**Primary source.** NVIDIA, 'NVIDIA Ampere Architecture In-Depth' (developer blog / GA100 whitepaper, 2020).

### AMD CDNA 3 — MI300X

#### Inside the signature block: Infinity Cache — the 256 MB last level living in the I/O dies

**Inside it.** MI300X inverts the usual stack-up: four I/O dies (IODs) form the package base, eight compute dies (XCDs) stack on top of them, and the IODs hold the memory system — the HBM3 controllers for eight stacks and 256 MB of Infinity Cache, the GPU's last-level cache. Each XCD is itself a small GPU: 40 compute units (38 enabled) around a 4 MB L2, with 64 KiB of local data share per CU. Because every XCD-to-HBM access funnels down through the IODs, the Infinity Cache sees the whole GPU's traffic — one 256 MB residency pool shared by 304 CUs instead of eight private caches that would duplicate the same weights eight times.

**Trace — one unit of work.**
1. **A CU load misses locally.**A wavefront on XCD 3 loads a weight line; it misses the CU's 32 KB L1 and the XCD's 4 MB L2.
2. **Down into the IOD.**The request crosses the XCD/IOD interface on the package's Infinity Fabric into the I/O die beneath.
3. **Infinity Cache lookup.**On a hit — the common case when 304 CUs are all streaming the same model weights — the line returns at on-package latency and HBM never sees the request.
4. **Miss path to HBM3.**On a miss, the IOD's controller issues the read across the 8,192-bit aggregate HBM3 interface (192 GB at 5.3 TB/s) and fills the cache line for the next CU that asks.
5. **Writeback stays coherent.**The IOD layer is also where the package's coherence and the seven external Infinity Fabric links (128 GB/s each) meet the memory system — one meeting point for on-package and off-package traffic.

**Bottleneck in / out.** Relieves HBM bandwidth pressure: 5.3 TB/s is vast until 304 CUs stream a 70B model's weights every token; a large shared last-level cache converts repeat reads into on-package hits and raises effective bandwidth well past the HBM figure. Creates a funnel — all memory traffic converges on four IODs, so XCD↔IOD fabric bandwidth and cache behaviour once the working set exceeds 256 MB become the new ceiling, and the cache's die area is spent on IOD silicon that computes nothing.

**Governing numbers.** 256 MB Infinity Cache (AMD datasheet); 8 XCDs × 40 CUs (38 active) = 304 CUs; 4 MB L2 per XCD; 64 KiB LDS per CU; 192 GB HBM3 at 5.3 TB/s over an 8,192-bit interface.

**Primary source.** AMD, 'AMD Instinct MI300X Accelerator' data sheet; AMD ROCm documentation, MI300 series architecture tables; Smith, Hot Chips 2024.

### Fujitsu A64FX

#### Inside the signature block: The Core Memory Group — a NUMA node as the unit of design

**Inside it.** The A64FX die is four identical Core Memory Groups (CMGs) joined by a ring network-on-chip with directory-based coherence. Each CMG is a complete, self-contained NUMA node: 12 compute cores plus 1 assistant core that absorbs OS and daemon work, a crossbar from the cores into a shared 8 MiB 16-way L2, its own memory controller, and its own 8 GiB HBM2 stack delivering 256 GB/s. Each core carries two 512-bit SVE pipelines. The design intent is explicit: bind a thread and its data inside one CMG and the ring — the only shared, contended resource — stays out of the critical path.

**Trace — one unit of work.**
1. **SVE FMA stream issues.**A compute core's two 512-bit SVE pipes fire fused multiply-adds — up to 32 double-precision results per cycle from one core when both pipes are fed.
2. **L1 feeds the pipes.**Operands stream from the core's 64 KiB L1 data cache, whose gather mechanism can pull strided array elements directly toward the vector registers.
3. **L2 inside the CMG.**L1 misses cross the CMG crossbar into the shared 8 MiB L2 — still on-node, still uncontended by the other 39 cores' traffic.
4. **HBM2 inside the CMG.**L2 misses are served by the CMG's own HBM2 stack at 256 GB/s; no ring traversal, no foreign memory controller.
5. **Only sharing crosses the ring.**A genuinely shared line is the exception that pays NUMA latency across the ring — the programming model's whole job is keeping that case rare.

**Bottleneck in / out.** Relieves the bytes-per-FLOP wall that starves wide SIMD on DDR machines: ~1 TB/s of HBM2 for 48 cores (a bytes/DP-FLOP ratio around 0.37) is why Fugaku's cores stay fed on real HPC codes. Creates a placement burden — performance now depends on binding threads and data to the same CMG; careless placement silently pays ring latency and a fraction of the bandwidth, so the architecture exports its central problem to the runtime and the programmer.

**Governing numbers.** 4 CMGs; 12 compute + 1 assistant core per CMG; 8 MiB 16-way L2 per CMG; 8 GiB HBM2 per CMG at 256 GB/s; 1,024 GB/s aggregate per chip; 512-bit SVE × 2 pipelines per core.

**Primary source.** Fujitsu, A64FX disclosures at Hot Chips 2018 / Supercomputing materials; architecture as described in the published A64FX literature.

### Google Argos VCU

#### Inside the signature block: The encoder core — one 2160p stream per core, ten cores per die

**Inside it.** An Argos die is ten identical encoder cores plus deliberately boring surroundings — Google's ASPLOS paper notes that everything else on the die is off-the-shelf IP (host interface, memory controllers for the LPDDR banks). Each core is a self-contained hardware encoder for VP9 and H.264-class workloads: a prediction/motion-search front end working against up to three reference frames, transform and quantization, an in-loop reconstruction path so the core's reference frames match a decoder's exactly, and a boolean arithmetic entropy coder emitting the final bitstream, with per-core SRAM and frame staging buffering the work. Google does not publish the core's internal stage latencies, queue depths or buffer sizes — the disclosure stops at core count and per-core rate, and so does this section.

**Trace — one unit of work.**
1. **One rung assigned.**YouTube's scheduler gives this core one output rendition — say the 2160p VP9 rung of a ladder of a dozen-plus versions of the same upload.
2. **Frame staged.**The 2160p frame lands in the core's staging memory from LPDDR; the core now works independently of the other nine.
3. **Prediction and mode decision.**Block by block, the core searches motion against up to three reference frames and chooses intra or inter prediction per block.
4. **Transform, quantize, reconstruct.**Residuals are transformed and quantized; the reconstruction loop rebuilds the decoded frame locally so future predictions reference exactly what viewers' decoders will see.
5. **Entropy-coded bitstream out.**The boolean arithmetic coder emits the compressed rung, which joins the other renditions produced in parallel by sibling cores and sibling cards.

**Bottleneck in / out.** Relieves CPU transcoding economics: warehouse-scale software encoding on general-purpose servers was the cost centre, and ten fixed-function cores per die are Google's answer — it reports 20–33× better compute efficiency at system level than its previous optimized server fleet. Creates codec rigidity: a VP9-era fixed-function core cannot become an AV1 core by firmware, so the follow-on VCU generation had to add AV1 in new silicon — the standing tax on all fixed-function media design.

**Governing numbers.** 10 encoder cores per die; 2 dies per accelerator card; each core encodes 2160p in real time at up to 60 fps using three reference frames (Google, ASPLOS 2021). Internal buffer sizes and stage latencies: not published.

**Primary source.** Ranganathan et al., 'Warehouse-Scale Video Acceleration: Co-Design and Deployment in the Wild', ASPLOS 2021; Google/YouTube VCU disclosure (2021).

### Apple Secure Enclave

#### Inside the signature block: Memory Protection Engine + the UID-bound AES path

**Inside it.** Apple's Platform Security guide names the enclave's internals — unusually, for Apple. A dedicated Secure Enclave Processor (SEP) runs sepOS (an Apple-customized L4 microkernel) at a deliberately lower clock to blunt clock/power side channels. It boots from its own immutable Boot ROM, has its own AES engine and true random number generator, and — from A11/S4 onward — a Memory Protection Engine with encrypted, anti-replay-protected memory. The SEP owns no private RAM: it works in a dedicated region of the device's ordinary DRAM, and the Memory Protection Engine encrypts every block it writes there (AES in XEX mode) with a CMAC authentication tag stored alongside. The fused, per-device UID key exists only inside the AES engine's hardware key path — software, including sepOS itself, can ask for operations with it but can never read it.

**Trace — one unit of work.**
1. **Mailbox request.**The application processor asks the enclave to verify a passcode attempt. The only channel is a hardware mailbox — no memory is ever shared or mapped between the two processors.
2. **Key derivation under the UID.**sepOS feeds the attempt into a key-derivation path keyed by the UID inside the AES engine; because the UID never leaves the enclave, the derived key cannot be computed on any other machine — offline brute force is architecturally impossible.
3. **Attempt counter enforced.**The anti-replay-protected attempt counter (held in secure storage) is checked and incremented, imposing escalating delays — the enclave is a rate-limited oracle by design.
4. **Memory traffic stays encrypted.**Throughout, every sepOS read/write to DRAM passes the Memory Protection Engine: blocks are decrypted and CMAC-verified on the way in, re-encrypted on the way out, under an ephemeral per-boot key.
5. **Verdict only.**On success, sepOS unwraps the data-protection class keys and releases them to the storage subsystem; on failure, the only thing that crosses the mailbox is 'no'.

**Bottleneck in / out.** Relieves the 'kernel compromise equals key compromise' failure: a fully owned application-processor OS finds only wrapped keys and a slow oracle, which is what makes a 4–6 digit passcode a meaningful secret at all. Creates a deliberate bottleneck — all unlock and key operations serialize through one small, slow-clocked core and a narrow mailbox, and the UID's absolute secrecy means enclave state loss is device data loss; availability was traded away for confidentiality, on purpose.

**Governing numbers.** Apple publishes mechanisms, not dimensions: no SEP clock, cache or queue sizes are disclosed. Published anchors: MPE encrypts with AES in XEX mode plus a per-block CMAC tag; anti-replay from A11/S4; two ephemeral memory-protection keys from A14/M1.

**Primary source.** Apple, 'The Secure Enclave', Apple Platform Security guide (support.apple.com).

### SK hynix HBM3 / HBM3E

#### Inside the signature block: Inside the stack — channels, TSVs and the base die

**Inside it.** An HBM3 stack is 8–12 thinned DRAM dies (about 30 µm each in SK hynix's parts) pierced by through-silicon vias (TSVs) and microbump-bonded onto a logic base die that carries the PHY and the host-facing test/repair logic. The 1,024-bit interface is not one wide bus: it is 16 independent channels (each split into two pseudo-channels), each with its own command/address and 64 data bits — the stack behaves as sixteen small memories operating in parallel. The stack sits on a silicon interposer millimetres from the host die; those short, slow wires are the whole trick, letting 1,024 pins run at a moderate 6.4 Gb/s each instead of a few pins at SerDes speeds.

**Trace — one unit of work.**
1. **Host issues a read.**The host's memory controller picks the channel whose address map owns the target line and issues the read on that channel's command bus — the other fifteen channels are free to serve other requests simultaneously.
2. **Base die routes upward.**The command crosses the base die and climbs the TSV column to the die and bank that hold the row.
3. **DRAM core senses.**The target die activates the row into its sense amplifiers — the same ~tens-of-nanoseconds DRAM core latency as any DDR part; HBM hides it with parallelism, not speed.
4. **Data returns down the TSVs.**The column data travels back through the stack to the base die's PHY.
5. **Width does the work.**64 bits from this channel × 16 channels in lockstep is 128 bytes per burst beat across the stack — 819 GB/s comes from width multiplied by channels, not from pin heroics.

**Bottleneck in / out.** Relieves the pin wall: package escape and signal integrity cap how many fast pins a die can have, so going wide-and-moderate (1,024 bits at 6.4 Gb/s/pin) wins bandwidth and energy-per-bit that narrow-and-fast interfaces cannot. Creates a beachfront and yield wall — bandwidth scales only by adding stacks around the host die's perimeter, stack height is capped by TSV/thinning limits, and one bad die in a 12-high stack threatens the whole assembly, which is why known-good-die test and stack-level repair dominate HBM cost.

**Governing numbers.** 1,024-bit interface per stack; 16 channels; up to 6.4 Gb/s per pin; 819 GB/s per stack; SK hynix HBM3 ships 8-high (16 GB) and 12-high (24 GB) stacks of 16 Gb dies.

**Primary source.** SK hynix, HBM3 development announcement (Oct 2021); JEDEC JESD238 HBM3 standard (Jan 2022).

### JTAG & Boundary Scan — IEEE 1149.1

#### Inside the signature block: The TAP controller — a 16-state machine in every chip

**Inside it.** Four signal pins (TCK, TMS, TDI, TDO; TRST optional) drive the Test Access Port controller: a finite-state machine with exactly 16 states, identical in every compliant device, sequenced entirely by the value of TMS on each TCK edge. Behind it sit the instruction register — which selects the active test mode (BYPASS, SAMPLE/PRELOAD, EXTEST, plus optional INTEST, IDCODE, USERCODE and vendor-private instructions) — and the data registers: the boundary-scan register (one cell per device pin, each cell able to observe its pin or drive it), the device-ID register, and the bypass register, which is exactly one bit long. Multiple chips chain TDI→TDO into a single board-level scan path.

**Trace — one unit of work.**
1. **Walk the FSM.**Five TCK cycles with TMS high reset any TAP to Test-Logic-Reset; a scripted TMS sequence then steers both chips on a board, in lockstep, toward Shift-IR.
2. **Load EXTEST.**The EXTEST opcode shifts into each instruction register and latches at Update-IR — the boundary cells now control the pins.
3. **Preload and drive.**Shift-DR loads chip A's boundary cells with a test pattern; Update-DR applies those values to A's actual output pins, across real board traces.
4. **Capture at the receiver.**Chip B's boundary cells capture whatever truly arrived at its input pins — solder opens, shorts and stuck nets included.
5. **Shift out and compare.**B's captured bits stream out of TDO and are diffed against the board netlist's expectation; a wrong bit localizes the fault to a specific net and pin pair.

**Bottleneck in / out.** Relieves board-level invisibility: BGA packages hid every solder joint beneath the chip and made probe-based board test physically impossible; boundary scan moved the probe points inside the silicon. Creates a speed wall — everything moves serially at TCK rates orders of magnitude below functional speed, so 1149.1 verifies interconnect and gross presence, never timing; at-speed internal test needed scan/ATPG, and embedded instruments later needed IEEE 1687 networks behind this same TAP.

**Governing numbers.** 16 TAP states; bypass register exactly 1 bit (a whole chip passes the chain in one clock); boundary register length equals the device's signal-pin count; instruction register at least 2 bits (four mandatory encodings' worth of space).

**Primary source.** IEEE Std 1149.1-2013, Standard Test Access Port and Boundary-Scan Architecture.

### Memory BIST & Repair (MBIST)

#### Inside the signature block: The algorithm engine — running a March test at full speed

**Inside it.** Per memory cluster, the MBIST block is four sub-blocks behind a test access port (IJTAG/JTAG): a controller that sequences the test and reports status; an algorithm engine — a small programmable sequencer generating address orders (up/down/any) and data backgrounds, with a library of hard-coded March variants selectable per memory; a comparator with fail logging that records the address and cycle of every mismatch; and redundancy analysis (BIRA), which takes the fail map and solves whether the memory's spare rows and columns can cover every failure, then programs the winning repair into eFuses at manufacturing test.

**Trace — one unit of work.**
1. **March C−, element 1 — initialize.**The engine writes 0 into every cell of the SRAM, in any address order: a known background against which faults can be seen.
2. **Ascending pass.**Marching up the address space, each cell is read (expect 0) then written 1; the next cell's read checks it was not disturbed by its neighbour's write — coupling faults announce themselves here.
3. **Ascending pass, inverted.**Still ascending: read 1, write 0 for every cell, flipping the background under the same scrutiny.
4. **Descending pass.**The same read/write pairs run in reverse address order — order-dependent coupling faults that are structurally invisible to an ascending-only march are caught by symmetry.
5. **Final read and verdict.**A last read-0 sweep confirms the array; every mismatch along the way sits in the fail log with its address, ready for BIRA to map onto spare rows/columns or condemn the die.

**Bottleneck in / out.** Relieves test access: embedded SRAMs have no pins, and an external tester cannot reach them at functional speed — the BIST engine tests the array at its own clock, catching speed-related faults no ATE pattern could. Creates an area-and-coverage negotiation: engine, comparator and spare rows/columns cost area beside every memory, and the standard 10-operation March C− still misses fault classes (dynamic and linked faults) that longer algorithms — 17 operations per cell and up — buy back with proportionally longer test time.

**Governing numbers.** March C− complexity is 10n (ten operations per cell); the element sequence is {any-order write 0; up (read 0, write 1); up (read 1, write 0); down (read 0, write 1); down (read 1, write 0); any-order read 0}. Longer production algorithms run to 17n+.

**Primary source.** Van de Goor, 'Testing Semiconductor Memories: Theory and Practice' (the March-test canon); IEEE Std 1500 / 1687 access context.


### Intel 4004

#### Inside the signature block: The address register — program counter and 3-level stack as a dynamic RAM array

**Inside it.** The 4004's most pinched resource was pins: a 16-pin DIP cannot carry separate address and data buses, so the whole machine funnels through four data lines (D0–D3) and an 8-phase machine cycle. The address register is the block that makes that work: a dynamic RAM cell array of 4 × 12 bits — one level holds the program counter, the other three form a push-down stack for subroutine return addresses, so calls nest only three deep. The stack address comes from the effective address counter and from a refresh counter (dynamic cells must be re-read to hold their charge), multiplexed into the decoder. The address, once read, sits in an address buffer and is demultiplexed onto the internal 4-line bus in three 4-bit slices; a 4-bit carry-look-ahead incrementer advances it by one after the last slice, and the incremented address is written back into the array. The instruction register is two 4-bit halves, OPR (opcode) and OPA (modifier), loaded from the internal bus at M1 and M2; a double-cycle flip-flop stretches the cycle to 16 clock periods for the five double-length instructions.

**Trace — one unit of work: a single-word ADD.**
1. **SYNC.** The CPU raises SYNC to mark the start of a machine cycle; ROMs and RAMs listen for it.
2. **A1 — low slice.** The address buffer puts the program counter's lowest 4 bits on D0–D3; the selected 4001 ROM latches them.
3. **A2 — middle slice.** The next 4 bits follow; the ROM now holds 8 of the 12 address bits.
4. **A3 — high slice.** The last 4 bits complete the 12-bit address; the incrementer advances the program counter by one and writes it back into the address register.
5. **M1 — opcode in.** The ROM returns the instruction's upper nibble (OPR) on the bus; it is latched into the OPR register.
6. **M2 — operand in.** The ROM returns the lower nibble (OPA — here, the index-register number); it is latched into the OPA register.
7. **X1–X3 — execute.** The decoder gates the named index register's 4 bits onto the internal bus and into the ADB register; the 4-bit adder combines it with the accumulator and the carry flip-flop; the sum lands back in the accumulator, the carry flip-flop records any overflow, and the condition logic notes whether the result is zero. The next SYNC starts the following instruction.

**Bottleneck in / out.** Relieves the pin-count wall: multiplexing 12-bit addresses, 8-bit instructions and 4-bit data over four lines is what lets a complete CPU fit in a 16-pin package at all. Creates the bus-time wall instead — five of the eight clock periods in every cycle are spent just moving the address and the instruction across those four wires, leaving three for execution; a double-word instruction costs a second full cycle. And because the address and index registers are dynamic RAM arrays, the chip pays a refresh tax: reset must be held for at least eight full instruction cycles (64 clock periods) so the refresh counter can scan every location before the CPU is usable.

**Governing numbers.** 8 clock periods per machine cycle (10.8 µs at 740 kHz); 16 clock periods for double-length instructions; address register 4 × 12 bits (1 PC + 3 stack levels); index register 16 × 4 bits; reset ≥ 8 instruction cycles (64 clock periods) to clear; up to 4K × 8 ROM and 1,280 × 4 RAM addressable.

**Primary source.** Intel, MCS-4 User's Manual (1973), §II '4 Bit Central Processor Unit (CPU) — 4004' and Figure 2, 'MCS-4 Basic Instruction Cycle'.

#### Inside the signature block: The 4-bit adder, accumulator and decimal-adjust ROMs

**Inside it.** The compute heart is deliberately minimal: a 4-bit ripple-through-carry adder, one operand from the ADB register (fed from the internal bus), the other from the 4-bit accumulator plus the carry flip-flop. The accumulator carries a shifter for rotate-left/rotate-right, and talks to the command-control register, the condition logic and the internal bus. The clever piece is beside it: small special ROMs that perform code conversion for the DAA (decimal adjust accumulator) and KBP (keyboard process) instructions — a lookup table in silicon that turns a binary sum into a decimal digit, so the chip adds in the decimal system its calculator workload actually uses. The condition logic watches ADD = 0, ACC = 0, the carry flip-flop and the external TEST pin to implement conditional jumps (JCN) and increment-and-skip (ISZ).

**Trace — one unit of work: adding two decimal digits, 9 + 8.**
1. **ADD executes.** The index register holding 8 reaches the ADB register; the adder computes 9 + 8 = 17, binary 10001 — the low nibble (0001) lands in the accumulator and the carry flip-flop is set.
2. **DAA executes.** The decimal-adjust instruction consults the special ROM: the low nibble overflowed past 9 (carry out of the nibble), so the conversion adds 6 to it.
3. **Result corrected.** The accumulator now holds 0111 (7) with the carry representing the tens digit — 0x17 in BCD, i.e. 17 decimal, the correct decimal answer, with no software conversion step.
4. **Condition logic watches.** ADD = 0 and ACC = 0 are sensed after every operation, so a following JCN or ISZ can branch on the result without a separate compare instruction.

**Bottleneck in / out.** Relieves the decimal-arithmetic wall: BCD lives in the hardware, so a calculator program adds digits natively instead of converting binary results in software — the manual advertises adding two 8-digit numbers in 850 µs. Creates the width wall instead: every operation wider than one digit is a software loop of 4-bit steps through the single accumulator, and the ripple-through-carry adder has no look-ahead, so the datapath is fundamentally one digit per instruction cycle.

**Governing numbers.** 4-bit ripple-through-carry adder; accumulator 4 bits + 1 carry flip-flop; adding two 8-digit numbers takes 850 µs (Intel's own figure); instruction set 45 instructions in the manual's count (16 machine — 5 double-length — plus 14 accumulator-group and 15 I/O & RAM).

**Primary source.** Intel, MCS-4 User's Manual (1973), §II '4-Bit Adder' and 'Instruction Register Decoder and Control'; cover, 'Addition of Two 8-Digit Numbers in 850 Microseconds'.

### MOS Technology 6502

#### Inside the signature block: The decode PLA and the timing sequencer — the whole control unit in one array

**Inside it.** There is no microcode ROM and no stored control program in the 6502. The entire control unit is one combinational array plus a small shift register. The fetched opcode byte is latched into the instruction register and its predecode register; the decode ROM — a 130-row by 21-column array — matches the opcode's bit pattern against its rows and raises 131 PLA outputs (pla0–pla130), which fan out as roughly 44 datapath control signals (dpc0–dpc43) telling every register, bus driver and ALU input what to do in the current cycle. A T-state shift register steps the machine through its timing states (T0–T5), so each instruction is a fixed walk through the states its PLA row names — T0 for the opcode fetch, T1–T2 for operand bytes, T3–T5 for the work itself. The program counter has its own dedicated increment circuit, able to increment all 16 bits in a single cycle, and it almost always increments during the fetch cycle so the next byte is already addressed.

**Trace — one unit of work: LDA $2000 (opcode $AD, absolute addressing, 4 cycles).**
1. **Cycle 1 (T0) — opcode fetch.** The PC puts the opcode's address on the bus; the opcode byte $AD returns and is latched into the instruction register and predecode register; the dedicated increment circuit advances the PC by one in the same cycle.
2. **Cycle 2 (T1) — low address byte.** The PLA's outputs have settled and steer the datapath; the byte $00 is fetched from the PC's new address and latched as the low half of the effective address; the PC increments again.
3. **Cycle 3 (T2) — high address byte.** The byte $20 is fetched and latched as the high half; the effective address $2000 is now complete and driven onto the address bus; the PC increments to the next instruction.
4. **Cycle 4 (T3) — data.** The byte at $2000 is fetched from memory and lands in the accumulator; the N and Z flags update from it. The next instruction's fetch begins on the following cycle.

**Bottleneck in / out.** Relieves the control-cost wall: a stored microcode sequencer would have cost transistors the $25 chip did not have — one PLA plus a shift register is the whole control unit, and it is a large part of how the die stayed at 3,510 transistors. Creates a fixed-at-mask control plane: decode is combinational pattern-matching, not a list of legal instructions, so opcode patterns the manual never lists still fire PLA rows — the source of the 6502's undocumented opcodes — and a decode behaviour can only be changed by cutting silicon, as Nintendo did when it removed decimal mode from the NES CPU. The PLA is also the single funnel every instruction's control passes through: its 21 output columns bound how many distinct control actions one opcode can raise.

**Governing numbers.** Decode ROM 130 rows × 21 outputs; 131 PLA outputs (pla0–pla130); ~44 datapath control signals (dpc0–dpc43); timing states T0–T5; PC increment across all 16 bits in a single cycle; LDA absolute takes 4 cycles.

**Primary source.** MOS Technology, MCS6502 Microcomputer Family Hardware Manual (January 1976); Visual6502 project, 6502/6507 Decode ROM transcription (NESdev wiki); visual6502 nodenames.js signal catalogue.

#### Inside the signature block: The ALU — two nibble adders with a decimal adjust

**Inside it.** The arithmetic heart is an 8-bit ALU built as two 4-bit nibble adders, not one byte-wide adder. The two operands and the carry-in are added as a pair of nibbles, and the carry-out from bit 3 is adjusted in decimal mode — so the ALU is, in the visual6502 project's words, a pair of binary nibble ALUs with a decimal adjustment. The accumulator feeds one side; the operand arrives from the internal data bus. The D flag in the status register selects binary-coded decimal: each byte then holds two decimal digits, and the adjust logic adds 6 to any nibble that overflows past 9. Decimal mode was one of the chip's selling points over its predecessor the 6800, and it was patent-protected — which is why later clones had to cut it out of the silicon rather than switch it off.

**Trace — one unit of work: adding $99 + $01 in decimal mode (ADC, D flag set, carry clear).**
1. **Setup.** SED sets the D flag; the accumulator holds $99; the carry flag is clear.
2. **Low nibble.** The low nibble ALU adds 9 + 1 = 10 — past 9, so the decimal adjust adds 6: the low nibble becomes 0 and a carry passes to the high nibble.
3. **High nibble.** The high nibble ALU adds 9 + 0 + the nibble carry = 10 — again past 9, so the adjust adds 6: the high nibble becomes 0 and the carry-out is set.
4. **Result.** The accumulator holds $00 with the carry flag set — the correct decimal answer, 99 + 1 = 100, expressed as $00 plus carry. The N, V and Z flags are also set by the operation, but on the NMOS 6502 they do not correspond to a 10's-complement decimal result and must not be branched on after a decimal ADC or SBC.

**Bottleneck in / out.** Relieves the decimal-arithmetic wall: decimal mode saves an instruction and a couple of cycles from each byte of decimal arithmetic, and removes the half-carry from the status byte — scorekeeping, money and game code could add packed BCD natively instead of converting in software. Creates a flag-semantics trap instead: in decimal mode on the NMOS part only the carry flag is usefully set, so N, V and Z after ADC/SBC are misleading; and because the feature was patent-protected, Nintendo's NES CPU (the 2A03) had decimal mode physically removed — five transistors deleted from the polysilicon mask — rather than disabled.

**Governing numbers.** Two 4-bit nibble adders make the 8-bit ALU; decimal mode affects only ADC and SBC; on the NMOS 6502 only the carry flag is usefully set in decimal mode; the NES 2A03 lost decimal mode by the removal of 5 transistors.

**Primary source.** NESdev wiki, Visual6502wiki/6502DecimalMode (visual6502 project); MOS Technology, MCS6502 Microcomputer Family Hardware Manual (January 1976).

### AMD Zen 4

#### Inside the signature block: The op cache and branch predictor — the front end that feeds the machine

**Inside it.** The Zen 4 front end is two prediction structures feeding one decoded-instruction store. The branch predictor predicts up to two taken branches per cycle, consulting a two-level branch target buffer: a 1.5K-entry L1 BTB in which each entry holds up to two targets — the second must sit in the same aligned 64-byte cache line as the first conditional branch — backed by a 7K-entry L2 BTB, plus a larger, improved conditional branch predictor. The predicted fetch address goes first to the op cache — 6,750 decoded macro-ops, 68% larger than Zen 3's 4,096 and able to deliver up to 9 macro-ops per cycle — and only on a miss to the 32 KB L1 instruction cache and the 4-wide decoder, which refills the op cache as it decodes. Between the op cache and the back end sits the micro-op queue, which meters decoded ops out at 6 µops per clock into rename and dispatch. Rename draws on a 224-entry integer physical register file and a 192-entry FP/vector physical register file widened to 512 bits, with a separate 68-entry mask register file for AVX-512 predicates; the reorder buffer holds 320 in-flight instructions.

**Trace — one unit of work: one iteration of a tight loop, already warm in the op cache.**
1. **Predict.** The loop-back branch is predicted from the L1 BTB — two targets per entry let the predictor cover the loop branch and the branch after it in the same line — and with two taken branches predicted per cycle the fetch stream stays ahead of execution.
2. **Op-cache hit.** The fetch address hits the op cache; up to 9 macro-ops of the loop body stream out in one cycle — no decode, no L1I access, no power spent re-decoding the same bytes.
3. **Queue.** The macro-ops land in the micro-op queue, which hands at most 6 µops per clock onward — the op cache's 9-per-cycle burst only matters when the queue has run low.
4. **Rename and allocate.** Each µop is renamed against the 224-entry integer or 192-entry FP physical register files (masks against the 68-entry mask file), takes a slot in the 320-entry reorder buffer, and is assigned to an integer or FP scheduler.
5. **Execute and retire.** The schedulers fire the µops when their operands are ready; results write back through the bypass network and the reorder buffer retires them in order.
6. **Miss path.** On an op-cache miss the same address fetches from the 32 KB L1I and the 4-wide decoder cracks the x86 bytes into macro-ops, filling the op cache so the next loop iteration takes the fast path.

**Bottleneck in / out.** Relieves the decode wall: a 4-wide decoder can only produce four instructions per cycle, so without the op cache the 6-µop dispatch engine would starve on any loop the decoder could not keep up with — and the op cache shortens redirect latency after a correct prediction, because the target's decoded ops are already waiting. AMD credits the wider front end and better branch prediction with 60% of Zen 4's IPC gain (per Tom's Hardware's account of AMD's disclosure). Creates two new ceilings instead: the micro-op queue's fixed 6 µops-per-clock output caps the op cache's 9-per-cycle burst, so the front end's gain shows up as burst refills rather than sustained 9-wide issue; and the 320-entry reorder buffer still bounds the speculation window — a mispredict flushes up to 320 in-flight instructions, and the deeper the window, the more work a wrong prediction discards.

**Governing numbers.** Op cache 6,750 macro-ops (Zen 3: 4,096), up to 9 macro-ops/cycle out; micro-op queue output 6 µops/clock; L1 BTB 2×1.5K entries, two targets per entry; L2 BTB 2×7K entries; 2 taken branch predictions per cycle; 4-wide decode; reorder buffer 320 entries (Zen 3: 256); integer register file 224, FP register file 192 × 512-bit, mask register file 68; L1I 32 KB.

**Primary source.** AMD, "AMD Next Generation 'Zen 4' Core and 4th Gen AMD EPYC 9004 Server CPU", Hot Chips 35, August 2023 (K. Troester & R. Bhargava); AnandTech, "AMD Zen 4 Ryzen 9 7950X and Ryzen 5 7600X Review: Retaking The High-End" (R. Smith & G. Bonshor, September 2022).

#### Inside the signature block: The double-pumped AVX-512 datapath — 512-bit instructions on 256-bit pipes

**Inside it.** Zen 4's floating-point cluster is four 256-bit-wide execution pipes — two FMA (MAC) pipes and two add/multiply pipes — plus a float-to-int pipe and one exception: a dedicated full-width 512-bit shuffle unit. The FP scheduler issues AVX-512 macro-ops to these pipes; a 512-bit op is split into its low and high 256-bit halves and occupies a pipe for two consecutive cycles — double-pumped, in AMD's term — so the datapath never widens and never throttles. Operands live in a 192-entry FP/vector physical register file widened to 512 bits, with a separate 68-entry mask register file holding the per-lane predicates. Feeding the pipes is the load/store unit: two 256-bit loads and one 256-bit store per cycle from the 32 KB, 8-way L1 data cache, behind an 88-entry load queue and a 64-entry store queue; the 1 MB, 8-way L2 sits behind at 14 cycles.

**Trace — one unit of work: one 512-bit FMA, VFMADD132PD zmm0, zmm1, zmm2 (eight double-precision lanes).**
1. **Dispatch.** The instruction decodes to a single macro-op; it is renamed against the 512-bit-wide FP register file (its mask operand against the mask file) and takes a slot in the FP scheduler and the 320-entry reorder buffer.
2. **Issue.** When zmm1 and zmm2 are ready, the FP scheduler issues the macro-op to one of the two 256-bit MAC pipes and reserves that pipe for two consecutive cycles.
3. **Cycle 1 — low half.** Lanes 0–3 (the low 256 bits) flow through the MAC pipe: multiply, then add into the accumulator half.
4. **Cycle 2 — high half.** Lanes 4–7 flow through the same pipe in the next cycle; the mask register suppresses any masked lanes so they neither compute nor write.
5. **Writeback.** The full 512-bit result lands in the FP register file — FMA latency is 4 cycles for the 256-bit form, and the 512-bit form occupies its pipe for a second consecutive cycle — and dependent ops can forward from the bypass network.
6. **Feeding it.** The operands arrived earlier through the load/store unit: a 512-bit vector load takes two cycles on the 256-bit load ports (two 256-bit loads per cycle from L1D), while a 512-bit shuffle, uniquely, takes one cycle on the dedicated full-width shuffle unit.

**Bottleneck in / out.** Relieves the instruction-count and power wall: one 512-bit instruction does the work of two 256-bit ones, so AVX-512 code needs up to 50% fewer instructions than AVX2 (AMD's figure), which cuts fetch, decode and out-of-order tracking power and lets the core hold frequency where full-width implementations throttle — AMD reports a +17% effective-frequency gain on an HPL kernel in power-limited systems. Creates the half-throughput wall instead: 512-bit FP throughput is two instructions per cycle (one FMA plus one FADD), half the four-per-cycle rate at 256-bit width; the load/store unit stays 256-bit per port, so 512-bit memory ops cost two cycles; and gather/scatter are microcoded sequences of roughly four to five macro-ops per element, slow enough that the LLVM cost model treats them as a special case.

**Governing numbers.** 4× 256-bit FP pipes (2 FMA/MAC + 2 ADD/MUL) plus a dedicated 512-bit shuffle unit; most 512-bit ops execute over 2 consecutive cycles; 512-bit FP throughput 2 instructions/cycle (1 FMA + 1 FADD) vs 4/cycle at 256-bit; FADD/FMUL latency 3 cycles, FMA 4 cycles (256-bit forms); FP register file 192 × 512-bit, mask register file 68; load/store 2× 256-bit loads + 1× 256-bit store per cycle; load queue 88, store queue 64; L1D 32 KB 8-way; L2 1 MB 8-way, 14-cycle latency; L3 latency 50 cycles.

**Primary source.** AMD, "AMD Next Generation 'Zen 4' Core and 4th Gen AMD EPYC 9004 Server CPU", Hot Chips 35, August 2023, "AVX-512" and "Power Efficient AVX-512" slides; independent teardown: "Zen4's AVX512 Teardown" (mersenneforum.org, September 2022).

### Intel Golden Cove

#### Inside the signature block: The 6-wide front end — 32-byte fetch, six decoders and the 4K µop cache

**Inside it.** Golden Cove's front end is the biggest genealogical change in the core since Sandy Bridge: everything in it exists to feed a 6-wide back end without paying the power cost of a 6-wide decoder. The branch predictor steers it — an improved direction predictor behind a 12K-entry branch target buffer (up from 5K in Sunny Cove), which Intel says an on-chip ML model grows or shrinks to fit the running workload; independent measurement found the BTB behaves like three levels, with a 1-cycle penalty for missing one level and hitting the next. Fetch comes from a 32 KB L1 instruction cache, unchanged in size but doubled in bandwidth to 32 bytes per cycle (from 16), with the L1 iTLB doubled to 256 entries for 4K pages (32 for 2M/4M huge pages). Two supply paths feed the µop queue: the legacy MITE path — six instruction decoders (up from four), with the old complex-decoder arrangement removed — and the decoded-µop cache (DSB) path, nearly doubled to 4K entries (from 2.25K) and widened to deliver 8 µops per cycle (from 6). The µop queue itself holds 144 entries when one SMT thread runs (72 per thread with both threads active), and the legacy decoder is clock-gated about 80% of the time, with the µop cache doing the feeding instead.

**Trace — one unit of work: one iteration of a tight loop, already warm in the µop cache.**
1. **Predict.** The loop-back branch is predicted from the BTB — up to two taken branches per cycle — and the fetch address is formed.
2. **µop-cache hit.** The address hits the 4K-entry µop cache; up to 8 decoded µops stream out in one cycle — no decode, no L1I access.
3. **Queue.** The µops land in the 144-entry µop queue, which absorbs bursts and smooths the flow into the back end.
4. **Allocate.** The queue meters up to 6 µops per cycle into rename and allocate, where move elimination and zeroing-idiom elimination can remove µops outright before they ever reach an execution port.
5. **Miss path.** On a µop-cache miss the same address fetches 32 bytes per cycle from the L1I and the six decoders crack the x86 bytes into µops, filling the µop cache so the next loop iteration takes the fast path.

**Bottleneck in / out.** Relieves the decode wall: a 4-wide decoder fed by 16 bytes per cycle could not have kept a 6-wide back end busy, and the wider fetch plus the enlarged µop cache let the front end deliver 6 µops per cycle while keeping the power-hungry decoder dark ~80% of the time. Creates two new ceilings instead: the wider machine raises the cost of a mispredict, which is why the BTB grew 2.4× and prediction was reworked — and the measured branch-throughput ceiling is lower than it looks: only about 128 branches can be handled at roughly one per cycle (a regression from Sunny Cove), and returns are slow once calls nest more than two deep, so deep call chains still stall the front end.

**Governing numbers.** Fetch 32 bytes/cycle (Sunny Cove: 16); decoders 6 (from 4); µop cache 4K entries, 8 µops/cycle out (2.25K / 6); µop queue 144 entries single-thread, 72 per thread under SMT; L1I 32 KB; L1 iTLB 256 entries (4K pages), 32 (2M/4M); BTB 12K entries (from 5K); 2 taken branches per cycle; decoder clock-gated ~80% of the time.

**Primary source.** Intel Architecture Day 2021, as reported by AnandTech, "Intel Architecture Day 2021: Alder Lake, Golden Cove, and Gracemont Detailed" (I. Cutress & A. Frumusanu, August 2021) and WikiChip Fuse, "Intel Details Golden Cove" (August 2021); independent measurement: Chips and Cheese, "Popping the Hood on Golden Cove" (December 2021).

#### Inside the signature block: The 12-port back end — unified scheduler, 512-entry ROB and the 3-load memory subsystem

**Inside it.** The back end is wider and deeper in every dimension at once. Allocation is 6-wide into a 512-entry reorder buffer (up from 352 in Sunny Cove, +45%), with rename tricks that collapse work before it is scheduled: move elimination at 6 per cycle and zeroing-idiom elimination, plus execution of simple instructions during rename/allocate to shorten dependency chains. The unified scheduler — one queue for integer and FP/vector math, separate from the address-generation scheduler — holds 205 entries in total (measured), dispatching to 12 execution ports (up from 10). Five of those are integer ALU pipes, the most in any x86 CPU to date, and all five do single-cycle LEA; floating-point and vector work runs on ports 0, 1 and 5, where new fast adders (FADD) on ports 1 and 5 cut FP-add latency to 3 cycles with a 2-cycle bypass for back-to-back adds (previously 4 cycles on ports 0/1 and 6 on port 5). The register files grew to match: the vector/FP file went from 224 to 332 entries, while the integer file stayed at 280 (248 usable plus 32). Feeding it all is the memory subsystem: a third load AGU on new port 11 gives three loads and two stores per cycle — three 256-bit loads or two 512-bit loads per cycle — out of a 48 KB L1 data cache (up from 32 KB) at 5 cycles of latency, behind a 192-entry load queue and a 114-entry store queue (up from 128/72); the data TLB grew to 96 entries, the fill buffer to 16, page walkers doubled from 2 to 4, and the core can now track 48 simultaneous L2 misses (up from 32). The server variant adds AMX, a matrix engine doing 2,048 int8 operations per cycle — 8× the AVX-512 VNNI rate — and a 2 MB private L2 (1.25 MB on client).

**Trace — one unit of work: a 256-bit vector load feeding a floating-point add (VADDPS ymm).**
1. **Allocate.** The load and the add each take a slot in the 512-entry ROB and are renamed against the 332-entry vector register file; the load also takes an entry in the 192-entry load queue.
2. **Address generation.** The load dispatches to one of the three load AGUs (ports 2, 3 or 11), which computes the address; the 96-entry data TLB translates it.
3. **L1D hit.** The 48 KB L1 data cache returns 32 bytes in 5 cycles — up to three such 256-bit loads can be in flight per cycle.
4. **FADD fires.** When the loaded operand is ready, the unified scheduler issues the add to the fast adder on port 1; the result is ready in 3 cycles, and a chained add can use the 2-cycle bypass.
5. **Writeback and retire.** The result lands in the vector register file and forwards over the bypass network to any waiting dependent; both µops retire in program order from the ROB.

**Bottleneck in / out.** Relieves the execution-width and memory-parallelism walls: 12 ports, five ALUs, three load AGUs and a 512-entry window let the core sustain 6-wide issue and keep working through cache misses — by Little's Law the 512-entry ROB covers even the 74-cycle L3 latency at 6 IPC (444 entries needed), which is why the huge ROB exists. Creates a lopsided window instead: the integer register file did not grow, so integer-heavy code runs out of integer registers (248 usable) before the ROB fills, and the 205-entry scheduler — up only 28% while the ROB grew 45% — is the tighter ceiling on how many µops can actually wait for execution. And the bandwidth came with latency: every cache level is slower than Zen 3's (L1D 5 vs 4 cycles, L2 14 vs 12, L3 74 vs 54), so the big ROB is partly paying for the core's own cache latency.

**Governing numbers.** Allocate 6 µops/cycle; ROB 512 entries (Sunny Cove: 352); unified scheduler 205 entries total (measured); integer register file 280 (248 usable + 32, unchanged), vector/FP register file 332 (from 224); branch order buffer 128 (from 96); execution ports 12 (from 10); integer ALUs 5, all with single-cycle LEA; load AGUs 3, store AGUs 2; FADD latency 3 cycles, bypass 2 cycles; FP multiply latency 4 cycles (measured); load queue 192, store queue 114; L1D 48 KB, 5-cycle latency, 3× 256-bit loads/cycle; L2 1.25 MB client / 2 MB server, 14-cycle latency; L3 74 cycles; DTLB 96 entries; page walkers 4 (from 2); outstanding L2 misses 48 (from 32); AMX 2,048 int8 ops/cycle (server).

**Primary source.** Intel Architecture Day 2021, as reported by WikiChip Fuse, "Intel Details Golden Cove" (August 2021) and Tom's Hardware, "Intel Architecture Day 2021" (August 2021); independent measurement: Chips and Cheese, "Popping the Hood on Golden Cove" (December 2021), including its measured ROB/scheduler/register-file capacity table and Little's Law analysis.

### Arm Neoverse V2

#### Inside the signature block: The MOP-cache front end — 1,536-entry L0 MOP cache, 6-wide decode and the TAGE predictor

**Inside it.** The front end exists to keep a 6-wide out-of-order back end fed without paying the power cost of decoding every instruction every time, and it is the most reworked part of the X3-generation design V2 inherits. The dynamic branch predictor steers fetch: Arm doubled the TAGE tables and made the branch-target buffers ten times larger than in V1, doubled the fetch-queue size, and doubled instruction-TLB and cache bandwidth. Fetch reads a 64 KB, 4-way set-associative L1 instruction cache with 64-byte lines, behind a fully associative L1 instruction TLB covering 4 KB, 16 KB, 64 KB and 2 MB pages. Two supply paths feed the rename stage. The cold path is six AArch64 decoders cracking instructions into internal macro-ops (MOPs). The hot path is the L0 macro-op cache: a 1,536-entry, 4-way skewed-associative store of already-decoded, already-optimised MOPs — on a hit, a loop's MOPs skip decode and the L1I entirely and flow straight to the 8-wide rename/dispatch stage. Dispatch feeds a 320-entry reorder buffer through per-class issue queues. Arm's own accounting for the whole front-end rework: 13% higher SPECspeed 2017 integer performance and 10.5% fewer system-level cache misses than V1.

**Trace — one unit of work: one instruction in a hot loop, already warm in the MOP cache.**
1. **Predict.** The TAGE predictor and the enlarged branch-target buffer predict the loop-back branch; the fetch address is formed.
2. **MOP-cache hit.** The address hits the 1,536-entry MOP cache; the loop's decoded MOPs flow out to rename/dispatch — no decode, no L1I access.
3. **Rename & dispatch.** MOPs are renamed against the physical register file and dispatched up to 8 per cycle into the per-class issue queues; each takes a slot in the 320-entry reorder buffer.
4. **Issue & execute.** When operands land, the issue queues fire the MOPs to the execution pipes — integer, load/store, or the four 128-bit SVE2 pipes.
5. **Miss path.** On a MOP-cache miss the same address fetches from the 64 KB L1I and the six decoders crack the AArch64 bytes into MOPs, filling the MOP cache so the next loop iteration takes the fast path.

**Bottleneck in / out.** Relieves the decode wall: six decoders alone could not sustain 8-wide dispatch, and decoding every iteration of a hot loop burns power for identical work — the MOP cache lets the front end feed the back end while the decoders sit idle. Creates two new ceilings instead: the MOP cache itself shrank by half from the X2 generation (3,072 entries to 1,536), so loops larger than about 1.5K MOPs spill back onto the decode path; and a wider machine raises the cost of a mispredict, which now flushes a 320-entry in-flight window — which is why the predictor hardware (TAGE tables, branch-target buffers) grew so much faster than the decoders.

**Governing numbers.** MOP cache 1,536 entries, 4-way skewed associative (X2: 3,072); L1I 64 KB, 4-way, 64-byte lines; L1 I-TLB fully associative (4 KB / 16 KB / 64 KB / 2 MB); decode 6-wide; rename/dispatch 8-wide; ROB 320 entries; vs V1: branch-target buffers 10× larger, TAGE tables 2×, fetch queue 2×, instruction-TLB/cache bandwidth 2×; +13% SPECspeed 2017 integer, −10.5% system-level cache misses vs V1.

**Primary source.** Arm, Arm Neoverse V2 Core Technical Reference Manual, doc 102375_0001_02_en, Issue 02 (L1 instruction memory system; register rename; instruction issue); Arm Community blog, "Arm Neoverse V2 Platform — Best-in-class cloud and AI/ML performance" (Hot Chips 2023); Wikipedia, "ARM Cortex-X3" (decode width, rename/dispatch width, ROB, MOP-cache size — the X3-generation design V2 inherits, citing Arm/AnandTech).

#### Inside the signature block: The platform fabric — CPU bridge, DSU-110 and the CMN-700 mesh with its distributed system cache

**Inside it.** This is the block the entry's special piece hangs on: V2 is sold as a Compute Subsystem, and the fabric is the subsystem. Each core exits through its own asynchronous CPU bridge — one bridge per core, buffering and synchronising between the core clock domain and the cluster — into the DSU-110, which manages all of the core's external interfaces to the SoC. In Arm's CSS reference platform the DSU-110 hangs off the CMN-700 coherent mesh: a two-dimensional grid of mesh crosspoints (XPs) that route packets, configurable up to a 12×12 mesh, with home nodes (HN-F) that each carry a slice of the distributed system-level cache plus an integrated point-of-serialisation and point-of-coherency, and a snoop filter tracking which cores hold which lines — up to 128 HN-Fs, 0–512 MB of system cache in total, and up to 1,024 MB of snoop-filter tag RAM. Requests and responses travel on separate 256-bit data channels, with an optional dual-channel configuration. The one big fork in the story: NVIDIA's Grace does not use this fabric at all — its 72 V2 cores hang off NVIDIA's own Scalable Coherency Fabric, a mesh with a distributed cache delivering 3.2 TB/s of bisection bandwidth, because Grace's single-NUMA-node, LPDDR5X design wanted its own coherency fabric. Same core, two fabrics — the platform, not the core, decides the memory system.

**Trace — one unit of work: one L2 miss on the Arm CSS path.**
1. **L2 miss.** A load misses the core's private 1 MB (or 2 MB) L2 and the request is handed to the CPU bridge.
2. **Clock-domain crossing.** The asynchronous CPU bridge buffers the request and synchronises it into the DSU-110/cluster domain.
3. **Mesh transit.** The DSU-110 injects the request into the CMN-700 mesh; it hops crosspoint to crosspoint toward the home node that owns the address.
4. **Home node decides.** The HN-F — the point of serialisation and coherency for that address — checks its slice of the distributed system cache and consults the snoop filter for sharers.
5. **Hit or memory.** On a system-cache hit the data returns along the mesh; on a miss the HN-F forwards the request to the memory controller (DDR5 or LPDDR5X) and the line is filled into the cache slice on its way back to the core.

**Bottleneck in / out.** Relieves the many-core coherence wall: a single shared bus or a central cache could never serve 72–256 cores, so coherence is distributed — every address has a home node, every HN-F owns a cache slice, and the snoop filter means most requests never broadcast. Creates a fabric wall instead: hop latency grows with mesh size, a badly hashed address range can hot-spot one HN-F slice, and the mesh — not the core — becomes the limiter on cross-core traffic, which is why Arm offers the dual 256-bit data channels and why NVIDIA replaced the fabric entirely with SCF for Grace. The 10.5% cut in system-level cache misses Arm reports for V2 versus V1 is the payoff for the bigger front end and the deeper system cache working together.

**Governing numbers.** Mesh up to 12×12 (144 crosspoints); up to 256 cores per die, 512 per system; system cache 0–512 MB total across up to 128 HN-Fs; snoop-filter tag RAM up to 1,024 MB; data channels 256-bit, single or dual; Arm launch claim: up to 4 TB/s across the CMN mesh; Grace: 72 cores, 1 MB L2 per core, 114 MB L3, SCF 3.2 TB/s bisection, NVLink-C2C 900 GB/s to the second die.

**Primary source.** Arm, Arm Neoverse V2 Core Technical Reference Manual, doc 102375_0001_02_en (L2 memory system; CPU bridge; DSU-110 interfaces); Arm, Arm Neoverse CMN-700 Coherent Mesh Network Technical Reference Manual, doc 102308_0307_08_en, Issue 08 (Features §2.3); NVIDIA, "NVIDIA Grace CPU" product page (72 cores, SCF 3.2 TB/s bisection, spec table: 1 MB L2 per core, 114 MB L3, NVLink-C2C 900 GB/s); NVIDIA Developer Blog, "NVIDIA Grace Hopper Superchip Architecture In-Depth" (117 MB L3 figure — see the disputed fact record).

### Arm Cortex-X4

#### Inside the signature block: The 10-wide decode front end — eight decoders, no MOP cache

**Inside it.** The front end exists to feed a 10-wide rename/dispatch stage, and X4's answer to that problem is the generation's most argued-about decision: delete the macro-op cache entirely. Fetch starts at a rebuilt instruction-fetch unit with a dynamic branch predictor, feeding a 64 KB, 4-way set-associative L1 instruction cache with 64-byte lines, behind a fully associative L1 instruction TLB with native support for 4 KB, 16 KB, 64 KB and 2 MB pages. Arm says it redesigned the single-instruction cache, folding the old MOP-cache approach into it with additional fused instructions, and tuned the front end for applications with large instruction footprints — the branchy, unpredictable phone code that dominates real workloads rather than synthetic loops. Behind the cache sit eight parallel decoders that crack fixed-length AArch64 instructions into internal macro-ops (MOPs); because some instructions crack into two MOPs, eight decoders feed up to 10 MOPs per cycle into rename and dispatch. The 1,536-entry MOP cache that X3 carried is gone — zero entries — so there is no cache-hit/cache-miss timing split anywhere in the front end: every instruction takes the same decode path, and the pipeline is a uniform 10 stages with a uniform 10-cycle mispredict penalty. That uniformity is only buildable because AArch64 instructions are fixed-length — there is no variable-length pre-decode stage to widen — which is the same reason the entry's special piece calls the ISA itself the front-end technology.

**Trace — one unit of work: one instruction in a hot loop, steady state.**
1. **Predict.** The dynamic branch predictor predicts the loop-back branch; the fetch address is formed.
2. **Fetch.** The 64 KB L1 instruction cache delivers the 4-byte instruction; the fully associative I-TLB has already translated the page, so the hot path never waits on translation.
3. **Decode.** One of the eight decoders cracks the instruction into MOP(s); a complex instruction can crack into two, which is how eight decoders sustain 10-wide dispatch.
4. **Rename & dispatch.** MOPs are renamed and dispatched up to 10 per cycle into the per-class issue queues; each takes a slot in the 384-entry reorder buffer.
5. **Execute & retire.** The issue queues fire the MOPs to the 21 execution ports; results return and the ROB retires instructions in order.

**Bottleneck in / out.** Relieves the decode wall and the MOP-cache timing split: X3's front end was 6-wide decode / 8-wide dispatch behind a 1,536-entry cache with two different latencies depending on hit or miss; X4 has one uniform path at twice the dispatch width. Creates two new ceilings instead: every iteration of a hot loop now re-decodes from scratch — the MOP cache's power and throughput saving is gone, paid in decoder toggles on every pass — and eight decoders plus 10-wide rename cost area and power, about 10% more core area than X3 at equal cache. A wider machine also makes a mispredict more expensive, flushing a 384-entry in-flight window, which is why the predictor and the fetch unit were rebuilt for large instruction footprints rather than benchmark loops.

**Governing numbers.** 8 decoders; rename/dispatch 10 MOPs per cycle (X3: 6-wide decode / 8-wide dispatch); MOP cache 0 entries (X3: 1,536); ROB 384 entries, 768 MOPs in flight with fused pairs (X3: 320 / 640); L1I 64 KB, 4-way, 64-byte lines; L1 I-TLB fully associative (4 KB / 16 KB / 64 KB / 2 MB); pipeline 10 stages; mispredict penalty 10 cycles.

**Primary source.** Arm, Arm Cortex-X4 Core Technical Reference Manual, doc 102484_0002_05_en, Issue 05 (Technical overview — L1 instruction memory system; Instruction decode; Register rename; Instruction issue); Wikipedia, "ARM Cortex-X4" (decode width, rename/dispatch width, ROB, MOP-cache removal — citing WikiChip Fuse, Arm Community and AnandTech); HWCooling, "ARM unveils record-breaking Cortex-X4 core with eight ALUs" (eight decoders, 10-cycle penalty); Android Authority, "Arm Cortex-X4, A720, and A520: 2024 smartphone CPUs deep dive" (redesigned instruction cache, fused instructions, large-footprint tuning).

#### Inside the signature block: The 21-port back end — eight integer ALUs and the 384-entry window

**Inside it.** The back end is where X4 spent its new width: 21 execution ports, up from 15 in X3. The integer cluster is the headline — eight ALUs (six simple single-cycle ALUs, one ALU/MAC and one ALU/MAC/DIV), up from six, with the old MUL unit upgraded to a full MAC so the core now has two integer MAC units where X3 had one; three branch units, up from two; and four address-generation units, up from three, after Arm split one load/store unit into dedicated load and store pipes so the core can sustain up to four memory operations per cycle. Behind them sits the 384-entry reorder buffer — a 768-MOP in-flight window when fused pairs are counted, up from 640 — fed by the 10-wide rename/dispatch stage through per-class issue queues. The vector side deliberately did not widen: the four 128-bit NEON/SVE2 pipes are unchanged from X3 (configurable as 2×128 or 4×128 at build time), though floating-point divide and square root are now pipelined. The memory system around the back end is new L1 temporal data prefetching, an optional doubling of the L1 data TLB, and a private L2 that doubles to a 2 MB option with no added latency — the L2 is private to the core, 8-way set associative, configurable at 512 KB, 1 MB or 2 MB, and reaches the DSU-120 cluster through an asynchronous CPU bridge. The DSU-120 takes up to 14 cores per cluster with up to 32 MB of shared L3.

**Trace — one unit of work: one integer multiply-accumulate instruction.**
1. **Dispatch.** The renamed MOP enters the integer issue queue and takes a slot in the 384-entry ROB.
2. **Wait.** The queue watches for its operands — register-ready, or a forwarded result from an earlier ALU.
3. **Fire.** The MOP issues to one of the two MAC-capable ALU ports; the multi-cycle MAC executes while the six simple ALUs keep taking single-cycle work around it.
4. **Broadcast.** The result returns to the ROB and wakes dependent MOPs waiting in the issue queues.
5. **Commit.** The instruction retires in order from the ROB; up to 768 MOPs may be in flight behind it.

**Bottleneck in / out.** Relieves scalar throughput: integer-heavy phone code — address arithmetic, loop control, UI work — now has eight ALUs, three branch units and four memory ops per cycle instead of six, two and three. Creates the vector imbalance instead: FP/SIMD-heavy code gains little, because the four 128-bit pipes are exactly where X3 left them, so the widening is a scalar bet. And the 768-MOP window means a mispredict or a long-latency load stalls a much bigger in-flight set than X3's 640 — which is why the back end grew a temporal prefetcher, a doubled D-TLB option and a 2 MB L2: the window is only useful if the memory system keeps it fed. Arm's accounting for the whole widening: 13% IPC uplift over X3 at the same process, clock and L3, 15% higher peak performance at 3.4 GHz, and up to 40% lower power at the same performance point.

**Governing numbers.** 21 execution ports (X3: 15); 8 integer ALUs — 6 simple + ALU/MAC + ALU/MAC/DIV (X3: 6, one MAC); 3 branch units (X3: 2); 4 AGUs, up to 4 memory ops per cycle (X3: 3); ROB 384 entries / 768 MOPs in flight (X3: 320 / 640); vector 4×128-bit NEON/SVE2 (2×128 config option); L1D 64 KB, 4-way, 64-byte lines; L1 D-TLB fully associative (4 KB / 16 KB / 64 KB, 2 MB / 512 MB blocks); L2 private, 8-way, 512 KB / 1 MB / 2 MB; DSU-120: up to 14 cores, L3 up to 32 MB; +13% IPC ISO-process, +15% peak performance at 3.4 GHz, up to 40% lower power at the same performance.

**Primary source.** Arm, Arm Cortex-X4 Core Technical Reference Manual, doc 102484_0002_05_en, Issue 05 (Technical overview — L1 data memory system; L2 memory system; Vector execute; configuration options); Wikipedia, "ARM Cortex-X4" (execution ports, ROB, DSU-120, performance claims — citing WikiChip Fuse, Arm Community and AnandTech); Android Authority, "Arm Cortex-X4, A720, and A520: 2024 smartphone CPUs deep dive" (ALU/branch table, 768-MOP window, load/store split, temporal prefetcher, DSU-120 power modes); HWCooling, "ARM unveils record-breaking Cortex-X4 core with eight ALUs" (ALU split, AGU count).

### IBM POWER10

#### Inside the signature block: The eight-slice execution engine — 512-entry instruction table feeding eight slices

**Inside it.** The SMT8 core is two SMT4 halves side by side. Each half has its own 48 KB L1 instruction cache and 32 KB L1 data cache and four 128-bit execution slices; the whole core has eight slices, a private 2 MB L2, an 8 MB region of the 120 MB on-die L3, and a 4,096-entry TLB. Each slice is a self-contained engine: an ALU, an FPU, a branch unit, a load-store unit and a 128-bit VSX SIMD unit, holding up to 20 in-flight instructions. All eight slices are fed from one shared 512-entry instruction table, backed by a 128-entry load queue and an 80-entry store queue (halved per thread when all eight threads run). The front end is the Instruction Fetch Unit (IFU): it fetches, decodes and dispatches every instruction — including 64-bit prefixed instructions and MMA instructions — into the table. Branch prediction accuracy roughly doubled over POWER9, which matters because the table is a 512-entry in-flight window: a mispredict flushes a lot of work. The design point: four vector pipelines per core plus the slice array, aimed at ~30% more core-level throughput than POWER9 at the same clock, and 2.6× the core-level energy efficiency on SPECint.

**Trace — one unit of work: one integer add instruction, steady state.**
1. **Fetch.** The IFU fetches the instruction from the thread's half's 48 KB L1 instruction cache; the I-TLB has already translated the page.
2. **Decode & dispatch.** The instruction is decoded (prefix/fuse handling for 64-bit prefixed forms) and dispatched into the shared 512-entry instruction table, taking one entry.
3. **Wait.** The instruction sits in the table until its operands are ready — register-ready, or a forwarded result from an earlier slice execution.
4. **Issue.** When operands are ready and a slice has a free slot (at most 20 in flight per slice), the instruction issues to one of the eight slices.
5. **Execute.** The slice's ALU runs the add; a load would instead go through the 128-entry load queue to the 32 KB L1 data cache, a store through the 80-entry store queue.
6. **Writeback & complete.** The result is written back and wakes dependent instructions waiting in the table; the instruction completes in program order.

**Bottleneck in / out.** Relieves POWER9's issue bottleneck: eight slices fed by a 512-entry window keep eight SMT threads' work flowing where a narrower core stalled on issue. Creates the shared-table ceiling instead: at full SMT8 all eight threads contend for the same 512 table entries, the same eight slices and the same load/store queues, so per-thread IPC falls as threads are added; and the table's wakeup/select logic is a centralized power and area cost that grows with the window — the reason the per-slice cap is 20 in flight, not unlimited. A long-latency load stalls its slice's 20 slots behind it, which is why the core grew a 2 MB private L2 and a 4,096-entry TLB: the window is only useful if the memory system keeps it fed.

**Governing numbers.** 8 slices (4 per SMT4 half); 512-entry shared instruction table; 20 in-flight instructions per slice; load queue 128 entries / store queue 80 entries (64 / 40 single-threaded); L1I 2×48 KB, L1D 2×32 KB; L2 2 MB private per core; L3 region 8 MB per core, 120 MB per die; TLB 4,096 entries; branch prediction accuracy ~2× POWER9; +30% core-level throughput vs POWER9 at equal clock, 3× at socket level (IEEE Micro); 2.6× core-level energy efficiency on SPECint, up to 3× at socket (ISCA 2021, projected).

**Primary source.** Starke, Thompto, Stuecheli & Moreira, "IBM's POWER10 Processor", IEEE Micro 41(2), 7–14 (2021), DOI 10.1109/MM.2021.3058632; Starke & Thompto, "IBM's POWER10 Processor", Hot Chips 32 (2020), DOI 10.1109/HCS49909.2020.9220618; Qiu et al., "Energy Efficiency Boost in the AI-Infused POWER10 Processor", ISCA 2021; Wikipedia, "Power10" (512-entry table, 20 per slice, 128/80 queues — citing the Hot Chips 32 deck); IT Jungle, "Drilling Down Into The Power10 Chip Architecture" (Aug 24, 2020) (two SMT4 halves, four slices and 48 KB I-cache / 32 KB D-cache per half).

#### Inside the signature block: The Matrix Math Engine — four MMA units, eight 512-bit accumulators

**Inside it.** Four Matrix Math Engines (MME) per SMT8 core execute Power ISA 3.1's Matrix-Multiply Assist (MMA) instructions. The MMA facility adds eight 512-bit accumulator registers, ACC[0:7]; each holds a 4×4 matrix of FP32, a 4×2 matrix of FP64, or a 4×4 matrix of INT32 elements. Each accumulator is associated with a group of four 128-bit vector-scalar registers (VSR[0:31]); while an accumulator is in use its four VSRs must not be used, while VSR[32:63] are never associated with an accumulator and stay free. The accumulators live inside the MME functional unit itself, not in the register file — IBM's stated reason is that keeping the partial sums in the unit significantly reduces switching power. MMA instructions are fetched, decoded and dispatched by the IFU like any other instruction and can overlap fully with other execution in the out-of-order core. The operation is a rank-k update: A ← A + X·Yᵀ, an outer product. For FP64, xvf64ger updates a 4×2 accumulator from a 4-element and a 2-element vector (8 fused multiply-adds, 16 FLOPs); for INT4, xvi4ger8 updates a 4×4 INT32 accumulator from two 4×8 matrices (128 multiply-accumulates, 256 ops). Four engines × 512 result bits = 2,048 result bits per cycle: 64 FP64 FLOPs per cycle, up to ~1,024 INT4 ops per cycle per core. Data types span FP64, FP32, FP16, BF16, INT16, INT8 and INT4.

**Trace — one unit of work: one xvf64ger double-precision outer product through the MME.**
1. **Prime.** xxsetaccz zeroes ACC0 (or xxmtacc assembles it from four VSRs); ACC0 is now primed and its four backing VSRs are locked out until it is deprimed.
2. **Fetch, decode, dispatch.** The IFU fetches the xvf64ger instruction like any other and dispatches it into the 512-entry instruction table.
3. **Wait & issue.** When the two 128-bit operand VSRs (one 4-element, one 2-element FP64 vector) are ready, the instruction issues to one of the four MMEs.
4. **Outer product.** The MME computes A ← A + X·Yᵀ: the 4×2 outer product, 8 fused multiply-adds, added into ACC0 in place — 16 FLOPs from one instruction.
5. **Stay in the unit.** The partial sum never leaves the MME: the next ger in the K-loop accumulates into the same ACC0, so a whole K-slice of the GEMM runs without a register-file write.
6. **Drain.** When the tile is done, xxmfacc moves ACC0 back into its four VSRs (or a paired store writes it to memory); the VSR group is free for reuse.

**Bottleneck in / out.** Relieves the BLAS3 wall: one ger instruction replaces a loop of vector FMAs and the register traffic that went with them — 4× the per-core matrix performance of POWER9 at constant frequency, and 10–20× the socket throughput on matrix-intensive work. Creates the accumulator ceiling instead: only eight accumulators, so a GEMM kernel can keep at most eight 4×4/4×2 tiles live, and each live accumulator locks out four VSRs, so accumulator pressure and VSR starvation bound tile size and unrolling. The four MMEs are shared across eight SMT threads, so a second thread's MMA stream halves the per-thread rate; and because the engine is in-core, peak matrix rate is bounded by the core's issue bandwidth and the L1/L2 feed, not by a separate accelerator's memory.

**Governing numbers.** 4 MME units per core; 8 accumulators × 512 bits; accumulator shapes 4×4 FP32 / 4×2 FP64 / 4×4 INT32; 64 VSRs × 128 bits — VSR[0:31] in four-register accumulator groups, VSR[32:63] always free; 2,048 matrix-result bits per cycle; 64 FP64 FLOPs/cycle per core; ~1,024 INT4 ops/cycle per core; 4× per-core matrix performance vs POWER9 at constant frequency; 10–20× matrix-intensive socket throughput vs POWER9; projected 10× FP32 / 21× INT8 socket AI performance on ResNet-50 and BERT-Large.

**Primary source.** Moreira et al., "A matrix math facility for Power ISA™ processors", arXiv:2104.03142 (April 2021); IBM, "Matrix-Multiply Assist Best Practices Guide", Redpaper REDP-5612-00 (April 2021); HPCwire, "IBM Debuts Power10" (Aug 18, 2020) (Thompto's 64 FP64 FLOPs/cycle and ~1K reduced-precision ops/cycle figures); Starke et al., IEEE Micro 41(2) (10–20× matrix-intensive); Qiu et al., ISCA 2021 (10× FP32 / 21× INT8 projections).

### SiFive Performance P670

#### Inside the signature block: The 4-wide rename/dispatch fabric — 160-entry ROB, four dispatch buffers, nine issue queues

**Inside it.** SiFive's own pipeline-overview diagram (launch, November 2022) shows the out-of-order heart as a widened P550 descendant: a front end that fetches up to 8 instructions per cycle from a 32 KB instruction cache, steered by a branch predictor built from four named pieces — a Next Line Predictor, a Branch Direction Predictor, a Return Address Stack and an Indirect Jump Target Predictor. Decode, rename and dispatch run 4-wide into a 160-entry reorder buffer, with rename pools sized per register class: 208 integer renames, 208 floating-point renames and 2×64 vector renames. From there the machine fans out through four dispatch buffers — Integer, Load/Store, Floating Point, Vector — each feeding its own cluster's issue queues: four 16-entry queues in the integer cluster, one 24-entry queue in the load/store cluster, two 16-entry queues in the floating-point cluster and two 16-entry queues in the vector cluster — 152 issue-queue entries in total. The integer cluster has two ALUs and two branch units, with MUL, DIV and int-to-float conversion sitting under the first ALU; the load/store cluster has two address-generation units over two load/store pipes, a 24-entry issue queue fed by two inlets, prefetchers and a 32 KB data cache; the floating-point cluster has two FMAC, two FADD and two FMUL pipes plus int-to-float and FP divide; the vector cluster has two 128-bit ALUs plus DIV and a vector-to-scalar unit. Behind all of it sits a private L2 of up to 512 KB per core and a shared L3 of up to 16 MB, linked at 32 bytes per cycle — the same 32 B/cycle width feeds the instruction cache, the L2-to-L3 link and the data-cache-to-L2 link. SiFive publishes no details beyond the diagram — no load/store queue depths, no branch-predictor table sizes, no cache latencies — and this section stops where the diagram does.

**Trace — one unit of work: one integer ADD, steady state.**
1. **Predict.** The Next Line Predictor and Branch Direction Predictor (with the Return Address Stack covering returns and the Indirect Jump Target Predictor covering indirect jumps) form the fetch address.
2. **Fetch.** Up to 8 instructions stream out of the 32 KB instruction cache in one cycle — twice the 4-wide decode rate, so the fetch buffer absorbs short fetch stalls.
3. **Decode, rename, dispatch.** The ADD is decoded in the 4-wide front end, its architectural registers are renamed against the 208-entry integer rename pool, and it takes a slot in the 160-entry reorder buffer.
4. **Dispatch buffer.** The renamed instruction is routed to the Integer Dispatch Buffer.
5. **Wait.** It sits in one of the four 16-entry integer issue queues until its operands are ready — register-ready, or a forwarded result from an earlier ALU.
6. **Fire.** The queue issues it to one of the two ALUs; a multiply, divide or int-to-float op would instead take the MUL/DIV/I2F path under the first ALU.
7. **Writeback and retire.** The result broadcasts back, wakes dependent instructions waiting in the issue queues, and the ADD retires in program order from the ROB.

**Bottleneck in / out.** Relieves the P550's 3-wide ceiling: SiFive widened the whole core to four instructions per cycle and enlarged the buffers throughout, so the front end can sustain 4-wide issue where the P550 capped at three. Creates the 160-entry ceiling instead: the reorder buffer is the in-flight window, modest beside Cortex-X4's 384 entries, so a mispredict flushes up to 160 instructions and a long-latency load stalls the window behind it — which is why the core grew a private L2 with a streaming prefetcher and cache stashing to L3. And the front end is deliberately asymmetric — it fetches 8 per cycle but decodes 4 — so decode, not fetch, is the sustained throughput cap; the fetch margin exists to refill the window fast after a redirect.

**Governing numbers.** Fetch up to 8 instructions/cycle; I-cache 32 KB; decode/rename/dispatch 4-wide; ROB 160 entries; rename pools 208 integer / 208 FP / 2×64 vector; issue queues 4×16 integer, 1×24 load/store, 2×16 FP, 2×16 vector (152 entries total); integer cluster 2 ALU + 2 branch; load/store 2 AGU, 2 LD/ST pipes, D-cache 32 KB; private L2 up to 512 KB; shared L3 up to 16 MB; links 32 B/cycle; pipeline 13 stages; over 3.4 GHz at 5 nm; over 13 SpecINT2k6/GHz (SiFive's current P600 page; the launch figure was over 12).

**Primary source.** SiFive, "SiFive Performance P670 detailed pipeline overview" diagram (November 2022, via CNX Software); SiFive, Performance P670 product brief, rev 1.0 (2025); SiFive, Performance P600 Series product page (sifive.com/cores/performance-p600).

#### Inside the signature block: The dual 128-bit vector cluster — two ALUs, DLEN = VLEN

**Inside it.** The vector cluster is the piece that makes the P670 a different core from its sibling the P650, which is the same scalar machine with the vector unit deleted. The brief specifies it precisely: two 128-bit vector ALUs, DLEN = VLEN — the datapath width equals the architectural vector length of 128 bits — implementing the ratified RISC-V Vector v1.0 specification with vector-length-agnostic operation and dynamic runtime setting of vector lengths, across INT8/16/32/64 and FP16/32/64, plus the vector cryptography extensions (the first RISC-V vector crypto on the market, per SiFive's launch release). The cluster is fed by its own Vector Dispatch Buffer and two 16-entry issue queues, one per ALU, with a DIV unit and a V2S (vector-to-scalar) unit beside the ALUs. Because DLEN equals VLEN, one vector instruction on a full 128-bit register is one pass through one ALU — 16 INT8 lanes, 8 INT16, 4 INT32/FP32, or 2 INT64/FP64 per cycle per ALU — and the two ALUs let two independent vector instructions execute in the same cycle. The vector registers are renamed from a dedicated 2×64-entry pool, separate from the scalar rename pools. What SiFive does not publish: the vector register file's port counts, the DIV latency, and how vector loads move through the load/store cluster — the diagram shows the structure, not the cycle counts.

**Trace — one unit of work: one vector add, `vadd.vv v0, v1, v2`, 16 INT8 elements, VLEN = 128.**
1. **Decode and rename.** The instruction is decoded in the 4-wide front end; its vector registers are renamed against the 2×64 vector rename pool, and the current vl/vtype setting (set dynamically at runtime) is attached to the instruction.
2. **Dispatch.** It is routed to the Vector Dispatch Buffer.
3. **Wait.** It sits in one of the two 16-entry vector issue queues until v1 and v2 are ready.
4. **Fire.** The queue issues it to one of the two 128-bit vector ALUs; all 16 INT8 lanes are added in a single pass, because the 128-bit datapath matches the 128-bit vector length.
5. **Dual issue.** In the same cycle the second ALU can take an independent vector instruction from the other issue queue — two vector ops per cycle, not one wide one.
6. **Writeback and retire.** The 128-bit result lands in the vector register file, wakes dependent vector ops, and the instruction retires in program order from the 160-entry ROB. A vector divide would instead take the DIV path; a move of a vector element to a scalar register takes the V2S path.

**Bottleneck in / out.** Relieves the P650's missing-vector gap: AI/ML, media and DSP work that needed a separate accelerator or scalar loops now runs on the core itself, with vector crypto in the same unit — the reason SiFive positions the P670 against the Cortex-A78 for wearables, robotics and networking. Creates the 128-bit width wall instead: with DLEN = VLEN = 128, wider vectors (via LMUL register grouping) cost multiple passes, and a 256-bit-equivalent operation takes two cycles where a 256-bit datapath would take one. And the vector unit shares the scalar machine's oxygen — the 4-wide front end, the 160-entry ROB and the load/store cluster — so a vector-heavy stream competes with scalar work for dispatch slots and for the two load/store pipes feeding it; the streaming prefetcher and cache stashing to L3 exist to keep those pipes from starving.

**Governing numbers.** 2× 128-bit vector ALUs; DLEN = VLEN = 128 bits; vector issue queues 2×16 entries; vector rename pool 2×64; per-cycle per-ALU throughput 16× INT8 / 8× INT16 / 4× INT32 or FP32 / 2× INT64 or FP64 lanes; datatypes INT8–64, FP16/32/64; RVV 1.0 + vector crypto; private L2 up to 512 KB, shared L3 up to 16 MB.

**Primary source.** SiFive, Performance P670 product brief, rev 1.0 (2025) ("2x128-bit ALU, DLEN = VLEN"; datatype list); SiFive, "SiFive Performance P670 detailed pipeline overview" diagram (November 2022, via CNX Software); SiFive press release, "SiFive's New High-Performance Processors Offer a Significant Upgrade for Wearable and Consumer Products" (November 1, 2022).

---

## Basic blocks primer — the primitives (2026-11-22)

The shared Level-3 vocabulary of the zoo: the fundamental circuit primitives every signature block is built from, defined once so entries point here instead of re-defining them. Each primitive: what it is and what it does, its key parameters, the trade-offs in choosing it, and where it typically appears in real chips.

### FIFO / queue (Queueing & flow)

**What it is.** A first-in-first-out buffer: data enters one end and leaves the other, in order. It decouples a producer and a consumer that run at different rates or in bursts, absorbing jitter so neither stalls the other.

**Key parameters.** Depth (entries) × width (bits); almost-full / almost-empty thresholds; synchronous vs asynchronous (dual-clock) ports.

**Trade-offs.** Depth costs SRAM or flops plus latency — too shallow and the producer stalls (or packets drop); too deep and you pay area for buffering you rarely use. Async FIFOs add synchronizer latency on their flags.

**Appears in.** Everywhere data crosses a rate boundary: NoC routers, PCIe/DMA engines, GPU command queues, switch ingress, UARTs.

### Credit counter (Queueing & flow)

**What it is.** Flow control as accounting: the sender keeps a counter of free slots in the receiver's buffer, spends one credit per flit sent, and earns credits back as the receiver drains. At zero, sending stops — losslessness by arithmetic, no drops, no retransmits.

**Key parameters.** Credit count (= the receiver buffer depth it mirrors); flit width; credit return latency (the round trip sets the minimum useful buffer depth); one counter per virtual channel.

**Trade-offs.** Buffers must cover the credit round-trip latency or links idle waiting for credits; per-VC counters multiply state. The alternative — drop and retry — trades this state for loss-recovery complexity.

**Appears in.** InfiniBand and NoC link flow control, PCIe / CXL flow control, on-chip router virtual channels, lossless Ethernet alternatives to PFC.

### Mux / demux (Switching & selection)

**What it is.** A multiplexer picks one of N inputs to drive one output under a select signal; a demux does the reverse, steering one input to one of N outputs. The mux is the atom of all data selection — every ‘choose’ in hardware is muxes.

**Key parameters.** Radix (2:1, 4:1, N:1); data width; select encoding (binary vs one-hot).

**Trade-offs.** Wide, high-radix muxes are trees of 2:1 muxes — delay grows with log(radix) and select fanout becomes the timing problem. One-hot selects are faster but cost select wires and decode area.

**Appears in.** Operand bypass/forwarding in CPUs, ALU input select, register-file write steering, and the mux inside every scan flop.

### Crossbar (Switching & selection)

**What it is.** A full any-to-any switch: every input can reach every output in the same cycle, built as one output mux per output fed by all inputs, plus arbitration when two inputs want the same output.

**Key parameters.** Radix N×M (inputs × outputs); data width per port; bisection bandwidth = ports × width × clock.

**Trade-offs.** Cost grows as N×M — crosspoints and wiring explode quadratically, so big crossbars lose to buffered meshes and NoCs. At small radix nothing beats its one-cycle, no-buffering latency.

**Appears in.** CPU bypass networks, NoC router internals, GPU interconnect at small radix, memory-controller scheduling fabrics.

### Switch arbiter (Switching & selection)

**What it is.** Decides which requester wins a shared resource each cycle — a crossbar output, a bus, a memory port. Round-robin, weighted, fixed-priority and matrix arbiters trade fairness against simplicity.

**Key parameters.** Requester count; grant latency (combinational vs registered/pipelined); fairness policy; QoS weights.

**Trade-offs.** A combinational arbiter sits on the critical path of everything it serves; pipelining the grant costs a cycle. Starvation-free policies cost state — pointers and counters per port.

**Appears in.** NoC routers, shared-cache bank access, memory controllers, switch schedulers — Tomahawk 5's traffic manager is an arbiter running at 51.2 Tbps of aggregate demand.

### Register file (Storage)

**What it is.** A small, very fast array of registers with multiple read and write ports — the operand store instructions actually compute from. Built from flops/latches with a mux tree per read port, not from dense SRAM bitcells.

**Key parameters.** Entries (depth) × width; read-port and write-port count; banking; the bypass/forwarding network attached to it.

**Trade-offs.** Area and power grow roughly with ports × entries — each added read port adds a full mux tree per bit. Port count, not capacity, is what makes register files expensive; banking trades port count for bank conflicts.

**Appears in.** CPU physical register files, GPU SM register files (256 KB per SM in Hopper), DSP accumulator files.

### SRAM bank (Storage)

**What it is.** One compiled SRAM macro: a dense bitcell array with row decoders, sense amplifiers and one or two ports. Big on-chip memories are many banks interleaved, so parallel accesses land in different banks instead of colliding.

**Key parameters.** Capacity (rows × width); port style (1RW, 2RW, true dual-port); bank count and interleave granularity; access latency in cycles.

**Trade-offs.** Single-port banks are densest but serialize reads against writes; more ports and banks buy bandwidth with area and bank-conflict stalls. SRAM is usually the largest single consumer of die area on an accelerator.

**Appears in.** Caches, scratchpads (TPU VMEM, Cerebras per-core SRAM), packet buffers, and any FIFO too deep to build from flops.

### CAM / TCAM (Storage)

**What it is.** Content-addressable memory: instead of presenting an address, you present a key and every entry compares itself in parallel, returning the match and its index. TCAM adds a third state — ‘don't care’ — per bit, enabling masked and prefix matches.

**Key parameters.** Entries × key width; match latency (usually one cycle); priority rule when several entries match; TCAM mask width.

**Trade-offs.** Every entry burns compare power on every lookup — CAMs are the most power-hungry memory per bit in the zoo and stay small. Match-line timing caps both size and clock.

**Appears in.** TLBs, cache tag arrays, out-of-order wakeup tag stores, switch/router forwarding and ACL tables, flow classification.

### Adder tree (Arithmetic)

**What it is.** A tree of adders reducing many terms to one sum in log2(N) levels — the reduction heart of dot products. Multipliers produce partial products; a compressor (Wallace/Dadda) tree crushes them first, then a final adder finishes.

**Key parameters.** Number of inputs; input and output width (guard bits grow toward the root); tree depth in levels; pipelining between levels; compressor vs plain-adder style.

**Trade-offs.** The root adder is the widest and slowest element; pipelining the tree adds latency and registers. Compressor trees save area and power but complicate layout and verification.

**Appears in.** Tensor Core and systolic accumulation, SIMD reductions, population count, ECC/checksum logic, MISR compaction in test hardware.

### Multiplier array (Arithmetic)

**What it is.** Many multipliers working in parallel on a tile of data — from a row of MAC lanes to a full 2-D systolic grid. The defining arithmetic structure of the AI era: peak throughput = multipliers × clock, and everything else in the chip exists to feed it.

**Key parameters.** Array dimensions (e.g. 128×128); operand precision (FP8 / FP16 / BF16 / INT8); accumulation precision (usually FP32); dataflow (weight-, output- or input-stationary).

**Trade-offs.** Multipliers dominate accelerator area and power; precision is the lever — halving operand width roughly quarters multiplier area. Arrays starve without operand reuse, so an array is only as good as its feeding network.

**Appears in.** TPU MXU (128×128 systolic), GPU Tensor Cores, DSP MAC clusters, image-filter convolvers.

### Barrel shifter (Arithmetic)

**What it is.** Shifts a word by any amount in one cycle: log2(W) stages of muxes, where stage k shifts by 2^k when that bit of the shift amount is set. Variable shifts without iterating.

**Key parameters.** Data width W; shift range; direction (left / right / rotate); arithmetic vs logical fill.

**Trade-offs.** Pure mux area — roughly W×log2(W) muxes — sitting on the ALU critical path, so wide shifters often earn their own pipeline stage. Cheap at 32 bits; at 512-bit vector widths it is built per-lane instead.

**Appears in.** ALU shift/rotate, address generation, floating-point alignment and normalization, bitfield extract, SerDes word alignment.

### Comparator (Arithmetic)

**What it is.** Compares two values and reports equal / greater / less — usually by riding a subtractor's carry chain. One comparator is a small thing; thousands in parallel become a CAM, a wakeup network, or a max-tree.

**Key parameters.** Width; signed vs unsigned; latency (folded into the carry chain); count, in parallel structures.

**Trade-offs.** Width costs carry delay; magnitude compare is pricier than equality. Parallel compare arrays pay power per compare, per cycle.

**Appears in.** Branch resolution, TLB/cache tag match, OoO wakeup tag broadcast, min/max and amax tracking (the Hopper Transformer Engine's scaling depends on it), packet classification.

### Priority encoder (Control & sequencing)

**What it is.** Given a vector of requests, outputs the index of the highest-priority one that is set — ‘who is first?’ as a circuit. It is the selection half of arbiters, interrupt controllers and schedulers.

**Key parameters.** Input width (requesters); fixed vs rotating (round-robin) priority; output encoding (binary index + valid bit).

**Trade-offs.** A naive ripple priority chain is O(N) delay; tree structures fix timing at area cost. Rotating priority needs a mask plus two encoders. Wide priority encoders sit on scheduler critical paths.

**Appears in.** Arbiter grant logic, interrupt controllers, oldest-ready instruction select in OoO schedulers, first-fail capture in diagnosis.

### Counter (Control & sequencing)

**What it is.** A register plus an incrementer, usually with a comparator against a terminal value: the simplest state machine in the zoo. It counts cycles, beats, packets, credits and retries — time made visible to logic.

**Key parameters.** Width in bits (sets max count); up / down / saturating; loadable preset; terminal-count compare.

**Trade-offs.** Nearly free individually; the cost is the compare and the control fanout when a terminal count gates other blocks. Wide counters on fast clocks need carry-lookahead, like adders.

**Appears in.** Loop and beat counters in DMA and tensor sequencers, refresh timers, performance counters, watchdogs and timeouts — and it is the arithmetic core inside every credit counter.

### FSM / controller (Control & sequencing)

**What it is.** A finite-state machine: a state register, next-state logic and output logic stepping through the legal sequence of a protocol or datapath. Every block whose description contains a ‘sequence’ has an FSM — or a counter pretending to be one.

**Key parameters.** State count and encoding (one-hot vs binary); inputs and outputs per state; Mealy vs Moore outputs.

**Trade-offs.** One-hot FSMs are fast and area-hungry; binary-encoded FSMs are dense but slower to decode. FSM bugs are protocol bugs — the verification burden, not the gate count, is the real cost.

**Appears in.** Cache and DRAM controllers, DMA engines, Cerebras dataflow firing control, test controllers, power-management sequencing.

### Pipeline register (Clocking & sync)

**What it is.** The register bank between two combinational stages: it slices a long logic path into clocked steps so the clock can run faster, at the cost of one cycle of latency per slice. Pipelining is just pipeline registers, placed with intent.

**Key parameters.** Width (bits stored); stage position (retiming); enable/stall and flush/bubble controls.

**Trade-offs.** Registers cost area and clock power on every bit, every cycle — and stall/flush control multiplies when a whole pipeline must freeze coherently. Latency vs frequency is the eternal trade.

**Appears in.** Every CPU/GPU pipeline stage boundary, systolic-array skew registers in the TPU MXU, SerDes DSP pipelines.

### Synchronizer (Clocking & sync)

**What it is.** A chain of (usually two) flops that safely samples a signal crossing into a different clock domain, giving metastability time to resolve. Single bits cross through synchronizers; buses cross through async FIFOs or handshakes instead.

**Key parameters.** Stages (2 or 3 flops); MTBF target; single-bit vs multi-bit crossing method.

**Trade-offs.** Each crossing costs 2–3 cycles of latency in the receiving domain — synchronizers are why async-FIFO flags and CDC handshakes feel slow. More stages buy reliability exponentially and cost latency linearly.

**Appears in.** Async FIFO pointers (Gray-coded, then synchronized), reset release, interrupt and status crossings, test/mission clock-mode switching.


---

## Level-3 decomposition — Down to the primitives (2026-11-22)

The six Anatomy Lab landmark entries decomposed one level below the Level-2 drill-down: for each signature block, the primitives it is built from and what each primitive does there. Governing parameters are stated only where primary sources publish them; where a vendor publishes no primitive-level parameters, the decomposition names the primitive and its role and says so plainly.

### NVIDIA Hopper H100

#### Down to the primitives: Fourth-generation Tensor Core — and the SM partition around it

- **Multiplier array** — The Tensor Core itself: a dense matrix-multiply-accumulate array multiplying the A fragment (from registers or shared memory) by the B fragment (from shared memory) into FP32 accumulators. Published: four Tensor Cores per SM, one per partition. Not published: the array's internal dimensions or per-cycle MAC count as a primitive spec — NVIDIA publishes throughput per SM, not array geometry.
- **Adder tree** — FP32 accumulation inside the MMA: partial products reduce through the array's adder network into accumulators that live in registers across the warp's 32 threads. Tree depth, compressor style and pipelining are not published.
- **Register file** — Holds A fragments and the FP32 accumulators. Published: 256 KB per SM (verified — fact records), shared across the four partitions. Per-partition port counts are not published.
- **SRAM bank** — The shared-memory half of the combined L1/shared block stages tiles and feeds the B operand with no register-file traffic. Published: 256 KB combined per SM, shared memory configurable up to 228 KB (verified — fact records). Bank count, width and port style are not published.
- **Mux** — Operand collectors mux operands between the register file, shared memory and immediate paths into the array's inputs — the selection layer that lets one instruction feed a whole tile multiply. Collector structure and mux radix are not published.
- **FIFO / queue** — Each partition's scheduler holds ready warps in issue queues between dispatch and execution, and the Tensor Memory Accelerator (TMA) sequences tile copies from descriptors instead of per-thread loads. Published: the TMA exists and moves large blocks between global and shared memory, including cluster copies (verified — new fact record). Queue depths are not published.
- **FSM / controller** — One warp scheduler plus dispatch unit per SM partition — four per SM — picks a ready warp each cycle. It is a priority-select controller, not a scoreboarded CPU scheduler. The structure is published in the whitepaper; the select policy and queue sizes are not.
- **Pipeline register** — Registers stage the shared-memory read, the MMA sweep and the register writeback; the warp-group (wgmma) pipeline is deep enough that warps overlap it entirely. Stage counts are not published.
- **Synchronizer** — Hopper's asynchronous transaction barrier tracks TMA transactions and thread arrivals so consumers wait on data movement, not on instructions — the cluster-scale synchronizer primitive. Published as a feature (verified — new fact record: ‘for doing atomic data movement and synchronization’). Barrier count and transaction-table depth are not published.

**Parameter honesty.** Parameter honesty: NVIDIA publishes the SM-level totals used above (register file, shared memory, Tensor Cores per SM) and the TMA/barrier features; everything below that level — array geometry, queue depths, collector structure — is not published, and this decomposition says so rather than estimating it.

#### Down to the primitives: Transformer Engine — the rescaling wrapped around the Tensor Core

- **Comparator** — The engine's bookkeeping is a running maximum: the software stack tracks each tensor's amax (maximum absolute value) history to choose its FP8 scale. That is a comparator reduction maintained per tensor — no dedicated hardware max-tree is documented by NVIDIA, and none is claimed here.
- **Multiplier** — One scaling multiply on the way in — values are scaled as they cast to FP8 — and the inverse scale applied to the FP32 accumulators on the way out. The multipliers live in the conversion datapath around the Tensor Core; their count and width are not published beyond the FP8 formats themselves (E4M3 / E5M2, per tensor, per layer — Level-2).
- **Mux** — A format-select mux chooses the FP8 flavour (E4M3 vs E5M2) and the FP8 vs 16-bit path per tensor — NVIDIA describes the engine as dynamically choosing between FP8 and 16-bit calculations per layer. Mux structure is not published.
- **Counter / register state** — Per-tensor scale factors and amax history are carried between layers as small state — architecturally a register/counter file, but held by the software stack (Transformer Engine library with cuDNN), not a documented hardware table. Sizes are not published.

**Parameter honesty.** The Transformer Engine is deliberately thin at the primitive level: a comparator-fed scale choice, two multiplies and a format mux around an unchanged multiplier array. That thinness is the design — the novelty is where the scaling sits, not new arithmetic.

### Google TPU v4

#### Down to the primitives: MXU — the 128×128 systolic array inside a TensorCore

- **Multiplier array** — The MXU itself: a 128×128 grid of multiply-accumulate cells — 16,384 MACs per MXU, four MXUs per TensorCore, two TensorCores per chip (verified — ISCA 2023, fact records). Each cell is one multiplier and one adder holding a stationary weight.
- **Adder tree — as a chain** — Accumulation here is deliberately NOT a tree: each cell adds its product to the partial sum arriving from its neighbour, so the GPU-style adder tree becomes a 128-long adder chain woven through the array. The chain's pipelining is the systolic flow itself.
- **Pipeline register** — Activations are skewed by per-row delay registers so each element meets the right cell on the right cycle, and partial sums pipeline cell to cell; the array's fill latency is its 128 steps. The ISCA paper describes the systolic dataflow; it does not publish a register-level schematic, so no per-cell register counts are stated.
- **SRAM bank** — VMEM stages operands for the array: 16 MiB per TensorCore; the two TensorCores share 128 MiB of CMEM (verified — ISCA 2023). Bank counts, widths and port styles are not published.
- **FIFO / queue** — DMA queues feed VMEM from HBM and drain results to it; the XLA-compiled schedule is built on the assumption that they never empty mid-tile. Queue depths are not published.
- **FSM / controller** — A scalar control unit sequences the MXU and VPU from the compiled schedule — counters stepping through tile loops rather than dynamic, scoreboarded scheduling. The paper names the scalar unit's role; its internal structure is not published.
- **Mux** — Operand muxes at the array edges select between VMEM, CMEM and VPU results. Radix and structure are not published.

**Parameter honesty.** Parameter honesty: the ISCA 2023 paper publishes the array geometry and memory sizes used above; below that level — queue depths, bank structure, sequencer internals — Google publishes nothing, and this decomposition invents nothing.

#### Down to the primitives: SparseCore — the dataflow engine for embeddings

- **FIFO / queue** — Outstanding embedding gathers queue inside the SparseCore while HBM latency is absorbed — the queueing is the point of the block. Depths are not published (new fact record: unverifiable — the ISCA paper discloses the SparseCore's role and effect, not its internal buffers).
- **Counter** — Index counters walk each feature's embedding index list and issue the gathers. Counter widths and issue rates are not published.
- **Adder tree** — Returned embedding vectors are reduced — summed or averaged — as they stream back, in a streaming adder reduction. Width and tree depth are not published.
- **Mux / steering** — Because embedding tables are sharded across the pod, gathered vectors steer over the ICI torus toward the TensorCore that needs them — muxing at pod scale. Per-chip steering structure is not published.

**Parameter honesty.** Published anchor: SparseCores accelerate embedding-reliant models 5×7× for about 5% of die area and power (verified — ISCA 2023 abstract). Everything internal to the SparseCore beyond that is unpublished.

### Apple Firestorm — M1

#### Down to the primitives: The wide front end and the ~630-entry reorder buffer — at the primitive level

- **FIFO / queue** — The reorder buffer is architecturally a circular FIFO: instructions allocate at the tail and retire in order at the head. Depth ~630 entries is an AnandTech microbenchmark measurement — Apple publishes no ROB figure (unverifiable — fact records).
- **Register file** — The rename file holds physical registers for renamed operands: roughly 354 integer and 384 floating-point registers, measured by microbenchmarks. Apple publishes neither the counts nor the port count (unverifiable).
- **CAM** — Issue queues wait on producer tags: each waiting entry compares broadcast result tags against its source tags — a small CAM per queue, the wakeup half of wakeup/select. Queue partitioning and entry counts are inferred from measurements (~148–154 outstanding loads, ~106 stores), not published.
- **Priority encoder** — The select half: when more instructions are ready than there are ports, an oldest-first priority select grants issue. Its width and structure are not published.
- **Mux** — Producer results bypass to consumer inputs through forwarding muxes at the execution ports — the wiring that makes a 630-entry window usable instead of decorative. Mux radix is not published.
- **Comparator** — Branch prediction fronts the whole machine: direction/target comparators and counter tables decide the fetch stream eight instructions wide. Predictor structure — table sizes, counter widths, whether it is TAGE-like — is not published; Apple discloses no predictor structure at all.
- **Barrel shifter / adder tree** — Inside the measured four 128-bit NEON pipes and the integer ALUs sit the standard shifter and adder primitives. Per-unit structure is not published.
- **Counter** — Fetch sequencing runs ahead on predicted paths, fed by a 192 KB L1 instruction cache (reported via Darwin kernel sources, per AnandTech). Fetch queue depth is not published.

**Parameter honesty.** Parameter honesty: Apple has published no primitive-level parameter for Firestorm — no FIFO depth, port count, table size or mux radix in this decomposition comes from Apple. Every number is a microbenchmark measurement and carries status unverifiable in the fact records (new Level-3 record included). Naming the primitives is safe — a reorder buffer IS a circular queue, wakeup IS a CAM — sizing them from outside is not.

### Cerebras WSE-2

#### Down to the primitives: The dataflow core — and its 5-port router

- **SRAM bank** — 48 KB of local SRAM per core in eight single-ported banks, each 32-bit wide (verified — fact records). Banking is how a single-ported technology feeds a wider datapath: the banks together sustain two full 64-bit reads and one full 64-bit write per cycle (verified — new fact record).
- **Register file** — 16 general-purpose registers, plus 44 data-structure registers (DSRs) holding tensor descriptors — pointer, length, shape, size (both verified — new fact records). Port counts are not published.
- **Multiplier array — a row, not a grid** — Four FP16 FMAC units on a 64-bit datapath (verified — new fact record), inside a compact six-stage pipeline (verified — new fact record). The wafer gets its ‘array’ by replicating this small row 850,000 times, not by widening it.
- **FSM / controller** — Hardware state machines sequence each tensor operation from its DSR descriptor, and dataflow control fires instructions when fabric data arrives — the core also juggles eight simultaneous tensor operations (‘micro-threads’, verified — new fact record) under a priority-aware scheduler. State counts and encodings are not published.
- **Comparator** — Incoming wavelets carry a color; matching the arriving color against the operation waiting on it is the trigger compare that fires computation. A wavelet is 32 bits — one FP16 data element plus 16 bits of control (verified — new fact record).
- **Mux — the router** — Each core's fabric router is a 5-port design: 32-bit bidirectional interfaces in the four cardinal directions and one port facing the core itself (verified — new fact record). Routing is static, with 24 independent routes — colors — time-multiplexed per link (verified — new fact record), so route selection was done by the compiler; there is no dynamic crossbar arbiter in the packet path.
- **FIFO / queue** — The fabric is lossless with, in Cerebras' words, very low buffering — shallow FIFOs at the router ports with backpressure instead of drops. Exact FIFO depths are not published.
- **Credit counter** — Losslessness on static routes is maintained with minimal buffering; whether the accounting is credit-based per color is not published. Cerebras discloses the lossless property, not the counter structure — stated plainly rather than assumed.

**Parameter honesty.** Parameter honesty: unusually for this zoo, Cerebras' Hot Chips 34 deep dive (IEEE Micro 43(3)) publishes genuine primitive-level numbers — bank organization, register counts, FMAC count, router ports, wavelet format, color count — and the new fact records added in this pass capture them with exact quotes. FIFO depths and port counts of the register file remain unpublished.

### Broadcom Tomahawk 5 (BCM78900)

#### Down to the primitives: The shared packet buffer — and the traffic manager around it

- **SRAM bank** — One packet buffer pooled across all 64 ports (verified — fact records): the buffer's aggregate SRAM bandwidth, not a crossbar, does the any-to-any work. NOT published: capacity in megabytes, cell size, bank count and port structure (new fact record: unverifiable — Broadcom discloses the shared-buffer architecture and its latency claims only).
- **Crossbar — deliberately absent** — A shared-buffer switch replaces the crossbar: a packet is written once and read once, and buffer bandwidth substitutes for an N×N switch matrix. Naming the absence matters — it is the primitive-level difference between Tomahawk and VOQ/crossbar designs elsewhere in this zoo.
- **FIFO / queue** — Per-port, per-class queues hold packets between the buffer and their scheduler grant, with dynamic admission thresholds on the shared pool standing in for static partitions. Queue counts per port and depths are not published.
- **Switch arbiter** — The traffic manager arbitrates queued packets onto each egress port at line rate — arbitration across 64 × 800G of demand, fed by 512 × 100G PAM4 SerDes lanes. The arbitration algorithm and its grant pipelining are not published.
- **TCAM / hash tables** — The single-pass parse → lookup stage resolves forwarding, ECMP and single-pass VxLAN bridging/routing. Merchant-switch lookups are hash tables plus TCAM-class structures, but Broadcom does not publish Tomahawk 5's table sizes or TCAM capacities — not published, and not estimated here.
- **Comparator** — Cognitive Routing compares utilization on the hashed path against alternatives, locally and downstream, and steers flowlets to lighter links (verified — fact records). Thresholds and counter widths are not published.
- **Counter** — In-band telemetry inserts metadata at line rate recording queue size, packet latency and switch utilization (verified — new fact record) — all of it read from hardware counters sampled per packet. The counter inventory is not published.
- **FSM / controller — as processors** — Six on-chip Arm processors run streaming telemetry and embedded applications such as statistics summarization (verified — new fact record): general-purpose control processors beside the fixed pipeline, deliberately not in the packet path.

**Parameter honesty.** Parameter honesty: Broadcom publishes radix and rates (64 × 800GbE; 256 × 200GbE also supported — new fact record), the shared-buffer architecture and the feature set. Buffer capacity, cell size, queue depths and table sizes — the numbers a Level-3 decomposition most wants — are not published, and this entry leaves them blank rather than filling them with the industry folklore numbers that circulate without a primary source.

### Scan Design & ATPG Architecture

#### Down to the primitives: The scan chain — and the EDT wrapped around it

- **Mux** — The founding primitive of the whole entry: the mux-D scan flop is a 2:1 mux before every flip-flop's D input, selecting functional data in mission mode and scan data in test mode. Its count is the design's flop count — there is no architecture-level number, by definition.
- **FIFO / shift register** — A scan chain is the flops themselves reconfigured as a serial shift register — a FIFO whose depth is the chain length. Chain count and length are DFT-insertion and floorplan choices made per design, not constants of the architecture (new fact record: unverifiable as architecture-level values).
- **Adder tree — as an XOR tree** — EDT response compaction XORs hundreds of chain outputs down into a MISR signature — an adder-tree cousin built from XOR gates. The compression ratio (typically an order of magnitude or more) is a tool and design setting, not a fixed parameter.
- **Comparator** — The shifted-out response, or the MISR signature, is compared against the simulated good-machine value — one wide equality compare per pattern, on the ATE or on-chip.
- **FSM / controller** — A test-mode controller sequences shift and capture, and the on-chip clock controller (OCC) generates the single at-speed capture pulse between slow shifts. These state machines are generated per design by DFT tools (Tessent-class); no canonical state count exists.
- **Counter** — Shift counters track position within the chain; pattern counters track ATPG vectors applied. Their widths follow chain length and pattern count — per design.
- **Synchronizer** — Test clocks and functional clocks cross under controlled clock muxing and synchronization so slow shift and at-speed capture never glitch into each other. The implementation is tool-inserted, per design.

**Parameter honesty.** Parameter honesty: scan is the one entry in this pass where ‘parameters not published’ is structural, not secretive — the architecture defines the primitives (mux, chain, compactor, controller) and every size is a property of the design being tested, exactly as the Level-2 section states for chain counts and coverage sign-off.

**New fact records added in this pass** (appended to `files/fact-records.json`; statuses per the fact standard):

- [verified] **NVIDIA Hopper H100** — Tensor Memory Accelerator (TMA) function. Source: NVIDIA Hopper Architecture In-Depth (NVIDIA).
- [verified] **NVIDIA Hopper H100** — Asynchronous transaction barrier function. Source: NVIDIA Hopper Architecture In-Depth (NVIDIA).
- [verified] **Cerebras WSE-2** — Fabric router port design. Source: Cerebras Architecture Deep Dive: First Look Inside the Hardware/Software Co-Design for Deep Learning (Cerebras).
- [verified] **Cerebras WSE-2** — Fabric packet (wavelet) format. Source: Cerebras Architecture Deep Dive: First Look Inside the Hardware/Software Co-Design for Deep Learning (Cerebras).
- [verified] **Cerebras WSE-2** — Static routes (colors) per fabric link. Source: Cerebras Architecture Deep Dive: First Look Inside the Hardware/Software Co-Design for Deep Learning (Cerebras).
- [verified] **Cerebras WSE-2** — Core tensor datapath composition. Source: Cerebras Architecture Deep Dive: First Look Inside the Hardware/Software Co-Design for Deep Learning (Cerebras).
- [verified] **Cerebras WSE-2** — General-purpose registers and pipeline depth. Source: Cerebras Architecture Deep Dive: First Look Inside the Hardware/Software Co-Design for Deep Learning (Cerebras).
- [verified] **Cerebras WSE-2** — Data-structure registers (DSRs) per core. Source: Cerebras Architecture Deep Dive: First Look Inside the Hardware/Software Co-Design for Deep Learning (Cerebras).
- [verified] **Cerebras WSE-2** — Per-core SRAM bandwidth per cycle. Source: Cerebras Architecture Deep Dive: First Look Inside the Hardware/Software Co-Design for Deep Learning (Cerebras).
- [verified] **Cerebras WSE-2** — Simultaneous tensor operations (micro-threads) per core. Source: Cerebras Architecture Deep Dive: First Look Inside the Hardware/Software Co-Design for Deep Learning (Cerebras).
- [verified] **Broadcom Tomahawk 5 (BCM78900)** — On-chip Arm processors for telemetry. Source: Broadcom Ships Tomahawk 5, Industry's Highest Bandwidth Switch Chip to Accelerate AI/ML Workloads (Broadcom (GlobeNewswire)).
- [verified] **Broadcom Tomahawk 5 (BCM78900)** — In-band telemetry metadata content. Source: Broadcom Ships Tomahawk 5, Industry's Highest Bandwidth Switch Chip to Accelerate AI/ML Workloads (Broadcom (GlobeNewswire)).
- [verified] **Broadcom Tomahawk 5 (BCM78900)** — Maximum 200GbE port radix. Source: Broadcom Ships Tomahawk 5, Industry's Highest Bandwidth Switch Chip to Accelerate AI/ML Workloads (Broadcom (GlobeNewswire)).
- [unverifiable] **Broadcom Tomahawk 5 (BCM78900)** — Shared packet buffer capacity, cell size and queue depths. Source: Broadcom Ships Tomahawk 5, Industry's Highest Bandwidth Switch Chip to Accelerate AI/ML Workloads (Broadcom (GlobeNewswire)).
- [unverifiable] **Google TPU v4** — SparseCore internal queue depths and buffer sizes. Source: TPU v4: An Optically Reconfigurable Supercomputer for Machine Learning with Hardware Support for Embeddings (ISCA 2023) (Google / ACM).
- [unverifiable] **Apple Firestorm — M1** — Primitive-level parameters (issue-queue partitioning, predictor tables, bypass radix, register-file ports). Source: Apple Silicon M1 / A14 Deep Dive (AnandTech).
- [unverifiable] **Scan Design & ATPG Architecture** — Scan chain count/length and EDT compression ratio as architecture-level constants. Source: IEEE Std 1500 / IEEE Std 1149.1 test-access standards (IEEE).

### Intel 4004

#### Down to the primitives: The address register — program counter and 3-level stack as a dynamic RAM array

- **Dynamic RAM cell array (storage)** — The block itself is storage, not logic: a 4 × 12-bit array of dynamic cells — one level holds the program counter, three form the subroutine stack (verified — MCS-4 User's Manual, fact records). Because the cells are dynamic, every level must be re-read to hold its charge, which is why the block carries its own refresh counter.
- **Mux** — The stack address has two sources and one decoder: the effective address counter (which stack level is current) and the refresh counter (which level is being re-read) are multiplexed to the address-register decoder (verified — MCS-4 User's Manual: "The stack address is provided by the effective address counter and by the refresh counter, and it is multiplexed to the decoder."). The same muxing idea recurs at the index register, whose address comes from the internal bus or its refresh counter.
- **Counter** — Two counters steer the array: the effective address counter steps the stack pointer on call and return, and the refresh counter scans every location so no dynamic cell loses its charge — the reason reset must be held for at least 8 instruction cycles (64 clock periods) before the CPU is usable (verified — fact records). Counter widths are not published in the manual.
- **Register (address buffer)** — The address, once read out of the array, sits in a 12-bit address buffer (the "buffer register" in Figure 3) before it is sent anywhere — the staging register between the RAM array and the bus.
- **Demux** — The buffer's 12 bits are demultiplexed onto the 4-line internal bus in three 4-bit slices at A1, A2 and A3 (verified — MCS-4 User's Manual). The demux is the pin-count trick in hardware form: one 4-bit slice per clock period instead of twelve parallel wires.
- **Adder (carry-look-ahead incrementer)** — A 4-bit carry-look-ahead circuit advances the address by one after each slice is sent out, and the incremented address is written back into the array (verified — fact records). Note the asymmetry with the compute adder: the incrementer gets look-ahead because it sits in the critical fetch path, while the data adder is ripple-through-carry.
- **FSM / controller** — The timing and SYNC generator sequences the eight clock periods (A1–A3, M1–M2, X1–X3) that pace the read, the three slices, the increment and the writeback; the double-cycle flip-flop stretches the sequence to 16 clocks for the five double-length instructions (verified — MCS-4 User's Manual). The generator's state count and phase logic are not published as a schematic.
- **Amplifier & mux (output driver)** — Figure 3 labels an "AMPLI. & MUX" block on the address path: the slices are amplified and multiplexed onto the D0–D3 pins, and the data-bus output buffers are tri-state — "1", "0" and floating — so only one block drives a line at a time (verified — MCS-4 User's Manual).

**Parameter honesty.** Parameter honesty: the manual publishes the array geometry (4 × 12 bits), the 4-bit slice width, the 8/16-clock cycle counts and the reset duration; it publishes no counter widths, no cell schematic, no decoder structure and no transistor counts, and this decomposition invents none of them.

#### Down to the primitives: The 4-bit adder, accumulator and decimal-adjust ROMs

- **Adder (ripple-through-carry)** — Four 1-bit full adders chained carry-to-carry: the carry out of each bit feeds the next, so the sum settles in four carry steps. One operand comes from the ADB register, the other from the accumulator plus the carry flip-flop (verified — MCS-4 User's Manual: "The 4-bit adder is of the ripple-through carry type."). A ripple chain, not a tree or a look-ahead network — the cheapest adder that fits, and the reason every operation wider than one digit is a software loop.
- **Register (ADB buffer register)** — The ADB register stages the bus operand for the adder: the index register's 4 bits arrive over the internal bus, land in ADB, and only then enter the adder. 4 bits wide, published as a named block in Figure 3.
- **Register file (dynamic RAM array)** — The operands themselves live in the 16 × 4-bit index register array — a second dynamic RAM array, read through a multiplexer onto the internal bus (verified — fact records). The compute block's register file is literally a tiny RAM, refreshed by the same kind of counter as the address register.
- **Mux & shifter** — The accumulator carries a mux and shifter that implement rotate left and rotate right (verified — MCS-4 User's Manual): a 4-bit rotate is a 4-way selection — each bit takes its neighbour's value, with the end bit taking the carry — so the shifter is four 2:1 muxes, not a barrel shifter. Figure 3 labels it "MUX & SHIFTER".
- **ROM (lookup table)** — The special ROMs are the block's clever primitive: a lookup table that performs code conversion for the DAA and KBP instructions (verified — MCS-4 User's Manual). The accumulator value addresses the ROM and the ROM outputs the corrected digit — decimal adjust done by table, not by arithmetic. The ROMs' dimensions are not published in the manual.
- **Comparator** — The condition logic is a zero/carry compare: it senses ADD = 0 and ACC = 0 (a 4-bit NOR on each), the carry flip-flop's state and the external TEST pin, and its output sets the condition flip-flop that gates JCN and ISZ (verified — MCS-4 User's Manual). No separate compare instruction exists because the comparator watches every result for free.
- **Flip-flop (1-bit registers)** — The block's state is single bits: the carry flip-flop holds the adder's overflow between digit steps, the condition flip-flop holds the branch decision, and the double-cycle flip-flop holds whether the current instruction needs a second cycle (verified — MCS-4 User's Manual: "A double cycle FF is set from any one of 5 double-length instructions.").
- **FSM / controller** — The instruction register (OPR + OPA, each 4 bits) is loaded from the internal bus through a multiplexer at M1 and M2, and the instruction decoder gates the decoded opcode with timing signals to produce the control signals for every functional block (verified — MCS-4 User's Manual). The decoder's internal structure — PLA, ROM or random logic — is not described in the manual, and this decomposition does not claim one.

**Parameter honesty.** Parameter honesty: the manual publishes the datapath widths (4-bit adder, 4-bit accumulator, 3-bit command-control code), the 45-instruction count and the 850 µs figure for adding two 8-digit numbers; it publishes no ROM dimensions, no decoder structure, no gate or transistor counts, and this decomposition invents none of them.

**New fact records added in this run** (appended to `files/fact-records.json`; statuses per the fact standard):

- [verified] **Intel 4004** — Stack-address mux: effective address counter and refresh counter multiplexed to the address-register decoder. Source: Intel MCS-4 User's Manual (1973), §III.
- [verified] **Intel 4004** — Address buffer demultiplexed onto the internal bus in three 4-bit slices (A1, A2, A3). Source: Intel MCS-4 User's Manual (1973), §III.
- [verified] **Intel 4004** — ADB register supplies one adder operand from the internal bus. Source: Intel MCS-4 User's Manual (1973), §III.
- [verified] **Intel 4004** — Accumulator shifter implements rotate right and rotate left. Source: Intel MCS-4 User's Manual (1973), §III.
- [verified] **Intel 4004** — Special ROMs perform code conversion for DAA and KBP instructions. Source: Intel MCS-4 User's Manual (1973), §III.
- [verified] **Intel 4004** — Condition logic senses ADD = 0, ACC = 0, carry FF and TEST to implement JCN and ISZ. Source: Intel MCS-4 User's Manual (1973), §III.
- [verified] **Intel 4004** — Instruction register (OPR + OPA) loaded from the internal bus through a multiplexer at M1/M2. Source: Intel MCS-4 User's Manual (1973), §III.
- [verified] **Intel 4004** — Double-cycle flip-flop set by any of the 5 double-length instructions. Source: Intel MCS-4 User's Manual (1973), §III.
- [verified] **Intel 4004** — Data-bus output buffers are tri-state ("1", "0", floating); only one buffer drives a line at a time. Source: Intel MCS-4 User's Manual (1973), §II, Figure 2.
- [unverifiable] **Intel 4004** — Primitive-level parameters not published in the manual: refresh/effective-address counter widths, special-ROM dimensions, instruction-decoder structure, cell/transistor counts. Source: Intel MCS-4 User's Manual (1973).

### MOS Technology 6502

#### Down to the primitives: The decode PLA and the timing sequencer — the whole control unit in one array

- **ROM array (the decode PLA)** — The control unit's memory is a mask-programmed ROM, not a RAM: a 130-row by 21-column array in which each row is one opcode-pattern-plus-timing-state combination. A row fires when the instruction-register bits, the cooked group input G and the current timing state T all match its pattern, and raises one of 131 PLA outputs (pla0–pla130) (verified — Visual6502wiki/6507 Decode ROM: "The Decode ROM is a 130x21 bits ROM in the 6502 that is used to decode the instruction and to control various units of the CPU."). The PLA is the only place the chip "knows" what an instruction is: there is no opcode table in software and no microcode store, just pattern rows.
- **Decoder (the G group signals)** — The opcode's low two bits are cooked into three group signals before they reach the ROM: G1 = IR0, G2 = IR1, G3 = !IR0 & !IR1 (verified — Visual6502wiki/6507 Decode ROM). An opcode ending in 11 turns on both G1 and G2, so every G=1 and G=2 row for its pattern fires at once — the mechanism behind the undocumented opcodes: LDA ($AD, G=1) and LDX ($AE, G=2) merge into LAX ($AF) because both rows fire (verified — same source). The decoder is three gates' worth of combinational logic, and it is the whole reason the 6502 has illegal opcodes at all.
- **Register (instruction register + predecode register)** — The opcode byte is latched into the 8-bit instruction register (IR), with inverted IR outputs available to the ROM, and a predecode register holds partially decoded signals. Predecode is the only decoding that sees an instruction during the two cycles before its T0/T1 states arrive (verified — Visual6502wiki, 6502 State Machine: "None of the decoding (except the predecode) sees the instruction during the two cycles that come before this"). The IR is the PLA's address; the predecode register is what lets the first two fetch cycles overlap the previous instruction's execution.
- **Shift register (the T-state sequencer)** — Timing is a shift register stepping a one-hot state through T0–T5; the PLA takes those states as inputs (verified — Visual6502wiki, 6502 State Machine: "The PLA has T0,T1,T2,T3,T4,T5 as inputs."). A separate T1F signal — "a 'real' T1" — is output on the SYNC pin. The sequencer also carries the out-of-band states: BRK gets VEC0/VEC1 instead of a T6, and read-modify-write instructions get two extra cycles, SD1 and SD2. The shift register is the chip's clock: nothing happens except in a named T-state.
- **Random logic (the additional-decoding layer)** — Between the PLA and the datapath sits a layer of combinational logic that is neither ROM nor register: the middle third of the die is, in Ken Shirriff's words, "a bunch of logic, mostly to do additional decoding of instructions" (verified — K. Shirriff, "The 6502 CPU's overflow flag explained at the silicon level", righto.com, January 2013). It combines PLA outputs with timing states and internal state (interrupts, RDY, the decimal-carry signals) to produce the roughly 44 datapath control signals (dpc0–dpc43) that actually drive the chip. The PLA decides *which* instruction is running; the random logic decides *when* each of its control actions fires.
- **Latch (control-signal latching)** — Every datapath control signal is latched during phi2, and is set mostly from opcode and timing data but also from some internal state (verified — Visual6502wiki, 6502 datapath: "All datapath control signals are latched during phi2; they are set mostly from opcode and timing data, but also some internal state."). The latch is what makes the two-phase clock work as a pipeline: phi2 freezes the control for the cycle, phi1 executes it.
- **Adder (the PC incrementer)** — The program counter's incrementer is a 16-bit adder chain with no internal latches of its own: when it increments, the new value is put directly on ADL and ADH (verified — Visual6502wiki, 6502 datapath: "When incrementing, the new value is put on ADL,ADH because there are no internal latches in the PC incrementer."). The I/PC signal gates whether it increments at all — it peeks at the predecoded next instruction and skips the increment for one-byte opcodes — and the PCLC/PCHC carry signals carry the increment across the low/high byte boundary. The incrementer is a pure combinational adder, not a counter: it recomputes PC+1 from the PC registers each cycle it is asked.
- **Register (address and data latches)** — The address leaves the chip through two 8-bit latches, ABL and ABH, loaded from ADL and ADH during phi1 and driven onto the address pins in that same phi1; the data latch DL is latched during phi2 and put on ADL, ADH or DB on the next phi1; the data output register DOR is latched from DB during phi1 and driven onto the data pins in phi2 on a write (verified — Visual6502wiki, 6502 datapath). These three registers are the chip's only contact with the outside world: everything else on the die talks to them through the internal buses.
- **Pass transistors / mux (bus interconnect and precharge)** — The four internal buses (SB, DB, ADL, ADH) are joined by pass transistors that connect two buses together when a control signal fires — SB/DB and SB/ADH are bidirectional pass-connects, and DL/ADL, DL/ADH and DL/DB pass-connect the data latch onto the buses (verified — Visual6502wiki, 6502 datapath, dpc table). And every internal bus is driven high during phi2 as a sort of precharge (verified — same source: "All internal busses (SB, DB, ADL, ADH) are driven high during phi2, as a sort of precharge."). Precharge is why the NMOS 6502 can use small pass transistors instead of full drivers: the bus is already high, and a transistor only has to pull it low.

**Parameter honesty.** Parameter honesty: the sources publish the ROM geometry (130 × 21), the 131 PLA outputs, the ~44 datapath control signals, the T0–T5 states plus T1F, VEC0/VEC1 and SD1/SD2, and the 8-bit instruction register; they publish no PLA transistor count per row, no shift-register latch count, no gate count for the random-logic layer, and no predecode-register width, and this decomposition invents none of them.

#### Down to the primitives: The ALU — two nibble adders with a decimal adjust

- **Mux (ALU input selection)** — The ALU's two operand inputs are selected by muxes, not wired: one side takes two options (SB/ADD or 0/ADD) and the other takes three (nDB/ADD, DB/ADD or ADL/ADD) (verified — Visual6502wiki, 6502 datapath: "SB/ADD, 0/ADD, nDB/ADD, DB/ADD, ADL/ADD Two options for one side, three for the other."). The nDB input — the inverted data bus — is what makes subtraction possible without a separate subtractor: SBC adds the complement. The muxes are the ALU's only operand steering, and they are why the same adder serves arithmetic, logic and address computation.
- **Adder (two 4-bit nibble adders)** — The sum is computed as a pair of 4-bit additions, not one 8-bit addition: the two operands and the carry-in are added as a pair of nibbles, and the carry-out from bit 3 is the hand-off to the high nibble (verified — Visual6502wiki/6502DecimalMode: "The two operands, and the carry in, are added as a pair of nibbles. The carry-out from bit3 is adjusted in decimal mode, but only for ADC."). Each nibble's carry ripples bit to bit inside the nibble, so the sum settles in two short serial chains rather than one eight-bit chain — and the boundary at bit 3 is exactly where the decimal adjust sits.
- **Logic network (AND / OR / EOR / shift)** — The same operand network that adds also does logic: the operation-select signals ANDS, EORS, ORS, 1/ADDC, SRS, SUMS, DAA and DSA choose the ALU operation (verified — Visual6502wiki, 6502 datapath: "ANDS, EORS, ORS, 1/ADDC, SRS, SUMS, DAA, DSA Select the ALU operation."). SRS is a logical right shift — there is no separate barrel shifter; shifts are one-bit operations through the ALU, and the rotate-right top bit is handled by not driving it (the ALU always computes a zero there). The NAND and NOR of the two operands' top bits are already available in the ALU for other purposes (verified — K. Shirriff, righto.com, January 2013) — the logic network is shared infrastructure, not a separate logic unit.
- **Comparator (the decimal adjust)** — The decimal adjust is not an adder at all: it is a handful of pulldown transistors gated by the DAA (ADC) and DSA (SBC) control signals. The 2A03 proves the count: removing five transistors from the polysilicon mask disables decimal adjustment entirely — three for ADC (t2750, t2202, t2556) and two for SBC (t1329, t3212) (verified — Visual6502wiki/6502DecimalMode: "With visual6502 and images from Quietust's investigation of the 2A03 we can see that a small number of changes, only to the polysilicon mask, disable the decimal adjustment by removing 5 transistors."). Five transistors is the whole decimal mode: the feature that sold the chip over the 6800 is five pulldowns.
- **Register (the ADD output register)** — The ALU's result is written into the 8-bit ADD register during phi2 and can be used the next cycle (verified — Visual6502wiki, 6502 datapath: "The ALU output register (ADD) is written during phi2."). Its outputs split: ADD/SB7 is separate from ADD/SB(0-6) because the top bit is handled separately for rotate-right, and ADD/ADL feeds address calculations. Before the result reaches the accumulator, lines 1–3 and 5–7 pass through the decimal adjust, to finish the proper BCD add/subtract result if necessary (verified — same source). ADD is the ALU's only output register, and the decimal adjust sits between it and the accumulator, not inside the adder.
- **Comparator (flag generation)** — The carry-out (ACR) and overflow (AVR) leave the ALU back to the control logic, latched during phi2 and used in phi1 (verified — Visual6502wiki, 6502 datapath: "The overflow and carry out signals AVR and ACR are output from the ALU back to the control logic, latched during phi2, used in phi1."). Overflow is computed from the top bits of the two operands and the carry out of bit 6 into bit 7 — V = not(((A7 NOR B7) and C6) NOR ((A7 NAND B7) NOR C6)) (verified — K. Shirriff, righto.com, January 2013) — reusing the NAND/NOR the ALU already computes. The decimal carries are picked up at phi2 as well, so the flag logic and the datapath agree on the same cycle boundary.
- **Register (the status register P)** — The processor status lives in its own 8-bit register: P/DB writes the flag values onto the data bus, and the data bus can be read to set the flags — read during phi2, latched into the flag register on the next phi1 (verified — Visual6502wiki, 6502 datapath). The D flag that selects decimal mode lives here, which is why the NES CPU could keep the flag settable while the adjust transistors were gone.
- **FSM / controller** — The operation-select signals are raised by the PLA and the timing sequencer at the right phase, and the decimal carries are picked up at phi2 — the ALU itself has no sequencer; it is pure combinational logic plus the ADD register, driven by the control unit's phase discipline. The ALU's "state" is exactly one register (ADD) plus the flags.

**Parameter honesty.** Parameter honesty: the sources publish the input mux widths (2 options on one side, 3 on the other), the two 4-bit nibble adders, the eight operation-select signals, the five decimal-adjust transistors by name, and the AVR/ACR flag signals; they publish no gate-level schematic of the adder, no transistor count for the ALU as a whole, and no width for the predecode register, and this decomposition invents none of them. The gate-level form of the nibble adders is Level-4 work, labelled standard construction there.

**New fact records added in this run** (appended to `files/fact-records.json`; statuses per the fact standard):

- [verified] **MOS Technology 6502** — Decode ROM row format: each row matches IR bits plus the cooked group input G plus the timing state T; G1 = IR0, G2 = IR1, G3 = !IR0 & !IR1. Source: Visual6502wiki/6507 Decode ROM (NESdev wiki).
- [verified] **MOS Technology 6502** — Opcode low bits 11 turn on both G1 and G2, so G=1 and G=2 rows fire together — the mechanism behind undocumented opcodes (LAX = LDA + LDX). Source: Visual6502wiki/6507 Decode ROM (NESdev wiki).
- [verified] **MOS Technology 6502** — The PLA takes T0–T5 as timing inputs; a separate T1F signal drives the SYNC pin; BRK uses VEC0/VEC1, RMW instructions add SD1/SD2. Source: Visual6502wiki, 6502 State Machine.
- [verified] **MOS Technology 6502** — Predecode is the only decoding that sees an instruction during the two cycles before its T0/T1 states. Source: Visual6502wiki, 6502 State Machine.
- [verified] **MOS Technology 6502** — The middle third of the die is random logic doing additional decoding between the PLA and the datapath. Source: K. Shirriff, righto.com, January 2013.
- [verified] **MOS Technology 6502** — All datapath control signals are latched during phi2. Source: Visual6502wiki, 6502 datapath.
- [verified] **MOS Technology 6502** — The PC incrementer has no internal latches; the incremented value is put directly on ADL/ADH. Source: Visual6502wiki, 6502 datapath.
- [verified] **MOS Technology 6502** — All internal buses (SB, DB, ADL, ADH) are driven high during phi2 as a precharge. Source: Visual6502wiki, 6502 datapath.
- [verified] **MOS Technology 6502** — ALU input selection: two options on one side (SB/ADD, 0/ADD), three on the other (nDB/ADD, DB/ADD, ADL/ADD). Source: Visual6502wiki, 6502 datapath.
- [verified] **MOS Technology 6502** — ALU operation selects: ANDS, EORS, ORS, 1/ADDC, SRS, SUMS, DAA, DSA. Source: Visual6502wiki, 6502 datapath.
- [verified] **MOS Technology 6502** — The ALU output register ADD is written during phi2; its lines pass through the decimal adjust before reaching the accumulator. Source: Visual6502wiki, 6502 datapath.
- [verified] **MOS Technology 6502** — Decimal adjustment is five pulldown transistors: three for ADC (t2750, t2202, t2556), two for SBC (t1329, t3212); removing them disables decimal mode. Source: Visual6502wiki/6502DecimalMode (NESdev wiki).
- [verified] **MOS Technology 6502** — Overflow reuses the ALU's NAND/NOR of the operands' top bits; V = not(((A7 NOR B7) and C6) NOR ((A7 NAND B7) NOR C6)). Source: K. Shirriff, righto.com, January 2013.
- [unverifiable] **MOS Technology 6502** — Primitive-level parameters not published: PLA transistor count per row, T-state shift-register latch count, random-logic gate count, ALU gate-level adder schematic and transistor count, predecode-register width. Sources: visual6502 project pages, righto.com.

### AMD Zen 4

#### Down to the primitives: The op cache and branch predictor — the front end that feeds the machine

- **CAM (the L1 and L2 BTB tag match)** — The branch target buffers are associative lookups: the fetch address is matched against stored tags, and a hit returns the target in a single cycle from the L1 BTB. Zen 4's L1 BTB keeps Zen 3's 1-cycle latency while tracking up to 3,072 branch targets, depending on branch density (verified — Chips and Cheese: "Zen 4's L1 BTB keeps the same 1 cycle latency, but improves capacity. Depending on branch density, Zen 4's L1 BTB can track up to 3072 branch targets, though in practice it'll probably be able to track 1024 to 2048 targets."). The 1.5K entries each hold up to two targets — the second in the same aligned 64-byte line as the first — which is how 1.5K entries cover up to 3,072 targets (per the AMD Software Optimization Guide: "Each BTB entry can hold up to two branches"). The L2 BTB is bigger and faster than Zen 3's: measured at 8,192 entries, up from 6,656, with the lookup penalty cut from three cycles to one (verified — Chips and Cheese: "Zen 4's L2 BTB sees a slightly size increase as well, with capacity going from 6656 entries on Zen 3 to 8192 on Zen 4. More significantly, Zen 4's L2 BTB is faster: Zen 3 suffered a three cycle penalty when it needed a branch target from its L2 BTB. Zen 4 brings that penalty down to just one cycle."). Note the capacity is disputed: AMD's own figures put the L2 BTB at 7K entries, so the wiki keeps 7K and records the 8,192-entry measurement as disputed.
- **CAM (the indirect target array)** — Indirect branches, which can go to many targets, get their own array separate from the main BTBs (verified — Chips and Cheese: "AMD's CPUs have a separate indirect target array for handling branches that go to multiple targets."). Measured tracking holds up to 3,072 targets without significant penalty (verified — same source: "In our testing, Zen 4 seems to be able to track up to 3072 branch targets without significant penalty."). It works as an overriding predictor: if the branch goes where it went last time, the fast main BTBs supply the target and the array stays out of the way. The array's entry count and associativity are not published.
- **SRAM bank (the op cache and the L1 instruction cache)** — The op cache is an associative SRAM array: 64 sets by 12 ways, each set-way entry holding up to 9 macro-ops, for a maximum of 6.75K ops, physically tagged (verified — AMD Software Optimization Guide 57647: "The Op Cache is organized as an associative cache with 64 sets and 12 ways. At each set-way intersection is an entry containing up to 9 macro ops. The maximum capacity of the Op Cache is 6.75 K ops."). It is the front end's working memory: a hit delivers up to 9 macro-ops per cycle with no decode and no L1I access. The miss path's store is the 32 KB, 8-way L1 instruction cache, which refills the op cache through the 4-wide decoder. The op cache's banking and port counts, and the L1I's banking, are not published.
- **FIFO / queue (the micro-op queue)** — Between the op cache and rename sits the micro-op queue, which absorbs the op cache's 9-per-cycle bursts and meters ops out at 6 µops per clock (verified — Chips and Cheese: "AMD says the micro-op cache delivers 9 ops per cycle, but throughput is limited by the 6-wide renamer downstream."). The queue's depth is not published, so this decomposition names the primitive and its rate only.
- **FIFO / queue (the reorder buffer)** — The ROB is a 320-entry circular buffer — a FIFO in structure, with in-order retirement at its head. Its capacity is confirmed by independent measurement (verified — Chips and Cheese: "We used a custom test with up to 128 integer adds, 128 floating point adds, 40 stores, and 55 branches (in that order) to confirm AMD's published 320 entry ROB capacity."). Each entry can hold up to 4 NOPs — fused NOP pairs are fused again at rename — which is why a NOP-only test measured a reordering capacity of 1,265 NOPs (verified — same source: "Zen 4's reorder buffer is also special, because each entry can hold up to 4 NOPs."). The ROB's associative lookup structure is not published.
- **Register file (the physical register files and the register alias table)** — Rename draws on three files: 224 integer registers, 192 FP/vector registers widened to 512 bits, and a 68-entry mask file. The register alias table is itself a small multi-ported register file mapping architectural registers to physical ones, and it carries the renamer's zero trick: a MOV or zeroing idiom is eliminated by pointing several architectural registers at one physical register, and a known-zero value is remembered by a bit in the RAT instead of allocating a real integer register (verified — Chips and Cheese: "Eliminated MOVs can be handled by pointing multiple architectural registers to the same physical one, meaning that no extra physical registers have to be allocated for such MOVs. If the renamer knows a register's value has to be zero, it can remember that by setting a bit in the register alias table instead of allocating a real integer register."). Measured, 202 of the 224 integer registers are available to hold speculative results (verified — same source: "We measured 202 integer registers (out of 224) available to hold speculative results, which covers 63% of ROB capacity."). The files' read/write port counts are not published.
- **Counter / stack (the return stack)** — Returns are predicted by a 32-entry return-address stack: a call pushes the return address, a return pops it (verified — Chips and Cheese: "Like prior Zen generations, Zen 4 has a return stack with 32 entries. However, AMD seems to have made improvements so that a single thread can use all 32 entries."). It is a tiny LIFO — a stack-pointer counter plus a 32-deep register array — and it is why call/return pairs predict almost perfectly until the stack overflows.
- **SRAM bank + counter (the direction predictor's pattern-history tables)** — The conditional predictor's two overriding levels are pattern-history tables: SRAM banks indexed by branch history, each entry a saturating counter — the standard construction of a direction predictor (verified — Chips and Cheese: "Like other CPUs in the Zen series, Zen 4's direction predictor has two overriding levels."). Zen 4's L2 level is much larger than Zen 3's and recognises extremely long patterns (verified — same source: "Zen 4 and Zen 3 seem to have similarly capable L1 predictors, but AMD has dramatically beefed up Zen 4's L2 predictor. ... It can recognize extremely long patterns and has enough storage to perform very well, even with a lot of branches in play."). AMD publishes no table sizes, history lengths or algorithm for either level, so this decomposition names the primitive and its role only.
- **Priority encoder / switch arbiter (the schedulers' wakeup and select)** — The back end's schedulers are the select logic: each cycle they wake every queued op whose operands are ready and pick which ones issue, with the FP side split into two triple-port schedulers sharing a large non-scheduling queue (verified — Chips and Cheese: "On the floating point and vector execution side, Zen 3 and Zen 4 retain the large non-scheduling queue found in previous Zen generations, but move from a unified quad-port scheduler to two triple port schedulers."). The integer side shares its scheduling queues between ALU and AGU ports, and the whole layout is unchanged from Zen 3 (verified — same source: "Scheduler and execution pipe layout were not changed from Zen 3."). A scheduling queue has to check every entry every cycle and send an instruction for execution on the same cycle it gets marked ready (verified — same source: "A scheduling queue potentially has to check every entry every cycle, to see if it's ready to execute. To make things even harder, it has to be able to send an instruction for execution on the same cycle it gets marked ready."). Scheduler queue depths and the exact select policy are not published.
- **Mux (the supply-path select)** — The front end's last primitive is a mux, not a store: each cycle the fetch stream comes either from the op cache (hit) or from the L1I-plus-decoder refill path (miss), and the BTB's two-target entries let the predictor cover two taken branches per cycle (per the AMD Software Optimization Guide: "Predicting with BTB pairs allows two fetches to be predicted in one prediction cycle."). The mux radix and the op cache's banking are not published.

**Parameter honesty.** Parameter honesty: the sources publish the op-cache geometry (64 sets × 12 ways, up to 9 macro-ops per entry, 6.75K ops maximum), the L1 BTB's 1-cycle latency and up-to-3,072 targets, the L2 BTB's measured 8,192 entries (disputed against AMD's 7K), the indirect target array's 3,072-target tracking, the 32-entry return stack, the micro-op queue's 6 µops-per-clock rate, the 320-entry ROB (4 NOPs per entry), the 224/192/68 register files, and the two triple-port FP schedulers; they publish no BTB associativity or entry layout, no direction-predictor table sizes, history lengths or algorithm, no op-cache or L1I banking or port counts, no micro-op queue depth, no scheduler queue depths or select policy, and no ROB lookup structure, and this decomposition invents none of them.

#### Down to the primitives: The double-pumped AVX-512 datapath — 512-bit instructions on 256-bit pipes

- **Multiplier array (the FMA multiply stage)** — Each 256-bit FMA pipe's first stage is a multiplier array: four double-precision lanes per pipe, each multiplying 53-bit mantissas (the IEEE 754 double format) into partial products that a reduction tree sums. The array is what makes the pipe a multiply-accumulate unit rather than a plain adder. AMD publishes no multiplier geometry — no Booth encoding, no compressor-tree structure, no partial-product count — so this decomposition names the primitive and its role only; the gate-level form is Level-4 standard construction.
- **Adder tree (the FMA accumulate stage and the FP add pipes)** — The FMA's second stage adds the product into the accumulator, and the two ADD/MUL pipes are adder trees in their own right: FADD/FMUL latency is 3 cycles and FMA latency 4 cycles in the 256-bit forms. The adder tree is also what the double pump reuses: the low half's accumulate and the high half's accumulate run through the same tree on consecutive cycles. The tree's depth and carry structure are not published.
- **Register file (the 192 × 512-bit FP file and the 68-entry mask file)** — Operands live in a 192-entry FP/vector physical register file widened to 512 bits, with a separate 68-entry mask register file holding the per-lane predicates. The widening is the most expensive part of the AVX-512 implementation (per Chips and Cheese: "The most expensive change probably had to do with extending the vector register file to make each register 512-bits wide."). Measured, the mask file supports around 52 renames — fewer than its 68-entry capacity (verified — Chips and Cheese: "With around 52 renames for mask registers, this register file is much smaller than what we see on Skylake-X or Ice Lake."). The files' read/write port counts are not published.
- **Mux (the low/high half-select of the double pump)** — A 512-bit op is split into its low and high 256-bit halves and occupies a pipe for two consecutive cycles; a half-select mux steers the low half in cycle 1 and the high half in cycle 2. The split happens as late as possible — the instruction stays one micro-op through most of the pipeline and is only broken into two 256-bit halves after it enters the execution pipe (verified — Chips and Cheese: "Zen 4 partially breaks this tradition, by keeping instructions that work on 512-bit vectors as one micro-op throughout most of the pipeline."). The mux's exact implementation is not published.
- **Barrel shifter / crossbar (the full-width 512-bit shuffle unit)** — The one exception to double-pumping: a dedicated full-width shuffle unit executes 512-bit shuffles in a single cycle. A shuffle is a cross-lane permutation — a crossbar (or barrel-shifter network) that routes any of the 16 32-bit lanes to any output lane in one pass. Its radix and lane-routing structure are not published.
- **FIFO / queue (the 88-entry load queue and 64-entry store queue)** — Feeding the pipes is the load/store unit, behind an 88-entry load queue and a 64-entry store queue. The queues are the ordering structures: loads wait here until their addresses are known and their data returns; stores wait here until retirement, because only known-good data can be committed to cache (per Chips and Cheese: "stores have to stay there until retirement because only known-good data can be committed to cache."). 512-bit stores are the exception to the one-micro-op rule: they are decoded into two micro-ops and consume two store-queue entries (verified — Chips and Cheese: "512-bit stores are an exception to the rule. They're still decoded into two micro-ops, which means they consume two valuable store queue entries."). The queues' associative lookup structure (how a load finds a matching store) is not published.
- **SRAM bank (the L1 data cache and the L2)** — The load/store unit draws on the 32 KB, 8-way L1 data cache, behind which sits the 1 MB, 8-way L2 at 14 cycles. L1D bandwidth is two 256-bit loads and one 256-bit store per cycle, unchanged from Zen 2 (verified — Chips and Cheese: "Zen 4's L1D can handle two 256-bit loads and one 256-bit store per cycle, which means that vector load/store bandwidth remains unchanged from Zen 2."). A 512-bit vector load therefore takes two cycles on the 256-bit load ports. The caches' banking and port counts are not published.
- **FSM / controller (the FP schedulers and pipe reservation)** — The FP scheduler issues AVX-512 macro-ops when their operands are ready, and reserves the chosen pipe for two consecutive cycles — a small controller FSM whose state is, per pipe, free or busy-for-the-second-half. The FP side uses two triple-port schedulers sharing a large non-scheduling queue. The scheduler's exact state encoding and select policy are not published.
- **Comparator (the per-lane mask evaluation)** — The mask register file's per-lane predicate bits are evaluated against the operation: masked lanes neither compute nor write. The comparator is one AND per lane between the mask bit and the operation enable — the primitive that turns a 512-bit op into a partially-masked one. The mask file's port count and the exact gating structure are not published.

**Parameter honesty.** Parameter honesty: the sources publish the four 256-bit FP pipes (2 FMA/MAC + 2 ADD/MUL) plus the dedicated 512-bit shuffle unit, the 192 × 512-bit FP register file and 68-entry mask file (around 52 mask renames measured), the 88-entry load queue and 64-entry store queue, the 32 KB 8-way L1D and 1 MB 8-way L2 at 14 cycles, the 2× 256-bit loads + 1× 256-bit store per cycle, and the FADD/FMUL 3-cycle and FMA 4-cycle latencies; they publish no multiplier-array geometry, no adder-tree depth or carry structure, no FP register-file port counts, no shuffle-unit crossbar radix, no load/store queue lookup structure, and no scheduler state encoding, and this decomposition invents none of them.

**New fact records added in this run** (appended to `files/fact-records.json`; statuses per the fact standard):

- [verified] **AMD Zen 4** — Op cache organization: 64 sets × 12 ways, up to 9 macro-ops per entry, maximum capacity 6.75K ops, physically tagged. Source: AMD Software Optimization Guide for the AMD Zen4 Microarchitecture, publication 57647, rev 1.01, April 2023.
- [verified] **AMD Zen 4** — BTB is a two-level structure; each entry holds up to two branches, and BTB pairs allow two fetches to be predicted in one prediction cycle. Source: AMD Software Optimization Guide 57647, §2.8.1.2.
- [verified] **AMD Zen 4** — L1 BTB keeps 1-cycle latency, tracks up to 3,072 branch targets depending on branch density (in practice 1,024–2,048). Source: Chips and Cheese, "AMD's Zen 4 Part 1: Frontend and Execution Engine".
- [verified] **AMD Zen 4** — L2 BTB measured at 8,192 entries (up from 6,656 on Zen 3); lookup penalty cut from 3 cycles to 1 cycle. Source: Chips and Cheese, "AMD's Zen 4 Part 1".
- [disputed] **AMD Zen 4** — L2 BTB capacity: AMD/Wikipedia put it at 7K entries; Chips and Cheese measured 8,192 entries. Wiki keeps 7K; no change made.
- [verified] **AMD Zen 4** — Indirect branches get a separate indirect target array; measured tracking holds up to 3,072 targets without significant penalty. Source: Chips and Cheese, "AMD's Zen 4 Part 1".
- [verified] **AMD Zen 4** — Return stack has 32 entries; on Zen 4 a single thread can use all 32. Source: Chips and Cheese, "AMD's Zen 4 Part 1".
- [verified] **AMD Zen 4** — Op cache delivers 9 ops per cycle, but throughput is limited by the 6-wide renamer downstream. Source: Chips and Cheese, "AMD's Zen 4 Part 1".
- [verified] **AMD Zen 4** — Renamer eliminates MOVs and zeroing idioms by pointing multiple architectural registers at one physical register and by setting a zero bit in the register alias table instead of allocating a real integer register; 202 of 224 integer registers measured available for speculative results. Source: Chips and Cheese, "AMD's Zen 4 Part 1".
- [verified] **AMD Zen 4** — ROB capacity of 320 entries confirmed by a custom test (128 integer adds, 128 FP adds, 40 stores, 55 branches); each ROB entry can hold up to 4 NOPs. Source: Chips and Cheese, "AMD's Zen 4 Part 1".
- [verified] **AMD Zen 4** — Scheduler and execution pipe layout unchanged from Zen 3; FP/vector side uses two triple-port schedulers sharing a large non-scheduling queue; a scheduling queue must check every entry every cycle and issue on the same cycle an op is marked ready. Source: Chips and Cheese, "AMD's Zen 4 Part 1".
- [verified] **AMD Zen 4** — Direction predictor has two overriding levels; Zen 4's L2 predictor is much larger than Zen 3's and recognises extremely long patterns. Source: Chips and Cheese, "AMD's Zen 4 Part 1".
- [verified] **AMD Zen 4** — 512-bit instructions are kept as one micro-op through most of the pipeline, split into two 256-bit halves as late as possible. Source: Chips and Cheese, "AMD's Zen 4 Part 1".
- [verified] **AMD Zen 4** — 512-bit stores are decoded into two micro-ops and consume two store-queue entries. Source: Chips and Cheese, "AMD's Zen 4 Part 1".
- [verified] **AMD Zen 4** — Two 256-bit FMA units and four 256-bit ALUs, unchanged since Zen 2; L1D handles two 256-bit loads and one 256-bit store per cycle, unchanged from Zen 2. Source: Chips and Cheese, "AMD's Zen 4 Part 1".
- [verified] **AMD Zen 4** — Mask register file supports around 52 renames measured, fewer than its 68-entry capacity. Source: Chips and Cheese, "AMD's Zen 4 Part 1".
- [unverifiable] **AMD Zen 4** — Primitive-level parameters not published: BTB associativity and entry layout, direction-predictor table sizes/history lengths/algorithm, op-cache and L1I banking/port counts, micro-op queue depth, scheduler queue depths and select policy, ROB lookup structure, FMA multiplier-array geometry, adder-tree depth/carry structure, FP register-file port counts, shuffle-unit crossbar radix, load/store queue lookup structure, scheduler state encoding. Sources: AMD Hot Chips 35, AMD SOG 57647, Chips and Cheese, mersenneforum teardown.

## Level-4 decomposition — Down to the gates (2026-11-22)

The first Level-4 pass (Sai, Nov 22, 2026: "Go as deep as you can, till you cant decompose anymore") takes every primitive in the seven Level-3 entries one level further: into its gate-level structure, and each gate into transistors. The transistor is the floor — below it is device physics, not architecture. The honesty rule is strictest at this depth: gate-level structure a primary source actually publishes (rare — essentially only the Intel 4004's 1973 manual) is claimed as a chip fact under the Fact standard; everything else is the generic textbook construction of the primitive, labelled **standard construction** in the text, and never presented as the chip's actual implementation. Generic textbook material gets no fact records; the pass added 7 records only to scope the boundary (298 -> 305).

### The bottom layer — the primer’s floor

The Basic Blocks Primer gains a bottom layer: the five leaf cells every Level-4 decomposition bottoms out in. Everything in this layer is generic textbook construction — it describes how these cells are built in any CMOS library and is NOT a fact about any specific chip.

- **Universal gates — NAND & NOR** — Every logic function reduces to NAND (or NOR) alone: NOT is a NAND with its inputs tied together, AND is a NAND followed by an inverter, OR falls out of NANDs by De Morgan's laws, and any sum-of-products maps to a two-level NAND–NAND network. NAND and NOR are functionally complete — they are the atoms every other gate, and so every primitive in this primer, is assembled from. Textbook-standard construction — generic CMOS, not a fact about any specific chip. Key parameters: Inverter = 2 transistors (1 PMOS pull-up + 1 NMOS pull-down). NAND2 = 4 (2 PMOS in parallel + 2 NMOS in series). NOR2 = 4 (the dual: 2 PMOS in series + 2 NMOS in parallel). AND2 / OR2 = 6 (NAND / NOR + inverter). XOR2 ≈ 10–12 depending on style. Trade-offs: Static CMOS is complementary: a PMOS pull-up network and an NMOS pull-down network, one of them always off, so there is no static current path. Series stacks are the cost — beyond roughly four series transistors, delay and device sizing blow up, which is why wide gates are built as trees of small ones. Appears in: The leaf cell of every Level-4 decomposition in this zoo — every FIFO comparator, mux select and adder carry bottoms out here.

- **Latch & D flip-flop** — Storage from feedback: an SR latch is two cross-coupled NAND (or NOR) gates, each gate's output feeding the other's input, so the pair holds one bit with no clock at all. Gate the inputs and it becomes a D latch, transparent while enabled. Put two D latches in series driven by opposite clock phases — master, then slave — and the pair copies its input only at the clock edge: a master–slave D flip-flop. Textbook-standard construction — generic, not a fact about any specific chip. Key parameters: SR latch = 2 gates ≈ 8 transistors. Master–slave D flip-flop ≈ 20–28 transistors in a typical standard-cell library (the range covers transmission-gate and NAND-based styles, with or without set/reset). A scan flop adds a 2:1 mux in front (≈ +6–8 transistors). Trade-offs: The flip-flop is the most expensive one-bit store in the zoo — roughly ten times the transistors of an SRAM bit — but it is fully static, fast and testable, which is why pipelines and control state are flops while bulk data lives in SRAM. Clock power is the tax: the clock pin toggles every cycle whether the data changes or not. Appears in: Every pipeline register, counter, FSM state bit and credit counter in this primer is built from these; the scan chain (Scan Design entry) is these, plus one mux each.

- **6T SRAM cell** — Two cross-coupled inverters (4 transistors) form a bistable latch holding one bit; two NMOS access transistors — the fifth and sixth — connect the internal nodes to a differential bitline pair (BL / BLB) under control of the wordline. Read: precharge both bitlines, raise the wordline, and the cell pulls one side down slightly — a sense amplifier detects the small differential. Write: drive the bitlines hard enough to overpower the cell's weaker pull-up and flip it. Textbook-standard construction — generic, not a fact about any specific chip. Key parameters: 6 transistors per bit. Cell stability is a sizing-ratio game: the access transistors must be weaker than the pull-downs (read stability) but stronger than the pull-ups (writability) — the classic cell-ratio and pull-up-ratio constraints. Trade-offs: The densest reliable one-bit store in CMOS — far smaller per bit than a flip-flop — bought with differential sensing, precharge energy on every access, and a static noise margin that shrinks as supply voltage scales down. Appears in: The leaf cell of every SRAM bank, cache array, packet buffer and SRAM-style register file in this zoo — Tomahawk's shared buffer and Cerebras' 40 GB are this cell, replicated.

- **CAM cell** — A CAM cell is storage plus compare: an SRAM cell holds the bit, and XOR match logic compares it against the search bit presented on complementary searchlines; every cell on a row drives one shared, precharged matchline, and a single mismatching cell pulls the matchline down — a wired-NOR of per-bit XNOR results. Textbook-standard construction — generic, not a fact about any specific chip. Key parameters: NOR-type CAM cell ≈ 10 transistors (6T storage + ≈4T compare). TCAM ≈ 16 transistors — two storage cells per bit, so the stored pair can encode 0, 1 or don't-care (X). Trade-offs: Every lookup precharges and evaluates every matchline in the array — compare energy per bit is the highest of any memory in the zoo, which is why CAMs stay small and SRAM-plus-hash structures do the bulk lookup work. Appears in: The leaf cell of TLBs, cache tag arrays, out-of-order wakeup tag stores and switch forwarding/ACL TCAMs — Firestorm's wakeup CAMs and Tomahawk's lookup tables bottom out here.

- **Full adder** — A half adder is an XOR gate (the sum) plus an AND gate (the carry). A full adder is two half adders cascaded — the second adds the carry-in — with an OR gate merging the two carry outputs. Chain N full adders and the carry ripples bit to bit; group the carries with lookahead logic and the same cells become the carry-lookahead adders inside every fast ALU. Textbook-standard construction — generic, not a fact about any specific chip. Key parameters: ≈28 transistors for the classic mirror-adder full-adder cell (one complex CMOS gate generating the carry, plus sum logic). A one-bit slice of every adder tree, counter incrementer and multiplier compressor tree in this primer is this cell. Trade-offs: Ripple is smallest and slowest; carry-lookahead and carry-select buy speed with area and wiring. Inside multipliers the adders hide in compressor trees (3:2 and 4:2) so partial products collapse in logarithmic depth before one final carry-propagate adder finishes. Appears in: The leaf cell of adder trees, ALUs, address incrementers and counters everywhere — including, in 4-bit ripple form, the Intel 4004's one adder (see its Level-4 decomposition).

### NVIDIA Hopper H100

#### Down to the gates: Fourth-generation Tensor Core — and the SM partition around it

- **Multiplier array** — Each MAC lane bottoms out in a gate-level multiplier: AND gates form the partial products from the operand bits, a compressor tree of full adders (≈28-transistor mirror cells — primer, bottom layer) collapses them, and one final carry-propagate adder finishes. Floating-point lanes add an exponent adder (a small lookahead adder), a sign XOR and a normalizer built from a priority encoder and a barrel shifter. Standard construction — NVIDIA publishes the array's throughput and dimensions, never its gate netlist.
- **Adder tree** — A reduction tree whose nodes are the primer's full adder: carry-save compressors (full adders used as 3:2 counters) at the leaves, one carry-propagate adder at the root. FP32 accumulation wraps each add with alignment shifters — log-stage mux arrays — and a rounding incrementer. Standard construction.
- **Register file** — At gate level a GPU register file is a latch/SRAM-style array: wordline decoders (NAND/AND trees), storage cells (cross-coupled latch pairs — the primer's latch cell), per-port column mux trees and sense amplifiers. Which cell style NVIDIA uses, and the transistor-level port structure, are not published. Standard construction.
- **SRAM bank** — Rows of 6T SRAM cells (primer, bottom layer) behind a row decoder — a tree of NAND gates selecting one wordline — transmission-gate column muxes (2 transistors per leg) and latch-type differential sense amplifiers. NVIDIA publishes shared-memory capacity and configurability (Level-3 fact records), not macro internals. Standard construction.
- **Mux** — The operand-collector and bypass muxes are trees of 2:1 muxes; each 2:1 mux is two transmission gates (one NMOS + one PMOS per leg) or an equivalent NAND-gate cell, with selects decoded by AND trees from the scheduler's one-hot grants. Standard construction.
- **FIFO / queue** — Storage (a flop array or a small SRAM) plus write and read pointer counters — D flip-flops with full-adder incrementers — plus full/empty comparators: XNOR gates per pointer bit feeding an AND tree, with Gray-coded pointers and 2-flop synchronizers where a queue crosses clock domains. Standard construction; the TMA and issue queue depths are not published.
- **FSM / controller** — State bits are D flip-flops (≈20–28 transistors each, primer); next-state and output logic are NAND/NOR networks — whatever the synthesis tool maps the state table to in the standard-cell library. No gate structure for Hopper's schedulers is published. Standard construction.
- **Pipeline register** — One master–slave D flip-flop per bit (≈20–28 transistors, primer), usually with a 2:1 enable mux in front so the stage can stall — the mux + flop pair is the same cell shape as a scan flop, minus the scan input. Standard construction.

Honesty: NVIDIA publishes Hopper at block and throughput level only — no gate-level structure for any part of the SM appears in any NVIDIA source. Everything above is the standard textbook construction of the named primitive, pointing at the primer's bottom layer; none of it is a claim about NVIDIA's actual netlist (new Level-4 fact record: unverifiable).

#### Down to the gates: Transformer Engine — the rescaling wrapped around the Tensor Core

- **Synchronizer** — Two D flip-flops in series in the receiving clock domain — the gate-level content is deliberately plain: the first flop is allowed to go metastable and the chain gives it a full clock period to resolve before the value is used. Standard construction.
- **Comparator** — Amax and scale comparisons ride a subtractor: XOR gates form the difference bits, a chain of full adders propagates the borrow, and the final borrow plus a NOR-tree zero-detect yield greater-than / equal. Floating-point compares check exponents first — a small priority network in front of the mantissa compare. Standard construction.
- **Multiplier** — The scale-factor multiply is one gate-level FP multiplier: a sign XOR, an exponent adder, a mantissa multiplier (AND partial products + full-adder compressor tree + final adder), then normalize-and-round logic. Standard construction.
- **Mux** — Format and scale selection muxes: transmission-gate 2:1 trees, 2 transistors per input leg, selects decoded from the engine's control state. Standard construction.
- **Counter / register state** — Amax history and scale factors live in D-flip-flop registers updated through comparator-driven muxes; any sequencing counters are flop banks with full-adder incrementers and XNOR+AND terminal-count comparators. Standard construction.

The Transformer Engine stays thin at gate level too — a comparator, two multipliers and a mux around an unchanged array. The novelty is where the scaling sits, not new gate-level arithmetic; and all of it is standard construction, as above.

### Google TPU v4

#### Down to the gates: MXU — the 128×128 systolic array inside a TensorCore

- **Multiplier array** — One processing element = one small multiplier + one adder, tiled 128×128. The BF16 multiplier at gate level: a sign XOR, an 8-bit exponent adder, and a mantissa multiplier of AND-gate partial products collapsed by full-adder compressors into a final carry-propagate adder; the accumulate adder is FP32, with alignment and normalization mux networks around it. All leaf cells are the primer's. Standard construction — the ISCA 2023 paper publishes the array geometry, not the PE netlist.
- **Adder tree — as a chain** — Systolic accumulation is a chain, not a tree: each PE adds its product into the partial sum arriving from its neighbour, so the gate-level content is one full-adder-based FP adder per PE — XOR/AND/OR carry logic from the primer's full-adder cell, replicated across the array. Standard construction.
- **Pipeline register** — The registers between PEs are plain D-flip-flop banks (≈20–28 transistors per bit, primer) — and in a systolic array they are the timing architecture: activations and partial sums advance exactly one PE per clock because a flop stands between every pair of neighbours. Standard construction.
- **SRAM bank** — VMEM and the on-chip buffers are 6T SRAM cell arrays (primer, bottom layer) with NAND-tree row decoders, transmission-gate column muxes and latch-type sense amplifiers. Google publishes capacities, not macro internals. Standard construction.
- **FIFO / queue** — Storage plus write/read pointer counters (D flip-flops + full-adder incrementers) plus full/empty comparators (XNOR gates per pointer bit into an AND tree). Standard construction.
- **FSM / controller** — The TensorCore sequencers are state registers (D flip-flops) plus synthesized NAND/NOR next-state logic. Standard construction.
- **Mux** — Operand steering muxes are transmission-gate trees (2 transistors per leg) under decoded one-hot selects. Standard construction.

Honesty: the ISCA 2023 paper publishes array geometry and memory sizes and nothing at gate level. Every gate structure above is standard construction from the primer's bottom layer, not Google's netlist (new Level-4 fact record: unverifiable).

#### Down to the gates: SparseCore — the dataflow engine for embeddings

- **FIFO / queue** — The gather/scatter queues decompose exactly as the MXU's queues do: storage + pointer counters + XNOR/AND flag comparators. Standard construction.
- **Counter** — Address and beat counters: D-flip-flop banks with full-adder incrementers and terminal-count comparators (XNOR per bit into an AND tree). Standard construction.
- **Adder tree** — Embedding reduction and address generation adders: full-adder cells in ripple or lookahead arrangement, from the primer's leaf cell. Standard construction.
- **Mux / steering** — Data steering is mux trees plus arbiters — and an arbiter at gate level is a priority-encoder tree of AND/OR gates with a grant register. Standard construction.

SparseCore internals are unpublished even at block level (Level-3 fact record: unverifiable), so its Level-4 decomposition is standard construction throughout — the primitives are named from the architecture's documented role, their gates from the textbook.

### Apple Firestorm — M1

#### Down to the gates: The wide front end and the ~630-entry reorder buffer — at the primitive level

- **FIFO / queue** — The ROB as a circular queue at gate level: entry storage in flop/latch arrays, head and tail pointer counters (D flip-flops + incrementers), and wrap/full comparators (XNOR gates per bit + AND tree). Standard construction — Apple publishes nothing, and even the ~630-entry depth is a microbenchmark measurement, not a specification.
- **Register file** — Rename storage at gate level is latch/flop cells behind mux trees: every read port costs a full N:1 mux per bit — transmission-gate or NAND trees — which is why port count, not capacity, prices a register file. Apple's cell style and port count are unpublished. Standard construction.
- **CAM** — Wakeup is the primer's CAM cell built from flops: per entry, tag storage plus one XOR compare gate per tag bit, all bits wired-NOR onto a matchline whose discharge says ‘my producer fired’. Queue partitioning and entry counts are inferred from measurements, not published. Standard construction.
- **Priority encoder** — Oldest-first select is a priority network: AND/OR prefix trees computing the highest-priority ready request, plus a valid gate. Ripple priority chains are too slow at this width, so tree structures are the textbook answer. Standard construction.
- **Mux** — Bypass/forwarding muxes: transmission-gate 2:1 trees (2 transistors per leg) fanning every producer result to every consumer input — the wiring that makes a 630-entry window usable. Radix unpublished. Standard construction.
- **Comparator** — The branch predictor's tag compares are XNOR trees; its direction counters are saturating up/down counters — a few flip-flops, a tiny adder/subtractor, and saturation comparators clamping at the rails. Predictor structure is wholly unpublished. Standard construction.
- **Barrel shifter / adder tree** — Inside the ALUs and the NEON pipes: adders are full-adder cells grouped by carry-lookahead prefix networks (AND/OR trees); shifters are log2(W) stages of 2:1 muxes. Standard construction.
- **Counter** — Fetch sequencing runs on a program-counter register plus an adder — full-adder cells with lookahead — and muxes choosing between sequential, predicted and redirect targets. Standard construction.

Honesty: Apple has published no internal structure for Firestorm at any level — block, primitive or gate. Every gate structure above is standard construction; the only numbers anywhere in this entry are microbenchmark measurements (unverifiable, fact records). Nothing here is an Apple fact (new Level-4 fact record: unverifiable).

### Cerebras WSE-2

#### Down to the gates: The dataflow core — and its 5-port router

- **SRAM bank** — Each core's 48 KB is 6T SRAM cells (primer, bottom layer) behind NAND-tree row decoders, transmission-gate column muxes and latch-type sense amplifiers — replicated across ~850,000 cores to make the published 40 GB total. Cerebras publishes the capacity and its single-cycle-access role (Level-3 fact records); the macro internals are standard construction.
- **Register file** — A small latch-based file: cross-coupled latch cells with mux-tree read ports — cheap at this size, and per-core rather than shared. Standard construction.
- **Multiplier array — a row, not a grid** — The core's FP16 multiply-accumulate at gate level: AND-gate partial products, a full-adder compressor tree, a final carry-propagate adder, plus exponent add and align/normalize mux networks for FP32 accumulation — the primer's leaf cells, one lane wide. Standard construction.
- **FSM / controller** — Dataflow firing control is an FSM: D-flip-flop state bits, NAND/NOR next-state logic, and comparators testing whether a microthread's operands have arrived. Standard construction.
- **Comparator** — Tag and colour matching: XNOR equality trees over the colour bits, and threshold compares riding a subtractor's carry chain. Standard construction.
- **Mux — the router** — The 5-port router's datapath is muxes: per output port a 5:1 mux — transmission-gate legs, 2 transistors each — selecting among the input ports under arbiter grants, with colour filtering as AND gates on tag-compare results. Port and colour counts are published (Level-3 fact records); the gate form is standard construction.
- **FIFO / queue** — Per-virtual-channel queues: flop or SRAM storage, write/read pointer counters (flip-flops + incrementers), XNOR+AND full/empty comparators. Depths are not published. Standard construction.
- **Credit counter** — One counter per channel: a flip-flop register with a full-adder increment/decrement path and a zero-detect NOR tree that gates transmission when credits run out. Standard construction.

Honesty: Cerebras is unusually open at primitive level (ports, colours, per-core SRAM — Level-3 fact records) and silent at gate level like everyone else. Everything below the primitive level above is standard construction (new Level-4 fact record: unverifiable).

### Broadcom Tomahawk 5 (BCM78900)

#### Down to the gates: The shared packet buffer — and the traffic manager around it

- **SRAM bank** — The shared buffer is banks of 6T SRAM cells (primer, bottom layer): NAND-tree row decoders, cell arrays, transmission-gate column muxes, latch-type sense amplifiers. The buffer's bandwidth — write once, read once, at 51.2 Tbps aggregate — is what replaces a crossbar. Capacity, cell size and banking are not published. Standard construction.
- **Crossbar — deliberately absent** — The one primitive in this decomposition with no gate level to describe: a crossbar would be an N×N array of transmission-gate crosspoints plus per-output arbiters, and Tomahawk's shared-buffer architecture deletes exactly that array. The absence is the published architecture (Level-3 fact records); there is no hidden crossbar netlist to reconstruct.
- **FIFO / queue** — Per-port, per-class queues at gate level: packet descriptors in SRAM or flop storage, head/tail pointer counters, and admission-threshold comparators — magnitude compares on a subtractor carry chain — deciding whether a new packet may claim shared-pool space. Queue counts and depths are not published. Standard construction.
- **Switch arbiter** — The traffic manager's grant logic is priority-encoder trees (AND/OR prefix networks) with round-robin mask registers, pipelined through flip-flops because a combinational grant across 64 ports of demand cannot close timing at line rate. The arbitration algorithm is not published. Standard construction.
- **TCAM / hash tables** — Where TCAM is used, the leaf cell is the primer's ≈16-transistor TCAM cell — two storage cells plus XOR compare driving a precharged matchline. Hash-based tables are SRAM banks behind XOR-tree hash logic with comparator chains resolving bucket matches. Broadcom publishes neither table sizes nor the split between the two. Standard construction.
- **Comparator** — Cognitive Routing's steering decisions are magnitude comparators: link-utilization counters subtracted on a full-adder carry chain, the borrow bit steering the flowlet. Thresholds and counter widths are not published. Standard construction.
- **Counter** — Telemetry is counters all the way down: flip-flop registers with full-adder incrementers, plus snapshot registers — parallel-load flip-flops — so software can read a coherent value without stopping the count. Standard construction.
- **FSM / controller — as processors** — The six on-chip Arm cores stop this decomposition honestly: they are licensed CPU cores, and their gate level is Arm's netlist, not Broadcom's published material. The primer's cells are certainly inside them; no public source decomposes these instances, so Level-4 does not pretend to.

Honesty: Broadcom publishes radix, rates and the shared-buffer architecture. Every gate structure above is standard construction; nothing is Broadcom's netlist (new Level-4 fact record: unverifiable).

### Scan Design & ATPG Architecture

#### Down to the gates: The scan chain — and the EDT wrapped around it

- **Mux** — The defining gate-level act of scan design: one 2:1 mux in front of every flip-flop's D input, selecting functional data or scan data under scan-enable — a transmission-gate pair (2 transistors per leg) or a NAND-gate mux inside the library cell. Scan flop = mux + master–slave D flip-flop ≈ 26–36 transistors in typical libraries (textbook range). Unlike every other entry in this pass, this structure is not a reconstruction: mux + DFF is the published, standardised definition of a scan cell.
- **FIFO / shift register** — The chain adds no storage at gate level at all: each scan flop's output wires to the next flop's scan input, so the shift register is the mission-mode flops themselves, reconfigured. What the chain does add is toggling — every shift clock moves data through every flop in the chain, which is why test power can exceed functional power.
- **Adder tree — as an XOR tree** — EDT decompressors and MISR compactors are XOR trees: XOR gates (≈10–12 transistors each, primer) combining chain outputs, and in the MISR one flip-flop per bit with XOR feedback taps placed by the signature polynomial (LFSR theory). The gate structure is textbook-standard; the compression ratio is a tool setting, not an architecture constant.
- **Comparator** — Response checking is an equality comparator: XNOR gates per bit feeding an AND tree, comparing shifted-out data or the final MISR signature against the simulated good-machine value — on the tester, or on-chip in LBIST. Standard construction.
- **FSM / controller** — Test-mode control and the on-chip clock controller (OCC): flip-flop state bits plus synthesized NAND/NOR logic; the OCC's single at-speed capture pulse comes from a small flop-and-gate pulse generator clocked by the functional clock. Tool-generated per design; the structure is textbook-standard.
- **Counter** — Shift counters and pattern counters: flip-flop banks with full-adder incrementers and XNOR+AND terminal-count comparators. Widths follow chain length and pattern count — per design. Standard construction.
- **Synchronizer** — Test and functional clocks meet under glitch-free clock muxing — AND/OR clock gates whose enables change only while the clock is low, driven through 2-flop synchronizers — so slow shift and at-speed capture never tear. Standard construction.

Scan inverts this zoo's usual honesty problem: the gate-level structure (mux + flip-flop, XOR compaction, clock control) is the published standard construction of the technique itself, defined by IEEE practice and every DFT textbook. What stays per-design and unpublished is only sizing — chain counts, lengths, compression ratios (Level-3 fact record).

### Intel 4004

#### Down to the gates: The address register — program counter and 3-level stack as a dynamic RAM array

- **Dynamic RAM cell array (storage)** — PUBLISHED — the exception in this whole pass: the MCS-4 manual states the address register is a dynamic RAM cell array of 4 x 12 bits, one level holding the program counter and three forming the stack (verified — fact records). Storage is charge held on a node, refreshed by cycling — not a static latch. NOT published: the per-cell transistor schematic. In 1971 PMOS practice a dynamic cell is one to three transistors storing charge on a capacitance; that cell-level picture is standard construction for the era, labelled as such.
- **Mux** — PUBLISHED (verified — fact records): the stack address, provided by the effective address counter and by the refresh counter, is multiplexed to the address-register decoder. The manual names the mux and its two sources; its transistor form — PMOS pass gates or NAND-gate select logic — is not published and is standard construction.
- **Counter** — The refresh counter is load-bearing in a dynamic design: PUBLISHED (verified) that reset must be applied for at least 8 full instruction cycles (64 clock cycles) to allow the index register refresh counter to scan all locations. The counter's own structure — flip-flops plus an incrementer — is standard construction, and its width is not published (unverifiable — fact records).
- **Register (address buffer)** — PUBLISHED (verified): the fetched address is stored in an address buffer and is demultiplexed to the internal bus during A1, A2, and A3 in three 4-bit slices. The buffer's cell structure is not published — in a dynamic PMOS chip it is most plausibly dynamic latches, but that is inference, labelled standard construction, not a manual claim.
- **Demux** — Steering one value onto one of several destinations under the cycle-phase signals: at gate level, pass gates or AND-gate trees gated by decoded phase lines. The three-slice demultiplexing behaviour is published (verified, above); the gate form is standard construction.
- **Adder (carry-look-ahead incrementer)** — PUBLISHED (verified): the address is incremented by a 4-bit carry look-ahead circuit (address incrementer) — genuine lookahead, in 1971, where the main adder (below) is plain ripple. Gate level, lookahead means generate/propagate terms combined in an AND/OR prefix network ahead of the sum XORs; the manual publishes that the circuit IS lookahead, not its netlist, so the prefix-network picture is standard construction fitted to a published fact.
- **FSM / controller** — Control is hardwired phase logic: the 8-clock instruction cycle (A1–A3, M1–M2, X1–X3) is generated by flip-flop phase stages with decode gates driving the datapath — no microcode, no ROM sequencer. The phase structure is published in the manual's cycle description; the gate form is standard construction.
- **Amplifier & mux (output driver)** — PUBLISHED (verified): each data bus output buffer has three possible states — ‘1’, ‘0’ and floating — tri-state drivers, so exactly one buffer drives a shared bus line at a time. In PMOS these are the largest devices on the die, sized to charge the external bus capacitance; device sizing is not published.

Published vs constructed, stated plainly: the manual publishes this block's organisation — dynamic RAM array, stack-address mux, refresh counter, three-slice demultiplexing, lookahead incrementer, tri-state drivers — and every one of those claims is verified in the fact records. It publishes no transistor schematic, so every PMOS cell form above is standard construction for the era.

#### Down to the gates: The 4-bit adder, accumulator and decimal-adjust ROMs

- **Adder (ripple-through-carry)** — PUBLISHED (verified): the 4-bit adder is of the ripple-through carry type — four full-adder cells in a chain, the carry rippling bit to bit, one operand from the ADB register and the other from the accumulator and carry (also manual-published). The full-adder cell itself bottoms out in the primer's XOR/AND/OR structure — in 1971 PMOS, ratioed and dynamic gate forms rather than modern complementary static CMOS. That transistor-level form is standard construction for the era, not published.
- **Register (ADB buffer register)** — PUBLISHED (verified): one term of the addition comes from the ADB register, which communicates with the internal bus on one side and can transfer data to the adder — a 4-bit staging register between bus and adder. Its cell structure is not published — standard construction.
- **Register file (dynamic RAM array)** — PUBLISHED (verified): the index register is a dynamic RAM cell array of 16 x 4 bits and has two modes of operation — sixteen 4-bit registers or eight pairs, stored as dynamic charge like the address register and kept alive by the same refresh mechanism. The cell schematic is not published; the array organisation is.
- **Mux & shifter** — PUBLISHED (verified): the accumulator is provided with a shifter to implement rotate right and rotate left instructions, and the instruction register loads from the bus through a multiplexer. A 4-bit rotate shifter at gate level is a handful of pass-gate muxes selecting each bit's neighbour; the mux/shifter functions are published, their gate forms are standard construction.
- **ROM (lookup table)** — PUBLISHED (verified): the special ROMs perform a code conversion for DAA (decimal adjust accumulator) and KBP (keyboard process) instructions. A ROM at gate level is a decoder — an AND/NAND tree over the input bits — driving an array where a transistor present at an intersection stores a 1: the table IS the transistor pattern. The special ROMs' dimensions are not published (unverifiable — fact records); the ROM structure is standard construction.
- **Comparator** — PUBLISHED (verified): the condition logic senses ADD = 0 and ACC = 0 conditions, the state of the carry FF, and the state of an external signal (TEST) to implement JCN and ISZ. Zero-detect at gate level is a NOR tree over the register's bits — every bit must be 0 for the output to rise. Function published; gate form standard construction.
- **Flip-flop (1-bit registers)** — The carry flag and the double-cycle flag are 1-bit state: PUBLISHED (verified) that a double cycle FF is set from any one of 5 double-length instructions. In a dynamic PMOS design these flags are dynamic latch stages, not the ≈20–28-transistor static CMOS D flip-flop of the primer — the primer's cell is the modern form; the 4004's exact cell schematics are not published.
- **FSM / controller** — Decode is hardwired: PUBLISHED (verified) that each instruction word divides into an upper 4-bit OPR field (operation code) and a lower 4-bit OPA field (modifier), and those fields feed decode gates that drive the datapath directly — there is no microcode ROM anywhere in the 4004. Gate form standard construction.

The 4004 is the exception that proves this zoo's rule: its 1973 manual genuinely publishes gate-adjacent structure — dynamic RAM arrays, a ripple-carry adder, a lookahead incrementer, tri-state bus drivers — and every such claim above is verified in the fact records with the manual's exact words. What even Intel never published is transistor schematics, so every PMOS cell form above is labelled standard construction for the era. (New Level-4 fact record scopes exactly this boundary.)

**New fact records added in this run** (appended to `files/fact-records.json`; statuses per the fact standard):

- [unverifiable] **NVIDIA Hopper H100** — Level-4 gate-level structure not published by NVIDIA; decomposition is standard textbook construction throughout.
- [unverifiable] **Google TPU v4** — Level-4 gate-level structure not published in the ISCA 2023 paper or elsewhere; standard construction throughout.
- [unverifiable] **Apple Firestorm — M1** — No primary source exists at any level; Level-4 is standard construction throughout, sizes rest on microbenchmark measurements.
- [unverifiable] **Cerebras WSE-2** — Primitive-level parameters published (Level-3 records) but no gate-level structure; standard construction below the primitive level.
- [unverifiable] **Broadcom Tomahawk 5 (BCM78900)** — Gate-level structure not published by Broadcom; standard construction throughout.
- [unverifiable] **Scan Design & ATPG Architecture** — The scan cell’s gate structure (mux + D flip-flop) is the standardised definition of the technique, not a vendor netlist; per-design sizing unpublished.
- [verified] **Intel 4004** — Level-4 boundary: the MCS-4 User’s Manual publishes gate-adjacent organisation (dynamic RAM cell arrays, ripple-through-carry adder, carry look-ahead incrementer, tri-state output buffers) but no transistor schematics; PMOS cell forms labelled standard construction. Source: Intel MCS-4 User’s Manual (1973), §III.

### MOS Technology 6502

#### Down to the gates: The decode PLA and the timing sequencer — the whole control unit in one array

The 6502 is the best-documented chip at gate level in this whole zoo: the visual6502 project reverse-engineered its complete transistor-level netlist, and Ken Shirriff has published transistor-level explanations of its circuits. The whole chip is 4,528 transistors — 3,510 enhancement transistors and 1,018 depletion pull-up transistors (verified — K. Shirriff, righto.com, January 2013). The technology frame for everything below: the 6502 is depletion-load NMOS. A logic gate is a network of enhancement-mode pull-down transistors and a single depletion-mode pull-up (verified — VisualChips, "NMOS Depletion Mode Transistors"), so NAND and NOR are the natural gates and AND/OR are built from them; there is no PMOS anywhere, and a pass transistor is a single enhancement-mode switch, not a CMOS transmission gate.

- **ROM array (the decode PLA)** — PUBLISHED (verified — fact records): the original 650X schematics draw the decode ROM as Xs on a grid marking pulldown transistors — the table IS a transistor pattern. At gate level each of the 130 rows is a wired-NOR: a depletion-mode pull-up holds the row line high, and an enhancement pulldown sits at every programmed intersection, so the row fires only when the programmed pattern of IR bits, G and T all match. The 130 × 21 geometry and the 131 outputs are published (Level-3 records); the per-row transistor count is the number of programmed intersections in that row, and no per-row table is published — standard construction fitted to the published ROM format.
- **Decoder (the G group signals)** — The three group signals are three gates' worth of logic, and their equations are published: G1 = IR0, G2 = IR1, G3 = !IR0 & !IR1. G1 and G2 are wires; G3 is a 2-input NOR of IR0 and IR1 — one depletion pull-up and two enhancement pulldowns in parallel. The equations are published (Level-3 records); the NOR gate form is standard construction.
- **Register (instruction register + predecode register)** — Each bit of the 8-bit IR is a latch: two cross-coupled NMOS inverters (each inverter = one depletion pull-up + one enhancement pull-down) with enhancement pass transistors gating the write on the clock phase. The inverted IR outputs (notir0–notir7 in the visual6502 signal catalogue) exist because the PLA needs both polarities of every bit. Standard construction; the IR's existence and the inverted outputs are published.
- **Shift register (the T-state sequencer)** — A one-hot shift register is a chain of latches, one per state, clocked by alternating phi1/phi2 pass transistors so the single high bit steps T0 → T1 → … → T5. The T0–T5 states and their role as PLA inputs are published (Level-3 records); the latch count and gate form are standard construction.
- **Random logic (the additional-decoding layer)** — Combinational NAND/NOR networks in depletion-load NMOS: every gate is a pull-down network of enhancement transistors under one depletion pull-up. NAND and NOR dominate because, in Shirriff's words, "these are much easier to implement with transistors than AND and OR gates" (verified — righto.com, January 2013). The middle third of the die being this layer is published (Level-3 records); its gate count is not — standard construction.
- **Latch (control-signal latching)** — The ~44 datapath control signals are latched during phi2 (published, Level-3 records). At gate level each latched control line is a latch stage driven by the PLA/random-logic output through a phi2 pass transistor — the latch freezes the control for the cycle while phi1 executes it. Standard construction fitted to a published timing fact.
- **Adder (the PC incrementer)** — A 16-bit adder chain: sixteen one-bit adders, each bottoming out in the primer's full-adder structure (XOR/AND/OR built as NMOS NAND/NOR networks), with the PCLC/PCHC carry signals crossing the low/high byte boundary. The incrementer is a pure combinational adder with no internal latches (published, Level-3 records); whether its carry is ripple or lookahead is not published, and this decomposition does not claim either.
- **Register (address and data latches)** — ABL, ABH, DL and DOR are 8-bit latches of the same cross-coupled-inverter form, written on phi1 or phi2 as published (Level-3 records). Their cell structure is not published — standard construction.
- **Pass transistors / mux (bus interconnect and precharge)** — PUBLISHED (verified — fact records): all pass gates are enhancement mode, and the precharges for all four internal buses are enhancement mode. A pass transistor is one enhancement transistor: it passes a full low and a threshold-degraded high, which is why the buses are precharged high during phi2 (published, Level-3 records) — the bus is already high, and a small pass transistor only has to pull it low. Standard construction for the pass-gate mux form; the enhancement-mode and precharge facts are published.

#### Down to the gates: The ALU — two nibble adders with a decimal adjust

- **Mux (ALU input selection)** — Pass-transistor muxes: the 2-option side is two enhancement pass transistors, the 3-option side is three, each gated by its select signal from the PLA. The 2/3 option counts are published (Level-3 records); the single-transistor-per-input form is standard construction.
- **Adder (two 4-bit nibble adders)** — Each nibble adder is four full adders in a chain, each full adder two half adders of XOR/AND/OR gates (primer structure), all built as NMOS NAND/NOR networks with depletion pull-ups. The carry-out from bit 3 is the hand-off to the high nibble (published, Level-3 records). The gate form is standard construction.
- **Logic network (AND / OR / EOR / shift)** — AND, OR and EOR are the same NAND/NOR network with different output taps; the shift (SRS) is pass-transistor steering, not a barrel shifter (published, Level-3 records). The NAND and NOR of the two operands' top bits are already available in the ALU for other purposes (published — Shirriff) — the logic network is shared infrastructure. Standard construction.
- **Comparator (the decimal adjust)** — PUBLISHED (verified — fact records, Level-3): the whole decimal mode is five enhancement pulldown transistors — three for ADC (t2750, t2202, t2556) and two for SBC (t1329, t3212) — gated by the DAA and DSA control signals. At transistor level there is nothing else: five named pulldowns, each one transistor, is the entire decimal-adjust circuit. This is the most chip-specific gate-level fact in the zoo.
- **Register (the ADD output register)** — Eight latches of the standard cross-coupled-inverter form, written during phi2 (published, Level-3 records). Standard construction.
- **Comparator (flag generation)** — PUBLISHED (verified — fact records): Shirriff publishes the actual transistor implementation of the overflow circuit. The NAND of A7 and B7 is three transistors; the NOR of A7 and B7 is three transistors; the overflow circuit itself is a three-transistor NOR feeding a four-transistor circuit that implements an AND gate and a NOR gate in a single circuit. Quote: "The circuit above shows the actual implementation of the overflow circuit in the 6502 using NMOS transistors." The carry-out (ACR) and overflow (AVR) are latched during phi2 and used in phi1 (published, Level-3 records).
- **Register (the status register P)** — Eight latches of the standard form; the D flag that selects decimal mode lives here (published, Level-3 records). Standard construction.
- **FSM / controller** — The ALU has no sequencer of its own: at gate level the "controller" is the PLA and random logic elsewhere on the die plus the phi1/phi2 clocking discipline. The ALU's state is exactly one register (ADD) plus the flags. Standard construction fitted to published structure.

Published vs constructed, stated plainly: the 6502 is the one entry in this zoo whose complete transistor netlist has been reverse-engineered in public (visual6502), so more of its gate level is verifiable than any other chip's — the 4,528-transistor count, the five named decimal-adjust pulldowns, and the overflow circuit's exact transistor schematic are published chip facts, verified in the fact records. Everything else below the primitive level is standard NMOS depletion-load construction fitted to the published block structure: pull-down networks of enhancement transistors, single depletion-mode pull-ups, single-transistor pass gates, cross-coupled-inverter latches. No per-block gate counts are published, and this decomposition invents none.

**New fact records added in this run** (appended to `files/fact-records.json`; statuses per the fact standard):

- [verified] **MOS Technology 6502** — 4,528 transistors: 3,510 enhancement + 1,018 depletion pull-ups. Source: K. Shirriff, righto.com, January 2013.
- [verified] **MOS Technology 6502** — NMOS gate = pull-down network + single depletion-mode pull-up. Source: VisualChips, "NMOS Depletion Mode Transistors".
- [verified] **MOS Technology 6502** — Original schematics draw the decode ROM as Xs on a grid marking pulldown transistors. Source: Visual6502wiki, "650X Schematic Notes" (NESdev wiki). Note: the Rev. C blueprints are preliminary; PLA line ordering differs from the Rev. D production chip.
- [verified] **MOS Technology 6502** — Overflow circuit's actual transistor implementation published. Source: K. Shirriff, righto.com, January 2013.
- [verified] **MOS Technology 6502** — Overflow final stage: four transistors implement an AND gate and NOR gate in a single circuit. Source: K. Shirriff, righto.com, January 2013.
- [verified] **MOS Technology 6502** — Bus precharge transistors are enhancement mode. Source: VisualChips, "NMOS Depletion Mode Transistors".
- [unverifiable] **MOS Technology 6502** — Per-primitive gate/transistor allocation (PLA row transistor counts, T-state shift-register latch count, ALU gate-level schematic, PC incrementer carry scheme) not published as a citable table; standard construction below the primitive level except the overflow circuit and the five decimal-adjust transistors. Sources: visual6502 project pages, righto.com.
