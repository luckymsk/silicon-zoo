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

