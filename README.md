# Silicon Zoo

A categorized, searchable library of the great ASIC architectures and microarchitectures across every field of silicon design — built to be shared.

**Live site:** https://luckymsk.github.io/silicon-zoo/

97 architecture entries across 14 categories (CPU, GPU, AI accelerators, Video & Media, Imaging & ISP, SoC platforms, DSP & Audio, Networking & DPU, Memory & Interconnect, Security, Automotive, Emerging, Testability & DFT, Low-Power Architecture), plus Foundations, a Silicon Success section (bandwidth, latency, power/PPA, testability, yield), a timeline and a glossary.

Every entry is broken down into its base building blocks — the fundamental pieces, how they connect, and the one or two pieces that make that architecture special. All 97 of 97 entries carry hand-written, chip-specific breakdowns. The drill-down goes three levels deep: 46 entries decompose into their signature blocks (Level-2), 45 into the fundamental circuit primitives — FIFOs, muxes, crossbars, arbiters, register files, SRAM banks, CAMs, adder trees (Level-3), and 45 all the way down to gates and transistors (Level-4). Every checked fact carries a primary source, an exact quote, and a status in `fact-records.json` (1,331 records).

- `index.html` — the full compendium as a single self-contained page
- `architecture-corpus.md` — the same content as a markdown corpus

The compendium keeps growing: new entries and deeper breakdowns are added over time. This repo is refreshed daily.
