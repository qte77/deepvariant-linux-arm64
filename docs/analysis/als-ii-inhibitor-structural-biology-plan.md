# Plan: Structural Biology Tool Selection for ALS II Inhibitor Discovery (UPEC/UTI)

## Context

**Paper**: "Genome-scale CRISPRi profiling reveals metabolic vulnerabilities of uropathogenic Escherichia coli in human urine" (DOI: 10.1101/2025.10.31.685768)

**Key findings from the paper**:
- Genome-wide CRISPRi screen in UPEC strain CFT073 identified **Acetolactate Synthase (ALS) II** (ilvG gene) as the sole active isoform for branched-chain amino acid (BCAA) biosynthesis in human urine
- BCAA synthesis (valine, leucine, isoleucine) is conditionally essential for E. coli survival in urine
- The herbicide **sulfometuron methyl** targets ALS but E. coli ALS II has inherent resistance; the authors engineered a re-sensitizing mutation as proof-of-concept
- Other vulnerabilities: iron acquisition, cell envelope integrity, arginine/methionine synthesis

**Goal**: Identify the most appropriate structural biology tool(s) to support discovery of novel ALS II inhibitors that could inhibit UPEC growth in urine.

---

## Target Characterization: E. coli ALS II (IlvG/IlvM)

| Property | Detail |
|----------|--------|
| Enzyme | Acetolactate synthase isozyme II (AHAS II) |
| Genes | ilvG (catalytic large subunit), ilvM (regulatory small subunit) |
| Cofactors | Thiamine diphosphate (ThDP), FAD, Mg²⁺ |
| Reaction | 2 pyruvate → acetolactate + CO₂ (first committed step of BCAA biosynthesis) |
| Known inhibitors | Sulfonylureas, imidazolinones (herbicides) — bind outside catalytic site |
| PDB structures (E. coli) | **2F1F** — regulatory subunit of isozyme III only |
| Closest homolog structures | **1OZF/1OZG/1OZH** — *K. pneumoniae* ALS (2.3 Å), **4RJJ/4RJK** — *B. subtilis*, **7EGV** — *T. harzianum* with harzianic acid inhibitor |
| Key challenge | No experimental crystal structure of E. coli IlvG catalytic subunit in PDB |

---

## Recommended Structural Biology Tool Pipeline

### Tier 1 (Primary Recommendation): AlphaFold 3

**Why AlphaFold 3 is the most appropriate tool:**

1. **No experimental structure exists** for E. coli ALS II catalytic subunit (IlvG) — AF3 can predict it with high confidence from sequence alone
2. **Protein–small molecule complex prediction** — AF3 can model IlvG bound to ThDP/FAD cofactors and candidate inhibitors, unlike AF2 which only predicts protein structure
3. **Multi-chain complex modeling** — AF3 can predict the IlvG–IlvM heterodimer with cofactors, capturing the biologically relevant assembly
4. **Validated accuracy** — AF3 shows greater accuracy for protein–ligand interactions than traditional docking tools (Nature, 2024)

**Access**: AlphaFold Server (alphafoldserver.com) or open-source Boltz-1 / OpenFold alternatives

**Workflow**:
1. Predict IlvG–IlvM heterodimer structure with ThDP + FAD + Mg²⁺
2. Validate against K. pneumoniae crystal structure (1OZH) — expect high structural conservation
3. Identify the herbicide binding channel (known to be outside catalytic site)
4. Map the resistance-conferring residues that distinguish ALS II from ALS I/III

### Tier 2 (Complementary): Structure-Based Virtual Screening

Once the AF3 model is obtained:

| Tool | Purpose |
|------|---------|
| **AutoDock Vina / GNINA** | Molecular docking of compound libraries into the herbicide binding pocket |
| **SiteAF3** (PNAS, 2025) | AI-enhanced binding site prediction on the AF3 model — especially useful for finding allosteric sites beyond the known herbicide channel |
| **YuelPocket** (PNAS, 2026) | GNN-based pocket detection for identifying druggable sites on the predicted structure |
| **PLIP** (Protein–Ligand Interaction Profiler) | Analyze interaction fingerprints of docked poses |

### Tier 3 (Validation & Optimization)

| Tool | Purpose |
|------|---------|
| **Rosetta / RoseTTAFold All-Atom** | Binding energy estimation, interface design for lead optimization |
| **MD simulations (GROMACS/OpenMM)** | Validate binding stability, assess selectivity vs human BCAT |
| **FPocket / DoGSiteScorer** | Complementary druggability assessment of predicted pockets |

---

## Recommended Strategy Summary

```
E. coli IlvG sequence
        │
        ▼
   AlphaFold 3  ──────────────────────────►  IlvG–IlvM–ThDP–FAD complex
        │                                            │
        ▼                                            ▼
  Validate vs K.pneumo                     SiteAF3 / YuelPocket
  crystal (1OZH)                           (binding site prediction)
        │                                            │
        ▼                                            ▼
  Map resistance residues              AutoDock Vina / GNINA
  (why ALS II resists                  (virtual screening of
   sulfometuron methyl)                 compound libraries)
        │                                            │
        ▼                                            ▼
  Design inhibitors that               MD validation (GROMACS)
  overcome ALS II resistance           Selectivity vs human enzymes
```

## Why NOT Other Tools

| Tool | Reason to deprioritize |
|------|----------------------|
| **Homology modeling (SWISS-MODEL, Modeller)** | Superseded by AF3 for accuracy; only useful if AF3 is unavailable |
| **Cryo-EM / X-ray crystallography** | Experimental — valuable but slow; AF3 provides starting model immediately |
| **Ligand-based virtual screening (ROCS)** | Requires known active ligands with good SAR data; sulfometuron methyl is the only known ALS II-relevant compound and it doesn't bind well |
| **Fragment-based screening** | Experimental, expensive; better after computational hits narrow the search space |

## Key Considerations

1. **Selectivity**: Human cells lack ALS — no human off-target concern for this specific enzyme, which is a major advantage for drug safety
2. **Resistance mechanism**: The paper shows ALS II has inherent resistance to sulfometuron methyl. Structural modeling of the resistance residues is critical to design inhibitors that overcome this
3. **Urine environment**: Inhibitor must be stable in urine pH (~5.5–7.0) and renal-excreted — physicochemical property filters should be applied during virtual screening
4. **Existing herbicide SAR**: The binding mode of sulfonylureas/imidazolinones in plant AHAS (PDB 5K6Q, 7EGV) provides starting pharmacophore hypotheses transferable to the AF3-predicted E. coli model

## Sources

- [AlphaFold 3 — Nature 2024](https://www.nature.com/articles/s41586-024-07487-w)
- [AlphaFold 3 drug design review — PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11292590/)
- [SiteAF3 — PNAS 2025](https://www.pnas.org/doi/10.1073/pnas.2521048122)
- [YuelPocket — PNAS 2026](https://www.pnas.org/doi/10.1073/pnas.2524913123)
- [K. pneumoniae ALS structure — PDB 1OZH](https://www.rcsb.org/structure/1OZH)
- [E. coli AHAS III regulatory subunit — PDB 2F1F](https://www.ncbi.nlm.nih.gov/Structure/pdb/2F1F)
- [T. harzianum AHAS with harzianic acid — PDB 7EGV](https://www.rcsb.org/structure/7EGV)
- [Plant AHAS crystal structure — Garcia 2017](https://febs.onlinelibrary.wiley.com/doi/10.1111/febs.14102)
- [Acetolactate synthase — Wikipedia](https://en.wikipedia.org/wiki/Acetolactate_synthase)
- [AlphaFold3 benchmarking — PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12661943/)
- [AI protein structure prediction 2025 — Charles River](https://www.criver.com/eureka/whats-hot-2025-protein-structure-prediction-using-ai)
