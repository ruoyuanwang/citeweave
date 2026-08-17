# Graph Neural Networks in Drug Discovery: A Bibliometric Portrait of a Rapidly Expanding Research Frontier (2018–2023)

## Structured Abstract
**Background:** Graph neural networks (GNNs) have emerged as a transformative computational paradigm in drug discovery, yet a systematic quantitative overview of the field’s structure and evolution is lacking.
**Objective:** This report provides a reproducible bibliometric analysis of the GNN-in-drug-discovery literature, mapping its growth, key contributors, thematic concentrations, and intellectual structure.
**Methods:** A corpus of 456 unique works was acquired from a single bibliographic source via cursor pagination until exhaustion [E001, E002]. Metadata completeness was assessed across 11 fields [E003]. Descriptive, trend, and network analyses were performed using whole counting, Bradford zoning, and candidate-first association-strength mapping with Louvain community detection [E019, E022–E027, E072].
**Results:** The corpus spans 2018–2023, with annual output peaking at 190 works in 2023 and a compound annual growth rate of 185.60% [E004, E005, E006]. The most prolific source is *Briefings in Bioinformatics* (45 publications) and the most prolific author is Thin Nguyen (19 publications) [E008, E010]. The citation distribution is highly skewed (mean 59.82, median 24.00), with 4.82% of works uncited [E015, E016]. Thematic networks reveal a dense core centered on “Computer science,” “Artificial intelligence,” and “Machine learning” [E027].
**Conclusion:** The GNN-in-drug-discovery domain exhibits rapid growth, concentrated productivity in a few journals and institutions, and a strongly interconnected conceptual vocabulary, while citation-based networks remain sparse due to the field’s nascency.

## 1. Introduction
### 1.1 Background
Graph neural networks provide a natural framework for representing molecular structures and biological interaction networks, leading to their rapid adoption in computational drug discovery. Works such as GraphDTA and Attentive FP report state-of-the-art performance in their abstracts for drug–target affinity prediction and molecular property prediction [E028, E029].

### 1.2 Rationale & Objective
Despite the proliferation of primary research, a data-driven overview of the field’s bibliometric landscape is absent. This report aims to quantify the growth, productivity, impact, and intellectual structure of the GNN-in-drug-discovery literature using reproducible bibliometric methods [E069, E070].

### 1.3 Research Questions
- **RQ1:** What are the volume, growth trajectory, and temporal span of the literature? [E004, E006, E007]
- **RQ2:** Which sources, authors, and institutions are most productive? [E008–E013]
- **RQ3:** What is the citation impact and its distribution? [E015–E018]
- **RQ4:** What are the dominant thematic concentrations and their evolution? [E020, E027]
- **RQ5:** What is the collaborative and intellectual structure as revealed by coauthorship, cocitation, and bibliographic coupling networks? [E022, E024, E025]

## 2. Methods
### 2.1 Data Acquisition and Corpus Assembly
A single bibliographic source was queried. Cursor pagination continued until the source returned no next cursor, yielding 456 unique source records that matched the source-reported expected count of 456, confirming acquisition completeness [E002]. Deterministic exact deduplication produced a canonical corpus of 456 unique works [E001].

### 2.2 Metadata Quality Assessment
Field completeness was measured as the percentage of non-null canonical values. Coverage was 100% for title, publication date, year, document type, source ID, cited-by count, and reference count. Language coverage was 99.56%, DOI coverage was 97.37%, publisher coverage was 96.05%, and abstract coverage was 85.09% [E003].

### 2.3 Analytical Procedures
- **Descriptive statistics:** Annual output was whole-counted by publication year [E007]. Productivity rankings for sources, authors, and institutions were generated via whole counting [E009, E011, E013]. Citation impact was summarized using mean, median, and upper quantiles [E017].
- **Concentration analysis:** Bradford’s law of scattering was applied to sources with known names, dividing them into three zones by cumulative document thirds [E019].
- **Network analyses:** Six networks were constructed using a candidate-first association-strength approach. Candidate-first association-strength mapping selects nodes by occurrence threshold, then normalizes co-occurrence counts by the product of node weights (c_ij / (w_i * w_j)). Communities were detected via the Louvain algorithm, with small clusters merged [E022–E027, E072]. Networks were visualized using VOS-style and ForceAtlas2 layouts, with edge reduction via maximum-spanning forest plus strongest edges per node [E061–E066].
- **Thematic analysis:** Keyword temporal dynamics were tracked by counting unique works per normalized keyword and publication year [E020]. A three-field plot mapped leading author–source–keyword combinations [E021].

### 2.4 Reproducibility
All analyses are deterministic and parameterized. Figures were rendered from versioned analysis outputs with saved parameters and source tables [E052–E068]. Methodological references are curated and DOI-verified [E069–E072].

## 3. Results
### 3.1 Corpus Overview and Growth
The corpus spans six years, from 2018 to 2023 [E004]. Annual output grew from 1 publication in 2018 to 190 in 2023, with consistent year-over-year increases [E007]. The compound annual growth rate over this period is 185.60%. This endpoint-based CAGR describes the overall trajectory but does not establish monotonic year-over-year growth [E006]. The final year, 2023, represents the observed peak, though it may be incomplete depending on retrieval date [E005].

### 3.2 Document Types
All 456 works carry the source-supplied document type “article.” The source classification does not distinguish reviews from original articles; therefore the proportion of review content cannot be determined from this metadata field alone [E014]. This is visualized in Figure 5 [E056].

### 3.3 Most Productive Sources, Authors, and Institutions
- **Sources:** *Briefings in Bioinformatics* ranks first with 45 publications, followed by *Journal of Chemical Information and Modeling* (40) and *Bioinformatics* (29) [E008, E009]. Figure 2 displays the top 15 sources [E053].
- **Authors:** Thin Nguyen leads with 19 publications, followed by Dongsheng Cao and Tingjun Hou (11 each) [E010, E011]. Figure 3 displays the top 15 authors [E054].
- **Institutions:** The Chinese Academy of Sciences ranks first with 24 publications, followed by Central South University (22) and Hunan University (19) [E012, E013]. Figure 4 displays the top 15 institutions [E055].
- **Concentration:** Bradford zoning reveals a concentrated core: Zone 1 comprises only 5 sources that account for 148 documents, while Zone 3 contains 120 sources for 151 documents. Document counts per zone are rounded to the nearest 50 [E019]. This is visualized in Figure 8 [E059].

### 3.4 Citation Impact
- **Distribution:** The citation distribution is right-skewed. The mean citation count is 59.82, while the median is 24.00 [E015]. The 90th percentile is 113.5 citations, and the maximum is 1,210 [E017]. These values are source- and retrieval-date-dependent and should not be treated as stable bibliometric indicators [E015, E017]. Figure 6 visualizes the distribution capped at the 98th percentile [E057].
- **Zero citation:** 22 works (4.82%) have zero source-reported citations [E016].
- **Top-cited documents:** The most-cited work is “GraphDTA: predicting drug–target binding affinity with graph neural networks” (2020) with 1,210 citations, followed by “Pushing the Boundaries of Molecular Representation for Drug Discovery with the Graph Attention Mechanism” (2019) with 1,056 citations [E018]. Figure 7 lists the top 15 documents [E058].

### 3.5 Thematic Landscape
- **Keyword trends:** The most globally frequent keywords are “Computer science” (437 documents), “Artificial intelligence” (393), and “Machine learning” (326). All show consistent year-over-year increases in document count, though the magnitude of increase varies [E020]. Figure 9 tracks the top 15 keywords over time [E060].
- **Three-field plot:** The most frequent author–source–keyword combinations involve Wen Zhang and *Briefings in Bioinformatics*, with five documents each across the keywords “Artificial intelligence,” “Graph,” “Machine learning,” “Theoretical computer science,” and “Computer science” [E021].
- **Keyword co-occurrence network:** The network contains 72 displayed nodes and 198 edges, organized into 7 clusters [E027]. The core cluster is anchored by “Computer science,” “Artificial intelligence,” “Machine learning,” “Graph,” and “Theoretical computer science” [E027]. This structure is contingent on the minimum occurrence threshold (5) and edge-reduction parameters [E027]. Figure 12 visualizes the network [E063], Figure 16 provides a temporal overlay [E067], and Figure 17 shows a density map [E068].

### 3.6 Network Structures
- **Coauthorship:** The network displays 60 nodes and 127 edges across 10 clusters [E024]. Tingjun Hou and Dongsheng Cao are central nodes in one cluster, while Mingyue Zheng anchors another [E024]. Figure 10 visualizes the network [E061].
- **Institution collaboration:** The network contains 71 nodes and 138 edges in 10 clusters [E026]. The Chinese Academy of Sciences and the University of Chinese Academy of Sciences form a prominent dyad [E026]. Figure 11 visualizes the network [E062].
- **Bibliographic coupling:** The network comprises 72 nodes and 210 edges in 4 clusters, constructed from 430 documents that met the inclusion criteria (minimum 5 references, minimum 2 shared references) [E022]. Top nodes by weighted degree include works on molecular representation and bioinformatics applications [E022]. Figure 15 visualizes the network [E066].
- **Cocitation:** The network displays 72 nodes and 194 edges in 5 clusters. Top nodes are identified by OpenAlex IDs; resolution to bibliographic metadata requires external lookup [E025]. Figure 13 visualizes the network [E064].
- **Citation network:** The direct citation network, constructed with a minimum occurrence threshold of 5, contains only 5 nodes and 3 edges in 2 clusters. This sparsity reflects both the short citation window and the thresholding of the candidate pool [E023]. Figure 14 visualizes the network [E065].

## 4. Discussion
### 4.1 Interpretation of Growth and Productivity
The rapid growth (CAGR 185.60%) indicates that GNNs in drug discovery are in a “hot topic” phase, consistent with the rapid methodological advances documented in the top-cited works [E006, E018, E028–E031]. The concentration of output in a few journals (Zone 1: 5 sources) and institutions (Chinese Academy of Sciences leads) suggests a nascent field with dominant publishing venues and research hubs [E008, E012, E019].

### 4.2 Interpretation of Citation Impact
The skewed citation distribution, with a mean (59.82) substantially exceeding the median (24.00), is typical of bibliometric data and reflects a few highly influential papers [E015, E017]. The 4.82% uncited rate may reflect the recency of the corpus, as newer works have had limited time to accumulate citations [E016]. The top-cited documents are methodological papers introducing novel architectures (e.g., GraphDTA, Attentive FP), underscoring the field’s method-driven character [E018, E028, E029].

### 4.3 Interpretation of Thematic and Network Structure
The keyword co-occurrence network reveals a conceptual core around computer science, AI, and machine learning, with “Graph” as a central bridging term [E027]. The temporal overlay suggests this core has been stable and intensifying [E067]. The sparse direct citation network (5 nodes, 3 edges) is an expected artifact of the short citation window (2018–2023) and the minimum occurrence threshold, which together limit the accumulation of within-corpus citations [E023]. In contrast, the richer bibliographic coupling and cocitation networks indicate shared intellectual foundations, even if direct citation links are not yet visible [E022, E025]. The coauthorship and institution collaboration networks reveal distinct research clusters, often centered around key productive authors and their institutions [E024, E026].

## 5. Limitations
- **Database and retrieval bias:** All findings are contingent on the single bibliographic source used, its coverage, and the retrieval date. Citation counts are source- and retrieval-date-dependent [E015, E017].
- **Citation window:** The corpus spans 2018–2023, meaning recent works, especially those from 2023, have had minimal time to accumulate citations. This directly limits the size and interpretability of the citation network [E023, E016].
- **Metadata dependence:** Analyses of keywords, document types, and abstracts rely on source-supplied metadata quality and classification practices. Abstract coverage is 85.09%; 68 works lack abstracts, limiting content-based analyses for those records. Keyword frequencies depend on source indexing and normalization [E003, E014, E020].
- **Network parameterization:** Network structures are a function of the chosen thresholds (e.g., minimum occurrence), normalization (association strength), and edge-reduction algorithms. Different parameters would yield different maps [E022–E027].
- **Content depth:** All content inferences are based on titles and abstracts, not full-text analysis. Claims about substantive findings of cited works are limited to what is reported in the abstract [E028–E051].
- **Productivity vs. quality:** Whole-count productivity rankings for authors, sources, and institutions are descriptive and must not be interpreted as measures of research quality or causal influence [E009, E011, E013].

## 6. Conclusion
This bibliometric analysis reveals a research domain—graph neural networks in drug discovery—characterized by explosive growth, concentrated productivity, and a tightly woven thematic core. The field is methodologically driven, as evidenced by its most-cited works and dominant keywords. While citation-based networks remain embryonic due to the short time window, coauthorship and bibliographic coupling networks already delineate a structured community. This report provides a reproducible, evidence-mapped baseline for researchers, policymakers, and funders to understand the landscape and evolution of this rapidly advancing frontier.
