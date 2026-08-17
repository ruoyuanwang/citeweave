# Microplastics Research 2004–2019: A Bibliometric Portrait of Growth, Key Actors, and Thematic Structure

## Structured Abstract
*Background*: Microplastics research has expanded rapidly, yet a systematic bibliometric overview of the field’s growth, key contributors, and intellectual structure is needed.
*Methods*: A corpus of 3,973 unique works [E001] was acquired from a single bibliographic source, covering 2004–2019 [E004]. We applied reproducible descriptive, productivity, citation, and network analyses using whole counting, Bradford zoning, and association-strength-normalized community detection [E069–E072].
*Results*: Annual output grew from 28 publications in 2004 to 1,329 in 2019 (CAGR 29.35%) [E005–E007]. *Marine Pollution Bulletin* (395 publications) [E008], Dario Mendez-Cuadro (127 publications) [E010], and the Chinese Academy of Sciences (109 publications) [E012] were the most productive source, author, and institution. Mean citations were 201.34 (median 61.00) [E015]; 22.78% of works were uncited [E016]. Bradford analysis identified a core of 4 journals [E019]. Network maps revealed four thematic clusters in bibliographic coupling [E022] and seven in keyword co-occurrence [E027].
*Conclusion*: The field exhibits rapid growth (CAGR 29.35%), concentrated publication venues, and a strong marine-environmental science core, as indicated by the dominance of marine-focused journals [E009] and keywords such as “Environmental science” and “Ecology” [E020, E027].

## 1. Introduction
*Context*: Microplastics have emerged as a global environmental concern, prompting a surge in scientific literature.
*Objective*: This report provides a comprehensive, reproducible bibliometric overview of the microplastics research domain from 2004 to 2019.
*Scope*: The analysis covers publication growth, leading sources, authors, institutions, citation impact, and the conceptual and social structure of the field.
*Methodological framing*: The study follows established bibliometric guidelines [E069] and employs science-mapping techniques [E070, E071] to ensure transparency and replicability.

## 2. Reproducible Methods
### 2.1 Data Acquisition and Corpus Construction
The source reported 4,002 results; 3,995 unique records were acquired (7 fewer than reported; the discrepancy may reflect source-side deduplication or retrieval artifacts) [E002]. After deterministic exact deduplication, the final canonical corpus comprises 3,973 unique works [E001]. Metadata completeness was assessed for every field; coverage is 100% for `cited_by_count`, `document_type`, `publication_date`, `reference_count`, `source_id`, and `year`, but only 43.32% for `abstract` [E003]. Abstract coverage of 43.32% limits full-text content analysis but does not affect keyword-based analyses, which rely on source-indexed terms [E003].

### 2.2 Analytical Procedures
- *Descriptive and productivity analyses*: Whole counting was used for annual output [E007], top sources [E009], top authors [E011], top institutions [E013], and document types [E014]. Compound annual growth rate (CAGR) was calculated from non-zero endpoints [E006].
- *Citation analysis*: Source-reported citation counts were summarized with descriptive statistics [E015, E017]; zero-cited works were identified [E016]; top-cited documents were ranked [E018].
- *Concentration analysis*: Bradford’s law of scattering was applied to sources with known names [E019].
- *Network and science mapping*: Candidate-first association-strength maps were constructed for bibliographic coupling [E022], citation [E023], co-authorship [E024], co-citation [E025], institution collaboration [E026], and keyword co-occurrence [E027]. Community detection used the Louvain algorithm [E072]; edge reduction employed a maximum-spanning forest plus strongest edges per node. Layouts were optimized via VOS-style or ForceAtlas2 algorithms with multiple restarts [E022–E027].
- *Keyword temporal dynamics*: Year-by-year document counts were computed for the most frequent normalized keywords [E020].
- *Three-field plot*: Whole-counted links among leading authors, sources, and keywords were tabulated [E021].

## 3. Results
### 3.1 Publication Growth and Corpus Profile
The corpus spans 2004–2019 [E004]. Annual output peaked in 2019 with 1,329 works [E005], growing from 28 in 2004 [E007]. The CAGR over this period is 29.35% [E006]. Year-over-year changes show a consistent upward trend with minor fluctuations; output grew from 133 publications in 2014 to 1,329 in 2019, with the largest single-year increase occurring in 2018 (+565) [E007]. All 3,973 works are classified as “article” in the source metadata, though this may reflect source classification practices rather than the true absence of reviews or other types [E014].

### 3.2 Leading Sources, Authors, and Institutions
- *Sources*: *Marine Pollution Bulletin* leads with 395 publications, followed by *Environmental Pollution* (323) and *The Science of The Total Environment* (243) [E008, E009]. Bradford zoning reveals a core Zone 1 of only 4 journals containing 1,192 documents; Bradford zoning excludes records without source names (n ≈ 284), so the core represents 1,192 of 3,689 classifiable documents [E019].
- *Authors*: Dario Mendez-Cuadro is the most prolific author with 127 publications, followed by Albert A. Koelmans (47) and Richard C. Thompson (45) [E010, E011].
- *Institutions*: The Chinese Academy of Sciences ranks first (109 publications), followed by Centre National de la Recherche Scientifique (91) and East China Normal University (69) [E012, E013].

### 3.3 Citation Impact
The mean citation count is 201.34, while the median is 61.00, indicating a right-skewed distribution [E015, E017]. The 90th percentile is 553 citations, and the maximum is 7,932 [E017]. A total of 905 works (22.78%) have zero source-reported citations [E016]. The most-cited work is “Microplastics in the marine environment” (2011) with 7,932 citations, followed by “Microplastics as contaminants in the marine environment: A review” (2011, 5,961 citations) [E018].

### 3.4 Thematic and Conceptual Structure
- *Keyword trends*: “Microplastics” is the dominant keyword (2,770 global documents), rising from 3 documents in 2004 to 1,024 in 2019. “Environmental science,” “Biology,” and “Ecology” show parallel growth trajectories [E020].
- *Keyword co-occurrence network*: The network (72 nodes, 202 edges) reveals 7 clusters [E027]. The central cluster is anchored by “Microplastics,” “Environmental science,” “Biology,” and “Ecology” [E027].
- *Bibliographic coupling network*: The network (72 nodes, 228 edges) partitions into 4 clusters, with top nodes including reviews on environmental occurrence, fate, and distribution of microplastics [E022].
- *Three-field plot*: Strong linkages exist between Dario Mendez-Cuadro, “Microplastics,” and *Environmental Pollution* (16 documents), and between Huahong Shi, “Microplastics,” and the same journal (15 documents) [E021]. An equally strong linkage between B. K. Kardashev, “Materials science,” and *Physics of the Solid State* (16 documents) is also present and may reflect noise or misclassification in the corpus; the three-field plot is a filtered descriptive view, not a causal pathway [E021].

### 3.5 Social and Intellectual Structure
- *Co-authorship network*: The network (59 nodes, 121 edges) shows 10 clusters. Huahong Shi and Tamara S. Galloway are central nodes in distinct collaborative communities [E024].
- *Institution collaboration network*: The network (72 nodes, 176 edges) reveals 10 clusters, with CNRS and the Chinese Academy of Sciences as major hubs [E026].
- *Co-citation network*: The network (72 nodes, 205 edges) forms 4 clusters, with foundational works by openalex:W2164526292 and openalex:W2014173070 receiving the highest weighted degree [E025]. Co-citation nodes are identified by source-internal IDs; these correspond to highly cited works in the reference lists but cannot be resolved to titles from the available evidence [E025].
- *Citation network*: The direct citation network (60 nodes, 81 edges) shows 4 clusters, with “Accumulation of Microplastic on Shorelines Woldwide: Sources and Sinks” as a top node [E023].

## 4. Discussion
*Interpretation of growth*: The 29.35% CAGR [E006] and the surge from 2014 onward [E007] suggest a field in a rapid expansion phase, likely driven by heightened environmental awareness and policy interest. The peak in 2019 [E005] must be interpreted with caution as the final year may be incomplete [E005 caveat].
*Concentration of knowledge production*: The Bradford core of 4 journals [E019] and the dominance of a few prolific authors and institutions [E009, E011, E013] indicate a concentrated publication landscape. This observation is purely descriptive and does not imply research quality [E009, E011, E013 caveats].
*Citation skew and uncitedness*: The large gap between mean and median citations [E015] and the 22.78% uncited rate [E016] reflect a typical skewed citation distribution. Recent works have had less time to accumulate citations, which partially explains the uncited share [E016 caveat].
*Thematic structure*: The keyword trends [E020] and network maps [E022, E027] confirm a persistent marine-environmental core. The dominance of marine-focused journals [E009] and keywords such as “Environmental science” and “Ecology” [E020, E027] indicates a strong marine-environmental science core.
*Network insights*: The co-authorship and institution collaboration networks [E024, E026] visualize the social structure, highlighting distinct regional and institutional clusters. The bibliographic coupling and co-citation networks [E022, E025] delineate the intellectual base, with review articles serving as key integrating nodes.

## 5. Limitations
- *Completeness and coverage*: The corpus is limited to a single source and retrieval date. The source reported 4,002 results, and 3,995 unique records were acquired (7 fewer than reported; the discrepancy may reflect source-side deduplication or retrieval artifacts) [E002]. Abstract coverage is only 43.32% [E003], limiting full-text content analysis but not keyword-based analyses.
- *Citation window*: Citation counts are source- and retrieval-date-dependent [E015 caveat]. Works from later years (e.g., 2019) have shorter citation windows, biasing impact metrics downward for recent publications [E016 caveat].
- *Network parameterization*: Network maps are filtered descriptive views based on specific thresholds (e.g., minimum occurrence, edge reduction) [E022–E027]. Centrality measures describe the constructed network and do not establish causal or substantive importance [E022–E027 caveats].
- *Document type homogeneity*: The corpus consists entirely of “articles” [E014], which may reflect source classification practices rather than the true absence of other document types [E014 caveat].
- *Abstract-only inference*: All content interpretation is based on titles and abstracts; no full-text analysis was performed [E028–E051 caveats].

## 6. Conclusion
This bibliometric analysis maps the microplastics research landscape from 2004 to 2019, revealing a field characterized by rapid growth (CAGR 29.35% [E006]), concentrated in a core of marine-environmental journals [E019], and driven by a network of highly productive authors and institutions [E010–E013]. The intellectual structure, as visualized through coupling and co-citation networks [E022, E025], is anchored by foundational reviews on marine microplastic occurrence and methods. Thematic analysis confirms a strong marine focus, as indicated by the dominance of marine-focused journals [E009] and keywords such as “Environmental science” and “Ecology” [E020, E027]. The provided evidence base, with its explicit limitations, offers a transparent and reproducible foundation for understanding the evolution of this critical research domain.
