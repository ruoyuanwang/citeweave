# A Bibliometric Landscape of Climate-Change Research (1990–2021): Productivity, Impact, and Intellectual Structure

## Structured Abstract
**Background**: Climate-change research has expanded rapidly, yet comprehensive bibliometric portraits of the field’s scale, growth, key actors, and thematic organization remain essential for research governance.
**Methods**: A reproducible workflow acquired 152,030 unique source records [E002], deduplicated to a canonical corpus of 151,562 works [E001] spanning 1990–2021 [E004]. We computed descriptive productivity, citation, and network indicators following established bibliometric guidelines [E069, E070, E071, E072].
**Results**: Annual output grew from 127 publications in 1990 to 19,170 in 2021 (CAGR 17.57%) [E005, E006, E007]. *Climatic Change* led source productivity (2,093 works) [E008, E009]. The Chinese Academy of Sciences was the most prolific institution (3,468 works) [E012, E013]. Mean citations were 49.56 (median 14.00) [E015, E017]; 19.39% of works remained uncited [E016]. Network analyses revealed seven thematic keyword clusters [E027, E063] and concentrated collaboration patterns [E024, E026].
**Conclusions**: The field exhibits compound annual growth of 17.57% between 1990 and 2021, concentrated core sources, and a highly skewed citation distribution. Thematic structure is dominated by climate-change, geography, and environmental-science keywords [E020, E027]. All findings are bounded by source-dependent metadata coverage and retrieval-date-dependent citation windows.

## 1. Introduction
### 1.1 Rationale
Understanding the structure and evolution of climate-change scholarship is critical for funders, policymakers, and researchers. Bibliometric methods provide systematic, reproducible descriptions of research landscapes [E069].

### 1.2 Objectives
This report aims to (i) quantify the growth and productivity of climate-change research from 1990 to 2021 [E004, E007]; (ii) identify leading sources, authors, and institutions [E009, E011, E013]; (iii) characterize citation impact and its skew [E015, E017]; and (iv) map the intellectual structure through co-occurrence, collaboration, and citation-based networks [E022–E027].

### 1.3 Scope
The analysis is based on a single-source, deduplicated corpus of 151,562 works [E001] acquired via exhaustive cursor pagination [E002]. All claims are descriptive; no causal or quality inferences are made.

## 2. Reproducible Methods
### 2.1 Data Acquisition and Corpus Construction
Records were retrieved from OpenAlex using cursor pagination until exhaustion, yielding 152,030 unique source records against a source-reported expectation of 152,599 [E002]. Deterministic exact deduplication on normalized DOI (with fallback to normalized title and first-author surname for records without DOI) produced a canonical corpus of 151,562 works [E001].

### 2.2 Metadata Coverage Audit
Field completeness was assessed as the percentage of non-null canonical values. Coverage was 100% for publication year, document type, cited-by count, reference count, and source ID; 99.96% for title; 99.5% for language; 80.8% for DOI; 79.79% for publisher; and 74.73% for abstract [E003].

### 2.3 Descriptive Analyses
Annual output was whole-counted by publication year [E007]. Compound annual growth rate (CAGR) was calculated from the first and last non-zero endpoints [E006]. Top sources, authors, and institutions were ranked by whole-counted publications [E009, E011, E013]. Document types were tabulated from source-supplied labels [E014]. Citation distributions were summarized with mean, median, percentiles, maximum, and zero-cited share [E015, E016, E017]. Bradford zoning was applied to sources with known names [E019]. Keyword temporal dynamics were tracked for the most frequent normalized keywords [E020]. A three-field plot linked leading authors, sources, and keywords [E021].

### 2.4 Network Analyses
Six candidate networks were constructed: bibliographic coupling [E022], direct citation [E023], coauthorship [E024], cocitation [E025], institution collaboration [E026], and keyword co-occurrence [E027]. All used association-strength normalization, Louvain community detection [E072], and edge reduction via maximum-spanning forest plus strongest edges per node (maximum 4 edges per node). Minimum occurrence thresholds were: bibliographic coupling 10, direct citation 456, coauthorship 30, cocitation 261, institution collaboration 145, and keyword co-occurrence 513 [E022–E027]. Networks were visualized with VOS-style or ForceAtlas2 layouts [E061–E068].

### 2.5 Methodological References
The workflow follows bibliometric guidelines [E069] and draws on established science-mapping [E070] and network-visualization [E071] approaches.

## 3. Results
### 3.1 Corpus Overview and Metadata Quality
The final corpus contains 151,562 unique works [E001]. Metadata completeness is high for core fields but lower for abstracts (74.73%) and publisher (79.79%) [E003]. All works carry the source-supplied label “article,” which reflects the source’s classification practices and may include reviews, editorials, and other types not distinguished in the source metadata [E014].

### 3.2 Temporal Dynamics
The publication window spans 1990–2021 [E004]. Output rose from 127 works in 1990 to a peak of 19,170 in 2021 [E005, E007], corresponding to a CAGR of 17.57% [E006]. Year-over-year changes were not monotonic; negative changes occurred in 1994 (−35) and 1998 (−1), and periods of deceleration were observed (e.g., 2015: +478 vs. 2014: +928) [E007]. The final year may be incomplete [E005 caveat].

### 3.3 Leading Sources, Authors, and Institutions
*Climatic Change* ranks first among sources with 2,093 publications [E008, E009]. The top-ranked author string in the source metadata is “AGB Poore” (5,808 works) [E010, E011]; however, this string is a known metadata artifact (see Section 5.3) and is excluded from substantive interpretation. Among the remaining author records, Kristie L. Ebi ranks highest with 150 publications [E011]. The Chinese Academy of Sciences leads institutions (3,468 works) [E012, E013]. Productivity rankings are descriptive and do not imply quality [E009, E011, E013 caveats].

### 3.4 Citation Impact
Mean citations are 49.56 and the median is 14.00 [E015, E017]. The distribution is highly skewed: the 90th percentile is 110 citations, the maximum is 22,581, and 19.39% of works are uncited [E016, E017]. Citation counts are source- and retrieval-date-dependent [E015 caveat]. The uncited rate reflects the full corpus including recent publications that have had less time to accumulate citations [E016 caveat].

### 3.5 Source Concentration
Among records with known source names (138,775 documents), Bradford zoning identifies 114 sources in Zone 1, accounting for 33.2% of publications (46,135 documents) [E019, E059]. Records without a source name are excluded from this analysis [E019 caveat].

### 3.6 Keyword Trends
“Climate change,” “Geography,” and “Environmental science” are the most frequent keywords, each showing increasing annual document counts over the observed period [E020, E060]. Evidence for “Environmental science” is available through 2005 in the current data extract [E020].

### 3.7 Network Structures
The keyword co-occurrence network (72 nodes, 196 edges, 7 clusters) is dominated by “Climate change,” “Geography,” and “Environmental science” [E027, E063]. The coauthorship network (60 nodes, 149 edges, 10 clusters) features the artifact string “AGB Poore” as the most central node by weighted degree; among verified author names, Christoph Müller and Frank Ewert show high centrality [E024, E061]. The institution collaboration network (72 nodes, 192 edges, 7 clusters) highlights CNRS and the Chinese Academy of Sciences as central hubs [E026, E062]. The cocitation network (72 nodes, 192 edges, 5 clusters) and bibliographic coupling network (72 nodes, 197 edges, 8 clusters) reveal thematic groupings of frequently co-cited and reference-sharing documents [E022, E025, E064, E066]. The direct citation network is sparser (60 nodes, 55 edges, 10 clusters) [E023, E065]. All network centrality measures are descriptive of this corpus and parameterization and do not establish causal or substantive importance [E022–E027 caveats].

## 4. Discussion
### 4.1 Interpretation of Growth
The 17.57% CAGR between 1990 and 2021 [E006] indicates rapid expansion, consistent with increasing global attention to climate change. However, year-over-year changes were not monotonic [E007], and the endpoint CAGR does not establish a sustained exponential trend [E006 caveat]. The peak in 2021 [E005] may reflect both genuine growth and incomplete-year effects [E005 caveat].

### 4.2 Concentration and Skew
The dominance of a few sources [E008, E019] and institutions [E012] aligns with typical bibliometric patterns of cumulative advantage. The citation skew (mean 49.56 vs. median 14.00) [E015] and 19.39% uncited rate [E016] underscore that impact is unevenly distributed; recent works have had less time to accumulate citations [E016 caveat].

### 4.3 Thematic Structure
The keyword co-occurrence network [E027] and temporal trends [E020] confirm that core themes (climate change, geography, environmental science) are tightly interlinked. The three-field plot [E021] is dominated by the artifact author string “AGB Poore” and the repository source “AgEcon Search (University of Minnesota, USA),” which limits its interpretability for thematic mapping; the plot is a filtered descriptive view, not a causal pathway [E021 caveat].

### 4.4 Network Insights
Collaboration networks [E024, E026] reveal descriptive patterns of coauthorship and institutional linkages, while citation-based networks [E022, E023, E025] identify documents that are frequently co-cited or share references within this corpus and parameterization. These structures are parameter-dependent and should not be interpreted as causal maps of influence or as indicators of substantive importance [E022–E027 caveats].

## 5. Limitations
### 5.1 Source and Retrieval Constraints
Completeness is relative to the OpenAlex source, query, and retrieval date [E002 caveat]. Citation counts are source- and retrieval-date-dependent [E015 caveat]; works have unequal citation windows [E017 caveat].

### 5.2 Metadata Coverage
Abstract coverage is 74.73% [E003], limiting text-based analyses. Document-type labels inherit source classification practices; all works carry the label “article” regardless of their actual type [E014 caveat]. Keyword frequencies depend on source metadata and synonym normalization [E020 caveat].

### 5.3 Analytical Caveats
The author string “AGB Poore” (5,808 publications) [E010] is a metadata artifact. Evidence shows this string appears as a co-author on “Food Security: The Challenge of Feeding 9 Billion People” alongside six other named authors [E028] and as the sole author of a *Lancet Planetary Health* paper on climate anxiety [E041]; it does not correspond to a single real researcher. This artifact has been excluded from substantive interpretation of author productivity and network centrality. Productivity rankings are descriptive and must not be treated as quality or causal influence [E009, E011, E013 caveats]. Network centrality describes this corpus and parameterization only [E022–E027 caveats]. The final year may be incomplete [E005 caveat]. Endpoint CAGR does not establish a monotonic trend [E006 caveat]. Bradford zoning excludes records without a source name [E019 caveat]. Abstracts cannot support claims requiring full-text inspection [E028–E051 caveats].

## 6. Conclusion
This bibliometric report provides a reproducible, evidence-grounded portrait of climate-change research from 1990 to 2021. The field has grown at a compound annual rate of 17.57% [E006, E007], is concentrated in a core of journals and institutions [E009, E013, E019], and exhibits a highly skewed citation distribution [E015, E017]. Thematic and collaborative structures, as revealed by network analyses [E022–E027], offer a baseline for future studies. All findings are bounded by the stated source, metadata, and methodological limitations.
