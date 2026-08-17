# A Bibliometric Analysis of Digital Twin Research (2012–2023): Publication Trends, Leading Actors, and Thematic Structure

## Structured Abstract
*Background:* Digital twin technology has rapidly emerged as a key enabler across industry, healthcare, and smart cities. This report provides a comprehensive bibliometric overview of the research landscape.
*Methods:* A corpus of 796 unique works was retrieved and deduplicated [E001, E002]. Performance analysis and science mapping were conducted using whole counting, Bradford’s law, and network analysis with Louvain community detection [E069–E072].
*Results:* The corpus spans 2012–2023, with a compound annual growth rate of 70.54% and a peak of 355 publications in 2023 [E004–E007]. The most prolific source is *IEEE Access* (n=24) [E008], the leading author is Kyle B. Enfield (n=22) [E010], and the top institution is Johns Hopkins University (n=12) [E012]. The most-cited document is “Digital Twin: Enabling Technologies, Challenges and Open Research” (2,623 citations) [E018]. Thematic mapping reveals a core structure dominated by “Computer science,” “Medicine,” and “Artificial intelligence” [E027].
*Conclusion:* The field exhibits rapid, accelerating growth concentrated in a core of journals and authors, with a strong interdisciplinary nexus between computer science and medicine. The findings are descriptive and bounded by the source and retrieval date.

## 1. Introduction
### 1.1 Background
Digital twin technology, often described as the integration of data between physical and virtual entities [E028], has become central to Industry 4.0 and precision medicine [E032]. The literature has expanded rapidly, necessitating a systematic mapping of its structure and trends.

### 1.2 Objectives
This report aims to (1) characterize the volume and growth trajectory of digital twin research [E004–E007], (2) identify the most productive journals, authors, and institutions [E008–E013], (3) analyze citation impact and document types [E014–E018], and (4) map the intellectual and social structure through network analysis [E022–E027].

### 1.3 Structure
The report follows the bibliometric workflow guidelines of Donthu et al. (2021) [E069], presenting reproducible methods, results with performance analysis and science mapping, a discussion of findings, and a transparent account of limitations.

## 2. Methods
### 2.1 Retrieval and Corpus Assembly
A bibliographic search was executed on a named source. Cursor pagination continued until the source returned no next cursor, confirming complete acquisition of 796 unique records matching the query [E002]. Deterministic exact deduplication yielded a final canonical corpus of 796 works [E001].

### 2.2 Data Quality and Coverage
Metadata completeness was assessed for every canonical work [E003]. Key fields showed high coverage: 100% for title, publication date, and cited-by count; 98.24% for DOI; and 86.18% for abstract [E003]. Missing data were noted for publisher (18.09%) and language (5.4%) [E003].

### 2.3 Performance Analysis
Annual output was whole-counted by publication year [E007]. Compound annual growth rate (CAGR) was calculated from the first to the last non-zero endpoint [E006]. Top sources, authors, and institutions were ranked by whole-counted publications [E009, E011, E013]. Citation impact was described using source-supplied cited-by counts, including mean, median, and zero-citation share [E015–E017]. Bradford’s law of scattering was applied to sources with known names; records without a source name were excluded from Bradford zoning [E019].

### 2.4 Science Mapping
Network analyses were performed using a candidate-first association-strength method [E022–E027]. Candidate nodes were selected by minimum occurrence thresholds, edges were normalized by association strength, and communities were detected using the Louvain algorithm [E072]. Networks were visualized using VOS-style layouts or ForceAtlas2, with edge reduction via maximum-spanning forest plus strongest edges per node [E022–E027, E071]. Networks constructed include bibliographic coupling, citation, coauthorship, cocitation, institution collaboration, and keyword co-occurrence [E022–E027].

## 3. Results
### 3.1 Publication Growth and Corpus Overview
The corpus spans 2012 to 2023 [E004]. Annual output grew from 1 publication in 2012 to 355 in 2023, representing a compound annual growth rate of 70.54% from the first to the last non-zero endpoint; this endpoint measure does not establish a monotonic trend [E005–E007]. The largest absolute year-over-year increases occurred in 2022 (+120) and 2023 (+118), though the 2023 count may be incomplete depending on the retrieval date [E005, E007]. (Figure 1 [E052])

### 3.2 Document Types and Source Landscape
All 796 works are classified as “article” in the source metadata; this uniform label reflects source classification practices and may not distinguish reviews or other types [E014]. (Figure 5 [E056]) The corpus is distributed across 468 sources, with a core Zone 1 of 37 journals accounting for 262 publications (approximately one-third of the corpus, noting that records without source names were excluded from zoning) [E019, E059]. *IEEE Access* is the most prolific source with 24 publications, followed by *International Journal of Advanced Research and Innovations* (19) and *Zenodo* (16) [E008, E009]. Zenodo is a general-purpose open repository; its inclusion among top sources reflects preprint deposition rather than peer-reviewed journal publication [E009]. (Figure 2 [E053])

### 3.3 Leading Authors and Institutions
Kyle B. Enfield is the most productive author with 22 publications, followed by Paramesh Shamanna (11) and Zhihan Lv (10) [E010, E011]. Productivity counts are descriptive and do not measure research quality or influence [E011, E013]. (Figure 3 [E054]) Johns Hopkins University leads institutional output with 12 publications, followed by Centre National de la Recherche Scientifique and Uppsala University (11 each) [E012, E013]. (Figure 4 [E055])

### 3.4 Citation Impact
The mean citation count is 53.20 and the median is 16.00, indicating a right-skewed distribution [E015, E017]. Citation counts are source- and time-dependent; recent works have had less time to accumulate citations [E015, E017]. The maximum citation count is 2,623 [E017]. A total of 126 works (15.83%) have zero source-reported citations [E016]. (Figure 6 [E057]) The most-cited document is “Digital Twin: Enabling Technologies, Challenges and Open Research” (Fuller et al., 2020, *IEEE Access*) with 2,623 citations [E018, E028]. (Figure 7 [E058])

### 3.5 Thematic Structure and Trends
The most frequent keywords are “Computer science” (n=607), “Medicine” (n=350), and “Artificial intelligence” (n=252) [E020]. Keyword frequencies reflect source-supplied or algorithmically assigned terms and depend on synonym normalization [E020]. Temporal analysis shows that “Computer science” has been present since 2012. “Artificial intelligence” appeared sporadically from 2014 and accelerated sharply after 2019, while “Health care” emerged in 2018 and surged thereafter [E020]. (Figure 9 [E060])

### 3.6 Network Analysis
The keyword co-occurrence network (72 nodes, 188 edges, 8 clusters) reveals a central cluster around “Computer science” and “Artificial intelligence,” linked to a distinct “Medicine” and “Health care” cluster [E027]. (Figures 12, 16, 17 [E063, E067, E068]) The coauthorship network (59 nodes, 128 edges, 10 clusters) reveals a densely connected research group including Paramesh Shamanna, Terrence Poon, and Mohamed Thajudeen [E024]. (Figure 10 [E061]) The institution collaboration network (72 nodes, 153 edges, 10 clusters) shows Maastricht University and Stanford University as highly connected nodes [E026]. (Figure 11 [E062]) The bibliographic coupling network (72 nodes, 230 edges, 8 clusters) shows “Impactful Digital Twin in the Healthcare Revolution” as a highly connected node [E022]. (Figure 15 [E066]) The citation network is sparse (5 nodes, 4 edges), reflecting the high minimum-occurrence threshold (15) required for candidate selection [E023]. (Figure 14 [E065]) The cocitation network (72 nodes, 203 edges, 7 clusters) identifies foundational references [E025]. (Figure 13 [E064])

## 4. Discussion
### 4.1 Interpretation of Growth
The 70.54% CAGR and the concentration of output in 2022–2023 suggest the field is in a phase of rapid, accelerating expansion, likely driven by the convergence of IoT, AI, and healthcare applications [E006, E007, E028, E030]. This observation is a descriptive trend, not a causal claim.

### 4.2 Concentration and Core Actors
Bradford zoning confirms a strong core-periphery structure in publication sources [E019]. The dominance of a single author (Kyle B. Enfield) and institution (Johns Hopkins University) indicates concentrated productivity, though this must not be interpreted as a measure of research quality [E010, E012, E011, E013].

### 4.3 Thematic Integration
The keyword co-occurrence and three-field plots reveal a deep interdisciplinary integration between computer science methodologies (AI, data science) and medical applications (health care, internal medicine) [E020, E021, E027]. The overlay visualization indicates that health-related keywords tend to have more recent average publication years, consistent with growing attention to healthcare applications [E067].

### 4.4 Citation Patterns
The highly skewed citation distribution, with a mean much higher than the median, is typical of scientific fields where a few seminal works attract disproportionate attention [E015, E017]. The most-cited works include titles suggesting they are surveys or enabling-technology overviews (e.g., “A Survey on Digital Twin…”, “Digital Twins: A Survey…”), though document-type labels in the source metadata do not distinguish reviews from articles [E014, E018].

## 5. Limitations
### 5.1 Source and Retrieval Bias
All findings are dependent on the single bibliographic source queried and the specific retrieval date. Citation counts are source- and time-dependent [E015, E017].

### 5.2 Citation Window
Recent works, particularly those from 2023, have had less time to accumulate citations, which biases citation-based metrics against newer publications [E016, E017].

### 5.3 Metadata Quality
While core fields are complete, 13.82% of abstracts and 18.09% of publisher names are missing, which may affect keyword and source analyses [E003]. The publisher field missingness may affect any source-level analyses that rely on publisher metadata, though the current analyses use source_id, which is 100% complete [E003]. Document-type labels inherit the source’s classification, which resulted in a uniform “article” categorization [E014].

### 5.4 Network Parameterization
Network structures are sensitive to the chosen thresholds (minimum occurrence, edge reduction method) and normalization (association strength). They describe the constructed corpus and do not establish causal or substantive importance [E022–E027].

### 5.5 Content Depth
All thematic interpretations are based on titles, abstracts, and keywords. No full-text analysis was performed, so claims about detailed methodologies or findings within publications cannot be supported [E028–E051].

## 6. Conclusion
This bibliometric analysis maps a rapidly expanding research field at the intersection of digital twin technology, computer science, and medicine. The corpus of 796 articles exhibits a 70.54% CAGR, peaking in 2023 [E001, E006, E005]. The intellectual structure is concentrated in a core of journals and authors, with *IEEE Access* and Kyle B. Enfield as the most prolific source and author, respectively [E008, E010]. Thematic mapping confirms a strong and growing integration of AI and data science methodologies into healthcare applications [E027]. These findings provide a descriptive baseline for researchers and stakeholders, with all interpretations bounded by the stated source, temporal, and methodological limitations.
