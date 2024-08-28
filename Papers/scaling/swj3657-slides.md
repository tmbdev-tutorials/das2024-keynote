# SemPubFlow: A Novel Scientific Publishing Workflow Using Knowledge Graphs, Wikidata, and LLMs

- Wolfgang Fahl, Tim Holzheim, Christoph Lange, Stefan Decker
- RWTH Aachen University, Fraunhofer FIT
- Date of Presentation

# Introduction

- Problem: Traditional publishing workflows involve redundant and manual curation
- Motivation: Enhance data quality and availability using FAIR principles
- Scope: Focus on CEUR-WS platform and its transition to a semantified workflow

# Background/Related Work

- CEUR-WS platform has been pivotal since 1995
- Challenges with traditional manual curation and data quality
- Previous attempts at semantification lacked consistency and durability

# Contributions

- Modernizing legacy pipeline with SemPubFlow approach
- Leveraging LLMs and Wikidata for metadata curation
- Introduction of Single Source of Truth (SSoT) and Single Point of Truth (SPoT)

# Objective

- Create a consistent and FAIR knowledge graph for CEUR-WS
- Shift data curation responsibility to stakeholders early in the event lifecycle
- Ensure continuity and integrity of scholarly communication

# Methodology Overview

- Metadata-first approach to publishing
- Integration of LLMs for metadata extraction
- Use of Wikidata for storing and linking metadata

# Datasets

- CEUR-WS volumes and proceedings
- Metadata from event series, events, papers, editors, authors, and institutions
- External sources: DBLP, k10plus, Wikidata

# Model Details

- Use of Wikidata as a knowledge graph
- Tokenization and Named Entity Recognition (NER)
- Disambiguation using event signatures

# Experiments

- Extraction and reconciliation of metadata
- Matching against DBLP and k10plus records
- Use of LLMs for homepage metadata extraction

# Results

- Improved indexing coverage for CEUR-WS volumes
- Higher timeliness of metadata availability
- Enhanced metadata quality and accuracy

# Performance Comparisons with Prior Work

- 100% indexing coverage compared to 69% for k10plus and 76% for DBLP
- Immediate metadata availability versus weeks/months delay in traditional systems

# Visualizations

- ((Figure showing locations of all CEUR-WS proceedings events))

# Discussion

- Key insights: Early curation and public availability improve metadata quality
- Interpretation: Metadata-first approach aligns with FAIR principles
- Limitations: Need for mass disambiguation and consistent updating

# Conclusion

- Summary: Successful start of CEUR-WS semantification with Wikidata
- Impact: Potential to improve scholarly publishing workflows
- Future Work: Extend SemPubFlow to other publishing outlets, enhance automation

# References

- M.D. Wilkinson et al., The FAIR Guiding Principles for scientific data management and stewardship, Scientific Data 3(1) (2016)
- D. Vrandečić and M. Krötzsch, Wikidata, Communications of the ACM 57(10) (2014)

# Acknowledgements

- Supported by Deutsche Forschungsgemeinschaft (DFG)
- Contributions from Jakob Voß, Thomas Hoeren, Jonas Kuiter

# Q&A

- Invitation for Questions
