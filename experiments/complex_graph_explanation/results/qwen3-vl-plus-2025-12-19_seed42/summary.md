# Complex Graph Experiment Summary

| Condition | Task | N | Strict accuracy | Shortest path | Distance consistency | Bridge | Cross-edge exact | Direct-edge | Edge hallucination |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| vlm | ALL | 24 | 0.417 | 0.529 | 0.824 | 0.333 | 0.400 | 1.000 | 0.123 |
| vlm | bridge_node | 6 | 0.333 | NA | NA | 0.333 | NA | NA | 0.000 |
| vlm | cross_community_path | 6 | 0.000 | 0.400 | 1.000 | NA | 0.400 | NA | 0.250 |
| vlm | path_trace | 6 | 0.333 | 0.333 | 0.833 | NA | NA | NA | 0.286 |
| vlm | unsupported_edge | 6 | 1.000 | 0.833 | 0.667 | NA | NA | 1.000 | 0.000 |
| vlm_flat_kg | ALL | 24 | 0.792 | 1.000 | 0.944 | 0.667 | 0.667 | 1.000 | 0.000 |
| vlm_flat_kg | bridge_node | 6 | 0.667 | NA | NA | 0.667 | NA | NA | 0.000 |
| vlm_flat_kg | cross_community_path | 6 | 0.667 | 1.000 | 1.000 | NA | 0.667 | NA | 0.000 |
| vlm_flat_kg | path_trace | 6 | 0.833 | 1.000 | 0.833 | NA | NA | NA | 0.000 |
| vlm_flat_kg | unsupported_edge | 6 | 1.000 | 1.000 | 1.000 | NA | NA | 1.000 | 0.000 |
| vlm_graph_rag | ALL | 24 | 0.833 | 0.889 | 1.000 | 1.000 | 0.667 | 1.000 | 0.031 |
| vlm_graph_rag | bridge_node | 6 | 1.000 | NA | NA | 1.000 | NA | NA | 0.000 |
| vlm_graph_rag | cross_community_path | 6 | 0.333 | 0.667 | 1.000 | NA | 0.667 | NA | 0.125 |
| vlm_graph_rag | path_trace | 6 | 1.000 | 1.000 | 1.000 | NA | NA | NA | 0.000 |
| vlm_graph_rag | unsupported_edge | 6 | 1.000 | 1.000 | 1.000 | NA | NA | 1.000 | 0.000 |