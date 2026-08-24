# Independent audit report

- Benchmark samples: 24
- Result records: 72
- Records with one or more findings: 13
- `error` means a deterministic inconsistency; `review` is a conservative free-text flag and must be read manually.

## Accuracy after additional deterministic checks

| Condition | Original correct | Original accuracy | Conservative correct | Conservative accuracy |
|---|---:|---:|---:|---:|
| vlm | 10/24 | 0.417 | 10/24 | 0.417 |
| vlm_flat_kg | 19/24 | 0.792 | 19/24 | 0.792 |
| vlm_graph_rag | 20/24 | 0.833 | 20/24 | 0.833 |

## Finding counts

| Kind | Count |
|---|---:|
| cross_edge_field_inconsistent | 7 |
| self_correction_language | 4 |
| unsupported_free_text_edge | 7 |

## Record-level findings

### CG001 / vlm / repeat 1

- **review / unsupported_free_text_edge**: N2-N4; also_structured_claim=True; in: N1与N2相连，N2与N4相连，构成长度为2的最短路径

### CG002 / vlm / repeat 1

- **info / cross_edge_field_inconsistent**: expected=[('N1', 'N4'), ('N3', 'N4')]; reported=[('N1', 'N4'), ('N3', 'N4'), ('N3', 'N8')]

### CG002 / vlm_flat_kg / repeat 1

- **info / cross_edge_field_inconsistent**: expected=[('N1', 'N4'), ('N3', 'N4')]; reported=[('N1', 'N4'), ('N3', 'N4'), ('N3', 'N8')]

### CG006 / vlm / repeat 1

- **review / unsupported_free_text_edge**: N1-N3; also_structured_claim=True; in: 径上无跨社区边。但原图中N3-N1、N1-N4均为C1内边，因
- **review / self_correction_language**: 应为, 重新
- **info / cross_edge_field_inconsistent**: expected=[]; reported=[('N1', 'N3'), ('N1', 'N4')]

### CG006 / vlm_graph_rag / repeat 1

- **review / unsupported_free_text_edge**: N2-N4; also_structured_claim=True; in: 。证据中存在N3-N2和N2-N4边。

### CG010 / vlm_graph_rag / repeat 1

- **review / self_correction_language**: 修正, 应为
- **info / cross_edge_field_inconsistent**: expected=[('N3', 'N6'), ('N3', 'N7')]; reported=[('N3', 'N6'), ('N3', 'N7'), ('N5', 'N6')]

### CG013 / vlm / repeat 1

- **review / unsupported_free_text_edge**: N3-N4; also_structured_claim=True; in: N5与N3相连，N3与N4相连，构成长度为2的最短路径

### CG014 / vlm / repeat 1

- **review / unsupported_free_text_edge**: N5-N6; also_structured_claim=True; in: 属C1，N8属C6；路径N6-N5-N8长度为2，两段边均跨社

### CG014 / vlm_flat_kg / repeat 1

- **info / cross_edge_field_inconsistent**: expected=[('N2', 'N6'), ('N5', 'N8')]; reported=[('N2', 'N5'), ('N2', 'N6'), ('N5', 'N8')]

### CG014 / vlm_graph_rag / repeat 1

- **review / unsupported_free_text_edge**: N5-N6; also_structured_claim=True; in: 度为2，是最短路径。图中N6与N5、N5与N8均有直接边连接。

### CG017 / vlm / repeat 1

- **review / unsupported_free_text_edge**: N1-N6; also_structured_claim=True; in: N6与N1相连，N1与N7相连，构成长

### CG018 / vlm / repeat 1

- **review / self_correction_language**: 修正, 应为, 重新
- **info / cross_edge_field_inconsistent**: expected=[('N1', 'N7'), ('N3', 'N7')]; reported=[('N1', 'N7'), ('N3', 'N5'), ('N3', 'N7')]

### CG022 / vlm_graph_rag / repeat 1

- **review / self_correction_language**: 修正, 错误, 应为, 重新
- **info / cross_edge_field_inconsistent**: expected=[('N2', 'N6'), ('N3', 'N8')]; reported=[('N2', 'N6'), ('N3', 'N8'), ('N6', 'N8')]
