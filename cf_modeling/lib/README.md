# CF Modeling Lib — Connective Field Wrappers

Provides wrappers around vicsompy core classes to support modular ROI pairs, grayordinate targeting, and GPU acceleration without modifying vicsompy source code.

## Modules

| Module | Purpose |
|--------|---------|
| `cf_model.py` | CfModel wrapper extending vicsompy.modeling.MssCf; grayordinate targets (59k vertices), no splicing, GPU patches |
| `data_adapter.py` | Bidirectional conversion between CIFTI grayordinates (59k) and sphere space (118k); BrainModelAxis-aware |
| `config_builder.py` | Build temporary YAML for MssCf initialization with custom ROI pairs and LBOE counts |
| `subject_adapter.py` | Minimal mock subject object satisfying vicsompy's subject interface for direct import paths |
