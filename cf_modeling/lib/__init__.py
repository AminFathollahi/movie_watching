"""
cf_modeling/lib
================
Internal library for the CF modeling pipeline.

Modules
-------
config_builder   Build a temporary YAML config for vicsompy's MssCf.
subject_adapter  Minimal mock subject object compatible with MssCf.
data_adapter     Convert CIFTI grayordinate ↔ full-sphere surface space.
cf_model         CfModel — MssCf subclass for grayordinate targets, no splicing.
"""
