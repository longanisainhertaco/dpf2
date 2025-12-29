# Repository Scaffold for DPF-NextGen

```
/ (project root)
├── README.md                     # Architecture overview and rationale for ECSIM + hybrid-PIC
├── docs/
│   ├── gap_analysis_dpf_nextgen.md
│   └── repository_structure.md
├── src/
│   ├── hybrid_core/
│   │   ├── __init__.py
│   │   ├── README.md             # Separation of fluid vs kinetic handlers
│   │   ├── fluid/                # Electron/fluid closures, resistivity models
│   │   │   └── __init__.py
│   │   └── kinetic/              # Ion pushers, collision operators, fusion sampling
│   │       └── __init__.py
│   ├── cad_interface/
│   │   ├── __init__.py
│   │   └── README.md             # OpenCASCADE bindings and CAD ingest utilities
│   └── ... (existing modules)
├── benchmarks/
│   └── pirt/                     # Phenomena Identification and Ranking Table cases
│       └── README.md
├── tests/
│   └── mms/                      # Method of Manufactured Solutions verification
│       └── README.md
└── validation/                   # Legacy validation harnesses (existing)
```

This scaffold highlights the explicit separation of fluid and kinetic components, CAD-native ingest, PIRT validation, and MMS-based verification.
