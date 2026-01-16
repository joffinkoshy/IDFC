# IDFC Document Processing Pipeline

## Architecture Overview

This project implements a comprehensive document processing pipeline for IDFC with the following modular architecture:

```
IDFC/
│
├── data/
│   ├── train/                    # Training images
│   │   ├── img_001.jpg
│   │   ├── img_002.jpg
│   │   └── ...
│   └── masters/                  # Master data
│       ├── dealer_master.csv
│       └── asset_master.csv
│
├── src/
│   ├── preprocessing/            # Image preprocessing
│   │   ├── image_loader.py
│   │   ├── image_normalizer.py
│   │   ├── orientation.py
│   │   ├── stamp_detector.py
│   │   ├── signature_detector.py
│   │   └── ocr_engine.py
│   │
│   ├── candidates/               # Candidate extraction
│   │   ├── base.py               # Base Candidate class
│   │   ├── dealer.py
│   │   ├── model.py
│   │   ├── hp.py
│   │   ├── cost.py
│   │   └── visual.py
│   │
│   ├── reasoning/                # Reasoning and validation
│   │   ├── dealer_reasoner.py
│   │   ├── model_reasoner.py
│   │   ├── hp_reasoner.py
│   │   ├── cost_reasoner.py
│   │   └── document_gate.py
│   │
│   ├── postprocessing/           # Post-processing
│   │   ├── confidence.py
│   │   ├── json_formatter.py
│   │   └── failure_report.py
│   │
│   ├── utils/                    # Utilities
│   │   ├── geometry.py
│   │   ├── fuzzy_match.py
│   │   ├── text_normalize.py
│   │   └── constants.py
│   │
│   └── pipeline.py               # Main pipeline orchestrator
│
├── experiments/                  # Analysis and experimentation
│   └── eda/
│       ├── candidate_counts.ipynb
│       ├── failure_modes.ipynb
│       └── confidence_vs_dla.ipynb
│
├── outputs/                      # Output files
│   └── predictions/
│
├── run_pipeline.py               # Main entrypoint
├── requirements.txt
└── README.md
```

## Pipeline Flow

1. **Preprocessing**: Image loading, normalization, orientation correction, stamp/signature detection, OCR
2. **Candidate Extraction**: Extract candidates for dealer, model, horsepower, cost, and visual elements
3. **Reasoning**: Validate and reason about extracted candidates using domain logic
4. **Post-processing**: Compute confidence scores, format JSON output, generate failure reports
5. **Output**: Save processed results to outputs/predictions/

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
python run_pipeline.py input_image.jpg output.json
```

## Key Features

- Modular design for easy maintenance and extension
- Comprehensive master data integration
- Advanced candidate extraction and reasoning
- Robust confidence scoring and failure reporting
- Built-in experimentation and analysis tools
- End-to-end document processing pipeline
