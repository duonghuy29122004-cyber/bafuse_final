# BaFuse Final2

BaFuse estimates battery state of health at the same reference performance test as its inputs. The five approved architectures are CNN-3M, TCN-2M, ResNet-2M, DenseNet-2M and ConvNeXt-2M. All use measured Discharge and Charge. CNN-3M also uses Physics.

## Research evidence

[Selected metrics and checkpoint provenance](release/selected_research_artifacts.json), [training protocol](release/training_protocol.json), and [feature schemas](release/feature_schemas.json) describe the frozen cohort and model revisions. The cell-disjoint split contains 99 Train observations across 9 cells, 40 Validation across 3, and 74 External across 12. External labels are provisional reconstructed capacity-retention references. MAE and RMSE use SoH percentage points.

The approved CNN-3M runtime uses its historical current-based Physics schema. The updated-Physics CNN-3M is a separate candidate with its own scaler and verified results. Its Validation MAE did not clear the replacement gate. It remains outside the approved runtime weights bundle.

## Reproduce locally

1. Create the Python environment using requirements-lock-cuda-cu126.txt, app/requirements_final2_streamlit.txt and rag/requirements_rag.txt as appropriate for the local runtime. Run CLI help before adding data.
2. Obtain Oxford Parts 1?3 from the linked archives under their terms. Use the mapping, frozen-v2 and Physics coverage procedures described in [Dataset and mapping](docs/02_DATASET_AND_MAPPING.md), [SoH and preprocessing](docs/03_SOH_AND_PREPROCESSING.md) and [Physics and PyDMA](docs/04_PHYSICS_AND_PYDMA.md). The public checkout does not supply the original curated row-level artifacts. Exact score reproduction requires the hash-matched frozen artifacts and checkpoint bundle from the project owner.
3. Restore separately authorized artifacts to the run paths expected by the manifests. The proposed approved weights bundle contains five verified selected checkpoints and the corresponding shared Train-only scalers and configuration. Verify digests before loading. The candidate benchmark additionally needs its own separate candidate assets and coverage. See [Training and benchmark](docs/06_TRAINING_AND_BENCHMARK.md).
4. Run the commands below from the repository root once the required local artifacts are present.

    python -m pytest tests
    python benchmark.py
    python evaluate.py --split validation
    python train.py --model CNN-3M --output-dir experiments/final2/my_candidate
    python demo.py

A full test run or inference needs the referenced local evidence. From this checkout alone, the CLI help and pure-code tests can be run. CSV labels, measured waveforms, per-RPT PyDMA results and half-cell references are held separately. The RAG client is included. The local document text/index and backend model assets need separate rights and deployment.

## Local artifact paths

Use experiments/final2/training_6_models as STUDY below. Map each pending bundle checkpoint back to its recorded filename under STUDY/run_001:

- checkpoints/CNN-3M/best.pt ? STUDY/run_001/checkpoints/CNN-3M/seed_2026_best.pt
- checkpoints/TCN-2M/best.pt ? STUDY/run_001/checkpoints/TCN-2M/seed_123_best.pt
- checkpoints/ResNet-2M/best.pt ? STUDY/run_001/checkpoints/ResNet-2M/seed_2026_best.pt
- checkpoints/DenseNet-2M/best.pt ? STUDY/run_001/checkpoints/DenseNet-2M/seed_42_best.pt
- checkpoints/ConvNeXt-2M/best.pt ? STUDY/run_001/checkpoints/ConvNeXt-2M/seed_123_best.pt

Place the matching normalization_parameters.json under STUDY/run_001, and training_config_v2.json at STUDY. The original run_001/checkpoint_manifest.json, validation_selection.json and selected_validation_metrics.csv are also needed by the frozen loader. Preserve original bytes and paths. The frozen dataset authority and its NPZ and manifest files belong under STUDY/frozen_dataset and STUDY. External benchmark reproduction also uses the original external_evaluation manifest and metrics. The updated-CNN candidate uses its own STUDY/updated_physics_candidate directory and data/processed/physics/coverage.csv. Neither bundle includes these local dataset inputs.

## Source and distribution

Data source: Oxford Research Archive [Part 1](https://ora.ox.ac.uk/objects/uuid:de62b5d2-6154-426d-bcbb-30253ddb7d1e), [Part 2](https://ora.ox.ac.uk/objects/uuid:be3d304e-51fd-4b37-a818-b6fa1ac2ba9d), and [Part 3](https://ora.ox.ac.uk/objects/uuid:78f66fa8-deb9-468a-86f3-63983a7391a9). Review the [ODbL](https://opendatacommons.org/licenses/odbl/1-0/) and original file notices before redistributing data or derived databases. Data does not inherit a source-code license. Distribution of model weights and scalers remains subject to the project's rights and approval.
