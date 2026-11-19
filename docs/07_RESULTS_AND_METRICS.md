# Results and metrics

Run \`python benchmark.py\` to recompute all nine metrics per architecture and print the result. Use \`python evaluate.py --split external\` for only External. Both commands are read-only. They use the updated CNN-3M candidate and the four selected 2M models with their matching scalers.

CNN-3M is the original Discharge + Charge + Physics architecture. TCN-2M, ResNet-2M, DenseNet-2M and ConvNeXt-2M share the measured Discharge + Charge contract.

The updated-Physics CNN-3M candidate selected seed 123 and epoch 9 by Validation MAE. Its CUDA benchmark is below. The candidate was retained because Validation MAE regressed relative to the approved runtime model on the same labels. The feature revisions remain distinct.

| Split | Observations | Cells | MAE | RMSE | R? |
|---|---:|---:|---:|---:|---:|
| Train | 99 | 9 | 0.8107 | 0.9915 | 0.9685 |
| Validation | 40 | 3 | 0.3695 | 0.4594 | 0.9933 |
| External | 74 | 12 | 1.0459 | 1.2707 | 0.9071 |

Metrics are pooled by observation. MAE and RMSE are in SoH percentage points. External labels are provisional reconstructed capacity-retention references.

See the [candidate selection and metrics](../experiments/final2/training_6_models/updated_physics_candidate/metrics.json).
