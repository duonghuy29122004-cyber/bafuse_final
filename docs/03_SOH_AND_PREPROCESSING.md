# SoH and preprocessing

SoH in percent is 100 times the ratio of RPT Step 4 discharge capacity to the same cell's verified beginning-of-life Step 4 capacity. Capacity uses trapezoidal integration of absolute current over native TestTime before resampling. Parts 1?2 use their verified Part 1 BoL reference, and Part 3 uses its same-cell Part 3 reference. Part 3 labels are provisional reconstructed capacity-retention references.

Discharge and Charge are measured Step 4 and Step 10 voltages, each interpolated to 128 points over normalized row progress. Input and target scalers use column-wise population standard deviation fitted on the 99 Train observations. Archived predictions are converted to percent using the saved target inverse transform.

See the [dataset loader](../src/bafuse/hybrid/final2_dataset.py) and [saved scalers](../experiments/final2/training_6_models/run_001/normalization_parameters.json).
