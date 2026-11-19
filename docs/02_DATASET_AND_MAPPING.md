# Dataset and mapping

The authority is [the Final2 v2 freeze manifest](../experiments/final2/training_6_models/frozen_dataset_manifest_v2.json), dataset ID bafuse-final2-p5-g10-guide-verified-v2. The [observation manifest](../experiments/final2/training_6_models/frozen_dataset/observation_manifest.csv) preserves Part, group, physical cell, RPT, raw path, Step 4/10 blocks and hashes, status, and beginning-of-life provenance.

| Partition | Accepted RPTs | Active cells |
|---|---:|---:|
| Train | 99 | 9 |
| Validation | 40 | 3 |
| External Part 3 | 74 | 12 |

Development uses Part 1 and Part 2 Groups 1?4. Split membership follows the physical cell across parts. REVIEW and REJECTED records are outside the frozen cohort. G10-Cell2 RPT10 has a source-specific Step 10 zero-current-row exclusion in the freeze manifest. There is no internal test split.
