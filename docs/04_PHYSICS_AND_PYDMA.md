# Physics and PyDMA

The approved updated Physics schema and readiness are below.

| Position | Feature | Definition and status |
|---|---|---|
| 1 | nominal_cycle_age | 48 ? verified Guide RPT ordinal, in nominal cycles. Preserve the continued ordinal across Parts 1?2 and use the documented Part 3 cadence. |
| 2 | voltage_droop_50_soc | Final measured rest voltage minus linearly interpolated discharge-pulse voltage at 10 s, in V. Use Step 21 ? 22 for Parts 1?2 and Step 19 ? 20 for Part 3. |
| 3 | discharge_voltage_slope_20_80 | Ordinary least-squares slope of Step 4 voltage against normalized row progress on [0.20, 0.80], in V per unit progress. |
| 4 | discharge_voltage_derivative_mad_10_90 | 1.4826 times median absolute deviation of central-difference voltage derivatives on [0.10, 0.90], in V per unit progress. |
| 5 | discharge_voltage_segment_contrast_10_30_vs_70_90 | Mean Step 4 voltage on [0.10, 0.30] minus mean on [0.70, 0.90], in V. |

The last three use the stored 128-point Step 4 definitions and the grid i/127. All 213 frozen observations have those historical descriptors. The age check matched 203 observations to Guide-based records in the unified manifest. Ten G10-Cell2 observations use the owner-confirmed frozen mapping and documented Part 3 cadence. Age is a nominal count of scheduled aging cycles, distinct from the raw test-procedure counter, equivalent full cycles, and elapsed time.

The historical CNN-3M uses mean absolute current and discharge current variability for positions 1 and 2. It requires the historical scaler. Updated CNN-3M training is gated on scientific verification.

Oxford sets nominal SoC using measured capacity and analyzes the first 10 seconds of the approximately ?1.5 A discharge pulse. Representative raw pulses last about 36 seconds. BaFuse uses the final measured zero-current rest voltage as its operational pre-pulse baseline. The audit anchors to the frozen same-RPT Step 10 sequence and rejects missing, irregular, or abnormal measurements. It does not use the SoH label or subsequent RPTs. See [Raj's Oxford thesis, ?2.3 and Appendix A](https://ora.ox.ac.uk/objects/uuid:d975c41e-4406-42f3-aef9-beb54c60c11f/files/d0g354f56b) and the [coverage extractor](../src/bafuse/features/physics_coverage.py).

PyDMA reports independent Charge-derived LLI, anode LAM and cathode LAM. The saved-reader view preserves quality status, fit RMSE, BoL, and source provenance. Missing results remain unavailable.

The coverage check accepts rest current within 0.01 A of zero, discharge current within 0.05 A of −1.5 A, and at most 1.5 s between the samples bracketing 10 s. It flags droop outside 0–1 V and records the exact source row spans. These quality checks do not clip or impute any feature. A frozen observation that fails a check blocks training until its evidence is resolved.

Sources: [Oxford Part 1](https://ora.ox.ac.uk/objects/uuid:de62b5d2-6154-426d-bcbb-30253ddb7d1e), [Part 2](https://ora.ox.ac.uk/objects/uuid:be3d304e-51fd-4b37-a818-b6fa1ac2ba9d), [Part 3](https://ora.ox.ac.uk/objects/uuid:78f66fa8-deb9-468a-86f3-63983a7391a9), [descriptor code](../src/bafuse/hybrid/p5_electrical_features.py), and [Charge reader](../app/model_runtime.py).
