# Model architectures

CNN-3M is the original baseline. Two CNN1D encoders process Discharge and Charge, and an MLP processes Physics. Fusion combines these outputs for scalar regression. TCN-2M, ResNet-2M, DenseNet-2M and ConvNeXt-2M compare architectures with the same two measured inputs.

The comparison uses the updated-Physics CNN-3M candidate and the four existing 2M checkpoints. The app continues to use its approved CNN-3M weights with their matching historical Physics schema because the candidate did not clear the Validation promotion gate.

See the [model builders](../src/bafuse/hybrid/final2_models.py) and [checkpoint manifest](../experiments/final2/training_6_models/run_001/checkpoint_manifest.json).

The runtime obtains its [TCN-2M builder](../src/bafuse/models/tcn_2m.py) from the source package. The original experiment copies are preserved at the paths recorded in their historical SHA-256 manifests.
