# Training and benchmark

Run \`python benchmark.py\` to compute Train, Validation and External metrics for the updated-Physics CNN-3M candidate and four selected 2M architectures. The historical 2M protocol used seeds 42, 123 and 2026, AdamW, MSE on Train-standardized SoH and Validation RMSE for selection. The updated CNN-3M used Validation MAE. External was opened after selection.

Run `python train.py --model CNN-3M` after the updated-Physics coverage gate. It requires a verified row for every frozen observation, uses fresh Train-only scalers, and stages one CNN-3M candidate directory. It uses overall Validation MAE for early stopping and seed selection. The historical RMSE-selected benchmark is unchanged. Training leaves the official pointer in place. It evaluates External after saving the Validation selection. Promotion retains the approved model if Validation MAE regresses.

A synthetic smoke check on CUDA exercised forward and backward passes, AdamW and a strict checkpoint round trip in a temporary directory. It produced no study metrics and did not change the checkpoint selection.

Train a single 2M architecture with python train.py --model TCN-2M --output-dir experiments/final2/tcn2m_new. Substitute ResNet-2M, DenseNet-2M, or ConvNeXt-2M to train another model. Each run initializes new weights, fits Train-only scalers, selects by Validation RMSE, and stages results separately from the approved checkpoint. CNN-3M retains its updated-Physics gate and Validation MAE selection.
