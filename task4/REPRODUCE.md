# Public-tested v9: Dice 0.86017

This bundle contains the five new model checkpoints, their source and configuration,
and frozen v3 validation/test probabilities. It reproduces the final v9 CSV.
See submission_v9_calibrated/README.md for measured comparison and recommendation.

Use the recorded Python 3.11 / PyTorch CUDA environment. Place the original `test`
folder beside this README, without renaming or re-encoding its images. Then run:

    python -m tools.reproduce_v8 --kind dual --output reproduced_model
    python -m tools.reproduce_v8 --kind fusion --fusion-recipe submission_v9_calibrated/inference_recipe.json --candidate-probabilities reproduced_model/test_predictions.npz --output reproduced_final

The commands load all five verified checkpoints, run D4 inference, fuse the frozen
v3 probabilities, execute the supplied encoder, and compare the final CSV SHA-256.

For a new-model training replay, add both original `train` and `test` folders:

    python -m vessel_lab.dual_domain_study
    python -m vessel_lab.dual_domain_fusion
    python -m vessel_lab.fusion_calibration

Input data and the 15 older v3 checkpoints are not duplicated here. The older
checkpoints remain in submission_v3/optimized_reproduce.zip in the original
workspace. To rebuild their probabilities as well, follow the final README.
Resumable new-model last.pt files remain in the original runs/dual_domain_v8 folder.
