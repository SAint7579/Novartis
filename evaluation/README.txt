Evaluation Scripts
==================

Quick Start:
-----------
python run_all_evals.py   # Runs all evaluations (skips existing results)

Individual Scripts:
------------------
visualize_all_models.py              - Generate latent space plots for all models
evaluate_perturbation_prediction.py  - Measure logFC recovery accuracy
evaluate_latent_retrieval.py         - Top-k retrieval metrics (full dataset)
evaluate_topk_fast.py                - Top-k retrieval (sampled, faster)
compare_vae_models.py                - Compare standard vs contrastive VAE
plot_latent.py                       - Plot single model (flexible CLI)
visualize_latent_space.py            - Plot single model (simple)

Outputs:
-------
Visualizations: ../latent_plots/
Metrics:        ../results/

