evaluation

python evaluate_per_generator.py `
  --checkpoint checkpoints/best_model.pt --label "Cross-gen model" `
  --checkpoint checkpoints/pergen_split_epoch_005.pt --label "Seen-gen 80/20 (ep5)" `
  --generators_root per-gen-dataset `
  --clip_cache_dir datasets_eq/clip_cache

python evaluate_per_generator.py `
   --checkpoint checkpoints/best_model.pt --label "Cross-gen model" `
   --checkpoint checkpoints/pergen_split_best_model.pt --label "Seen-gen 80/20" `
   --generators_root per-gen-dataset-test `
   --clip_cache_dir datasets_eq/clip_cache_test



python evaluate_per_generator.py `
  --checkpoint checkpoints/best_model.pt --label "Cross-gen model" `
  --checkpoint checkpoints/pergen_split_best_model.pt --label "Seen-gen 80/20 model" `
  --generators_root "per-gen-dataset-test" `
  --clip_cache_dir datasets_eq/clip_cache_test


# Cross-gen model (apples-to-apples with Man & Cho)
python evaluate_per_generator.py `
    --checkpoint checkpoints/best_model.pt `
    --generators_root "per-gen-dataset-test" `
    --clip_cache_dir datasets_eq/clip_cache_test

# Seen-gen model (trained on same generator types)
python evaluate_per_generator.py `
    --checkpoint checkpoints/pergen_split_best_model.pt `
    --generators_root "per-gen-dataset-test" `
    --clip_cache_dir datasets_eq/clip_cache_test



# Variation 1 — DFDC only (fastest, ~2-3 hours, fixes the near-random problem):

python finetune.py `
    --checkpoint checkpoints/best_model.pt `
    --include_generators DFDC `
    --epochs 20 `
    --patience 7 `
    --save checkpoints/finetuned_dfdc_only.pt

# Variation 2 — All generators equally (general improvement, no oversampling):

python finetune.py `
    --checkpoint checkpoints/best_model.pt `
    --epochs 20 `
    --patience 7 `
    --save checkpoints/finetuned_all_gens.pt

# Variation 3 — All generators, underperformers boosted (the one from before):

python finetune.py `
    --checkpoint checkpoints/best_model.pt `
    --target_generators DFDC Deepfake StyleGAN StyleGAN2 BigGAN CycleGAN StarGAN GauGAN "DALL-E" `
    --target_weight 3 `
    --epochs 20 `
    --patience 7 `
    --save checkpoints/finetuned_boosted.pt

# Example: cancelled after epoch 10, resume from there
python finetune.py `
    --checkpoint checkpoints/best_model.pt `
    --include_generators DFDC `
    --resume checkpoints/finetune_epoch_010.pt `
    --save checkpoints/finetuned_dfdc_only.pt



# heatmap
python generate_sample_heatmaps.py --checkpoint checkpoints/pergen_split_best_model.pt