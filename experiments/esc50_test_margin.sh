GPUID=0

# for lr in 1e-2 1e-3 1e-4
# do
#     python train_test.py --root /workspace/joel/CNNAudioClassification/data --out_dir checkpoints/test-$lr/ \
#         --dataset_name esc50 --num_workers 8 --batch_size 32 \
#         --seed 42 --soft_epsilon 0.2 --lr $lr --epochs 300 \
#         --n_time_masks 3 --time_mask_param 40 --n_freq_masks 3 --freq_mask_param 15 \
#         --gpu_id $GPUID
# done
for margin in 0 1 2
do
        python train_test.py --root /workspace/joel/CNNAudioClassification/data --out_dir checkpoints/test_freq_margin{$margin}/ \
                --dataset_name esc50 --num_workers 8 --batch_size 32 \
                --seed 42 --soft_epsilon 0.2 --lr 1e-2 --epochs 300 \
                --n_time_masks 3 --time_mask_param 40 --n_freq_masks 3 --freq_mask_param 15 \
                --margin_ratio $margin --gpu_id $GPUID
done