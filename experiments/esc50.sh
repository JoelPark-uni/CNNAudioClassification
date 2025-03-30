dataset_name=esc50
for seed in 0
do
    python train.py --root /workspace/CNNAudioClassification/data --out_dir checkpoints/$dataset_name/seed$seed \
        --dataset_name esc50 --num_workers 8 --batch_size 32 \
        --seed $seed --soft_epsilon 0.2 --lr 1e-2 --epochs 500 \
        --n_time_masks 3 --time_mask_param 40 --n_freq_masks 3 --freq_mask_param 15 \
        --use_segmentation --margin_ratio 0
done

# python train.py --root /workspace/CNNAudioClassification/data --out_dir checkpoints/ \
#             --dataset_name esc50 --num_workers 8 --batch_size 32 \
#             --seed 42 --soft_epsilon 0.2 --lr 1e-2 --epochs 300 \
#             --n_time_masks 3 --time_mask_param 40 --n_freq_masks 3 --freq_mask_param 15