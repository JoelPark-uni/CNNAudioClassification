for t_mask in 30 40 50
do
    for f_mask in 15
    do
        python train.py --root /workspace/CNNAudioClassification/data --out_dir checkpoints/masks{$t_mask}-{$f_mask} \
            --dataset_name esc50 --num_workers 8 --batch_size 32 \
            --seed 42 --soft_epsilon 0.2 --lr 1e-2 --epochs 300 \
            --n_time_masks 3 --time_mask_param $t_mask --n_freq_masks 3 --freq_mask_param $f_mask
    done
done

# python train.py --root /workspace/CNNAudioClassification/data --out_dir checkpoints/ \
#             --dataset_name esc50 --num_workers 8 --batch_size 32 \
#             --seed 42 --soft_epsilon 0.2 --lr 1e-2 --epochs 300 \
#             --n_time_masks 3 --time_mask_param 40 --n_freq_masks 3 --freq_mask_param 15