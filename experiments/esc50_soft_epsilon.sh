for soft_epsilon in 0.1 0.2
do
    python train.py --root /workspace/CNNAudioClassification/data --out_dir checkpoints/e{$soft_epsilon} \
        --dataset_name esc50 --num_workers 8 --batch_size 32 \
        --seed 42 --soft_epsilon $soft_epsilon --lr 1e-2 --epochs 300 \
        --n_time_masks 3 --time_mask_param 40 --n_freq_masks 3 --freq_mask_param 15
done
