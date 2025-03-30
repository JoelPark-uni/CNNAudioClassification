GPUID=1
dataset_name=urbansound8k

for margin in 0 1 2
do
    python train.py --root /workspace/joel/CNNAudioClassification/data --out_dir checkpoints/$dataset_name/margin$margin \
        --dataset_name $dataset_name --num_workers 8 --batch_size 32 \
        --seed 42 --soft_epsilon 0.2 --lr 1e-2 --epochs 300 \
        --n_time_masks 3 --time_mask_param 40 --n_freq_masks 3 --freq_mask_param 15 \
        --gpu_id $GPUID --use_segmentation --margin_ratio $margin
done