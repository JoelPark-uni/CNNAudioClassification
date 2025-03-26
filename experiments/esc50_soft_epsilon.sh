for soft_epsilon in 0.1 0.2
do
    python train.py --root /workspace/CNNAudioClassification/data --out_dir checkpoints/e{$soft_epsilon} \
        --dataset_name esc50 --num_workers 8 --batch_size 32 \
        --seed 42 --soft_epsilon $soft_epsilon --lr 1e-2 --epochs 300 \
        --time_masking 20 --freq_masking 10
done
