

export CUDA_VISIBLE_DEVICES=0 && python pre_train.py \
--datapath dataset \
--dataname BRATS2020 \
--savepath pre_results \
--batch_size 2 \
--lr 2e-4 \
--weight_decay 1e-4 \
--num_epochs 300 \
--iter_per_epoch 150 \
2>&1 | tee  pre_results/preTrain.log

export CUDA_VISIBLE_DEVICES=0 && python train.py \
--datapath dataset \
--dataname BRATS2020 \
--savepath fine_results \
batch_size 2 \
--resume pre_results/model_last.pth \
--lr 2e-4 \
--weight_decay 1e-4 \
--num_epochs 300 \
--iter_per_epoch 150 \
2>&1 | tee  fine_results/finetune.log


export CUDA_VISIBLE_DEVICES=0 && python post_train.py \
--datapath dataset \
--dataname BRATS2020 \
--savepath post_results \
batch_size 1 \
--resume fine_results/model_last.pth \
--lr 2e-4 \
--weight_decay 1e-4 \
--num_epochs 900 \
--iter_per_epoch 150 \
2>&1 | tee  post_results/postTrain.log