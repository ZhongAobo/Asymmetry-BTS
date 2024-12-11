cd /home/SHARE2/ZXY/RF_main/RF2U-HVED
source activate py39_pyt112

export CUDA_VISIBLE_DEVICES=1 && python pretrain.py --savepath results --scale 8 --num_epochs 300 \
  2>&1 | tee  logs/pretrain_epoch300_scale8.log

export CUDA_VISIBLE_DEVICES=1 && python train.py --savepath results --resume results/pretrain_300.pth --num_epochs 300 \
  2>&1 | tee  logs/train_epoch300.log

export CUDA_VISIBLE_DEVICES=1 && python posttrain.py --savepath results --resume results/train_300.pth --num_epochs 300 \
  2>&1 | tee  logs/posttrain_epoch300.log