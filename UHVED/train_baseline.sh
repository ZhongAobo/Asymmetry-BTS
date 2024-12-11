cd /home/SHARE2/ZXY/RF_main/RF2U-HVED
source activate py39_pyt112


export CUDA_VISIBLE_DEVICES=1 && python train.py --savepath results --num_epochs 300 \
  2>&1 | tee  logs/train_epoch300.log
