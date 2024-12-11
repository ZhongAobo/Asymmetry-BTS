cd /home/SHARE2/ZXY/RF_main/RF2U-HVED
source activate py39_pyt112


export CUDA_VISIBLE_DEVICES=1 && python eval.py --resume results/model_900.pth \
  2>&1 | tee  logs/eval.log