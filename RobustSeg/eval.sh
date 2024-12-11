cd /home/SHARE2/ZXY/RF_main/RF2Robust
source activate py39_pyt112


export CUDA_VISIBLE_DEVICES=1 && python eval.py --resume result/model_last.pth \
  2>&1 | tee  logs/eval.log