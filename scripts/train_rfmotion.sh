source activate
conda activate rfmotion

export CUDA_VISIBLE_DEVICES='0'
cd ..
rm -r ./nohup.out
# python -m train --cfg configs/config_rfmotion.yaml --cfg_assets configs/assets.yaml --nodebug
nohup python -m train --cfg configs/config_rfmotion.yaml --cfg_assets configs/assets.yaml --nodebug