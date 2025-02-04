source activate
conda activate rfmotion

export CUDA_VISIBLE_DEVICES='0'
cd ..
rm -r ./nohup.out
nohup python -m train --cfg configs/config_vae.yaml --cfg_assets configs/assets.yaml --nodebug
#  -m train --cfg configs/config_vae.yaml --cfg_assets configs/assets.yaml --nodebug