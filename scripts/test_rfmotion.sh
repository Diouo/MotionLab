source activate
conda activate rfmotion
cd ..

python -m test --cfg configs/config_rfmotion.yaml --cfg_assets configs/assets.yaml
# nohup python -m test --cfg configs/config_rfmotion.yaml --cfg_assets configs/assets.yaml