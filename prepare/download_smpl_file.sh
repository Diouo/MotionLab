mkdir -p ./checkpoints
cd ./checkpoints
gdown "https://drive.google.com/uc?id=1INYlGA76ak_cKGzvpOV2Pe6RkYTlXTW2"
rm -rf smpl

unzip smpl.zip
echo -e "Cleaning\n"
rm smpl.zip
echo -e "Downloading done!"
