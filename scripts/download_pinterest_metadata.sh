mkdir -p data
cd data

# pinterest dataset from "Learning Image and User Features for Recommendation in Social Networks" ICCV'15
# https://sites.google.com/site/xueatalphabeta/academic-projects

fileid="0B0l8Lmmrs5A_REZXanM3dTN4Y28"
filename="pinterest_iccv.zip"
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}

unzip pinterest_iccv.zip
mv pinterest_iccv pinterest