#!/bin/sh

# allow copy of 100000 items into stack shell
ulimit -S -s 100000

baseDirDes=~/'mask-remover-nn/CelebAMask-HQ-Model'
masksImagesDir=~/'mask-remover-nn/DataGenerator/images'

imgTrainDirDes='CelebA-HQ-img-train'
imgValDirDes='CelebA-HQ-img-val'
imgAnnoTrainDirDes='CelebA-HQ-anno-train'
imgAnnoValDirDes='CelebA-HQ-anno-val'

baseDirSrc=~/'datasets/CelebAMask-HQ'
imgDirSrc='CelebA-HQ-img'
imgAnnoDirSrc='CelebAMask-HQ-mask-anno'

# creating Train data & Validation image data dirs
mkdir $baseDirDes
echo $baseDirDes was created.

# train & val img dirs
mkdir $baseDirDes/$imgTrainDirDes
echo $baseDirDes/$imgTrainDirDes was created.

mkdir $baseDirDes/$imgValDirDes
echo $baseDirDes/$imgValDirDes was created.

# train & val imgs annotations dirs
mkdir $baseDirDes/$imgAnnoTrainDirDes
echo $baseDirDes/$imgAnnoTrainDirDes was created.

mkdir $baseDirDes/$imgAnnoValDirDes
echo $baseDirDes/$imgAnnoValDirDes was created.
# copying 80% of CelebA-HQ-img images to train dir
echo coping images to $baseDirDes/$imgTrainDirDes..
cp -i $baseDirSrc/$imgDirSrc/{0..23999}.jpg $baseDirDes/$imgTrainDirDes

# copying the last 20% of CelebA-HQ-img images to val dir
echo coping images to $baseDirDes/$imgValDirDes..
cp -i $baseDirSrc/$imgDirSrc/{24000..29999}.jpg $baseDirDes/$imgValDirDes
echo finished copying.

# copying of CelebA-HQ annotations images to anno-train & anno-val dir
for f in $baseDirSrc/$imgAnnoDirSrc/*; do
    if [ -d "$f" ]; then
	dirName=$(($(basename $f)))
	limit=11
	echo coping images annotations to $baseDirDes/$imgAnnoTrainDirDes & $imgAnnoTrainDirDes..
        if [ $dirName -le $limit ]; then # copy to anno-train dir
	    cp -i $f/* $baseDirDes/$imgAnnoTrainDirDes
	else # copy to anno-val dir
	    cp -i $f/* $baseDirDes/$imgAnnoValDirDes
	fi
    fi
done
echo finished copying.


echo "Do you wish to generate data?"
select yn in "Yes" "No"; do
    case $yn in
        Yes ) echo generating synthetic data..
        # the follow line will not execute because I changed to argparser
python data_generator.py $masksImagesDir $baseDirDes/$imgTrainDirDes $baseDirDes/$imgAnnoTrainDirDes $baseDirDes/$imgValDirDes $baseDirDes/$imgAnnoValDirDes
echo finished!; break;;
        No ) exit;;
    esac
done



