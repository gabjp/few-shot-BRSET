CUDA_VISIBLE_DEVICES=0 nohup python train.py \
 --learning-rate 0.001 \
 --epochs 60 \
 --tasks diabetic_retinopathy \
 --train-set 40-shot \
 --checkpoint-path ../../www/BRSET/$1/$2 \
 --save-path ../../www/BRSET/$1/test/diabetic_retinopathy \
 --batch-size 16 \
 --img-resize 256,256 \
 --transform \
 --schedule \
 --l2 0 \
 --optimizer adam  > diabetic_retinopathy.out &

CUDA_VISIBLE_DEVICES=0 nohup python train.py \
 --learning-rate 0.001 \
 --epochs 60 \
 --tasks vascular_occlusion \
 --train-set 40-shot \
 --checkpoint-path ../../www/BRSET/$1/$2 \
 --save-path ../../www/BRSET/$1/test/vascular_occlusion \
 --batch-size 16 \
 --img-resize 256,256 \
 --transform \
 --schedule \
 --l2 0 \
 --optimizer adam  > vascular_occlusion.out &

CUDA_VISIBLE_DEVICES=0 nohup python train.py \
 --learning-rate 0.001 \
 --epochs 60 \
 --tasks hemorrhage \
 --train-set 40-shot \
 --checkpoint-path ../../www/BRSET/$1/$2 \
 --save-path ../../www/BRSET/$1/test/hemorrhage \
 --batch-size 16 \
 --img-resize 256,256 \
 --transform \
 --schedule \
 --l2 0 \
 --optimizer adam  > hemorrhage.out &

CUDA_VISIBLE_DEVICES=0 nohup python train.py \
 --learning-rate 0.001 \
 --epochs 60 \
 --tasks scar \
 --train-set 40-shot \
 --checkpoint-path ../../www/BRSET/$1/$2 \
 --save-path ../../www/BRSET/$1/val/scar \
 --batch-size 16 \
 --img-resize 256,256 \
 --transform \
 --schedule \
 --l2 0 \
 --optimizer adam  > scar.out &

CUDA_VISIBLE_DEVICES=0 nohup python train.py \
 --learning-rate 0.001 \
 --epochs 60 \
 --tasks macular_edema \
 --train-set 40-shot \
 --checkpoint-path ../../www/BRSET/$1/$2 \
 --save-path ../../www/BRSET/$1/val/macular_edema \
 --batch-size 16 \
 --img-resize 256,256 \
 --transform \
 --schedule \
 --l2 0 \
 --optimizer adam  > macular_edema.out &