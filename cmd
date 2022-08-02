
# shapenet 数据集
CUDA_VISIBLE_DEVICES=0,2,3,4 python train.py --outdir=training-runs --cfg=shapenet --data=../dataset_preprocessing/shapenet_cars/cars_128.zip  --gpus=4 --batch=12 --gamma=0.3 --metrics None --mbstd-group 3 --cmax 256 --cbase 16384

python gen_samples.py --outdir=out --trunc=0.7 --shapes=true --seeds=1     --network=training-runs/00012-shapenet-cars_128-gpus4-batch12-gamma0.3/network-snapshot-001803.pkl --fov-deg 45


# blender chair数据集
CUDA_VISIBLE_DEVICES=5,6,7 python train.py --outdir=training-runs --cfg=blender --data=../dataset_preprocessing/chair.zip  --gpus=3 --batch=9 --gamma=0.3 --metrics None --mbstd-group 3 --cmax 256 --cbase 16384 --snap 4

python gen_samples.py --outdir=out --trunc=0.7 --shapes=true --seeds=5     --network=training-runs/00039-blender-chair-gpus3-batch9-gamma0.3/network-snapshot-000208.pkl