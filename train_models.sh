
nohup python step1_train_models.py --devices 0 --fold 0 --width 32 --arch Unet --batch-size 1 --warm_restart > ./Unet3D.log 2>&1 &

nohup python step1_train_models.py --devices 0 --fold 0 --width 32 --arch EquiUnet --batch-size 1 --warm_restart > ./EquiUnet.log 2>&1 &

nohup python step1_train_models.py --devices 0 --fold 0 --width 32 --arch ACMINet --batch-size 1 --warm_restart --deep_sup > ./ACMINet.log 2>&1 &
