REM python advtrain.py dummyadvhp001 --epochs 17 --batch_size 64 --hp_lambda 0.01
REM python advtrain.py dummyadvhp005 --epochs 17 --batch_size 64 --hp_lambda 0.05
python advtrain.py dummyadvhp10 --epochs 17 --batch_size 64 --hp_lambda 0.1
python advtrain.py dummyadvhp50 --epochs 17 --batch_size 64 --hp_lambda 0.5
python advtrain.py dummyadvhp100 --epochs 17 --batch_size 64 --hp_lambda 1.0
python advtrain.py dummyadvhp250 --epochs 17 --batch_size 64 --hp_lambda 2.5
python advtrain.py dummyadvhp500 --epochs 17 --batch_size 64 --hp_lambda 5
python advtrain.py dummyadvhp730 --epochs 17 --batch_size 64 --hp_lambda 7.3
python advtrain.py dummyadvhp810 --epochs 17 --batch_size 64 --hp_lambda 8.1
python advtrain.py dummyadvhp1000 --epochs 17 --batch_size 64 --hp_lambda 10