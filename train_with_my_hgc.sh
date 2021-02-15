export DATAPATH=data
python -W ignore train_with_my_hgc.py --task nc --dataset pubmed --model MyHGCN --lr 0.003 --dim 501 --num-layers 2 --act relu --bias 1 --dropout 0.7 --weight-decay 0.0005 --manifold Hyperboloid --log-freq 5 --cuda 0

