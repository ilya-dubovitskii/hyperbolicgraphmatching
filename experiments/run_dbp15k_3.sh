python single_graph_experiment.py --num_folds 16 --dataset dbp15k --space euclidean --category zh_en --device cuda:3 > zh_en_euclidean.log &
python single_graph_experiment.py --num_folds 16 --dataset dbp15k --space euclidean --category en_zh --device cuda:3 > en_zh_euclidean.log

python single_graph_experiment.py --num_folds 16 --dataset dbp15k --space hyperbolic --category zh_en --device cuda:3 > zh_en_hyperbolic.log
python single_graph_experiment.py --num_folds 16 --dataset dbp15k --space hyperbolic --category zh_en --device cuda:3 > fr_en_hyperbolic.log


