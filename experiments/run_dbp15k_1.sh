python single_graph_experiment.py --num_folds 12 --dataset dbp15k --space euclidean --category en_fr --device cpu > en_fr_euclidean.log &
python single_graph_experiment.py --num_folds 12 --dataset dbp15k --space euclidean --category fr_en --device cpu > fr_en_euclidean.log

python3 single_graph_experiment.py --num_folds 12 --dataset dbp15k --space hyperbolic --category en_fr --device cuda:0 > ../logs/en_fr_hyperbolic.log
python single_graph_experiment.py --num_folds 12 --dataset dbp15k --space hyperbolic --category fr_en --device cpu > fr_en_hyperbolic.log


