python single_graph_experiment.py --num_folds 16 --dataset dbp15k --space euclidean --category en_fr --device cuda:1 > en_fr_euclidean.log &
python single_graph_experiment.py --num_folds 16 --dataset dbp15k --space euclidean --category fr_en --device cuda:1 > fr_en_euclidean.log

python single_graph_experiment.py --num_folds 16 --dataset dbp15k --space hyperbolic --category en_fr --device cuda:1 > en_fr_hyperbolic.log
python single_graph_experiment.py --num_folds 16 --dataset dbp15k --space hyperbolic --category en_fr --device cuda:1 > fr_en_hyperbolic.log


