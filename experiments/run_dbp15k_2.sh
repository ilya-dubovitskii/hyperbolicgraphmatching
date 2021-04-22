python single_graph_experiment.py --num_folds 16 --dataset dbp15k --space euclidean --category en_ja --device cuda:2 > en_ja_euclidean.log &
python single_graph_experiment.py --num_folds 16 --dataset dbp15k --space euclidean --category ja_en --device cuda:2 > ja_en_euclidean.log

python single_graph_experiment.py --num_folds 16 --dataset dbp15k --space hyperbolic --category en_ja --device cuda:2 > en_ja_hyperbolic.log
python single_graph_experiment.py --num_folds 16 --dataset dbp15k --space hyperbolic --category en_ja --device cuda:2 > ja_en_hyperbolic.log


