for emb_dim in 30 60 100 200
do
    python single_graph_experiment.py --num_folds 5 --sim sqdist --dataset largebio --space euclidean --device cuda:3 --emb_dim $emb_dim &
    python single_graph_experiment.py --num_folds 5 --sim sqdist --dataset anatomy --space euclidean --device cuda:3 --emb_dim $emb_dim;
    python single_graph_experiment.py --num_folds 5 --sim sqdist --dataset largebio --space hyperbolic --device cuda:3 --emb_dim $emb_dim;
    python single_graph_experiment.py --num_folds 5 --sim sqdist --dataset anatomy --space hyperbolic --device cuda:3 --emb_dim $emb_dim;
    
    echo "---------------$emb_dim done---------------"
done