for dataset in anatomy largebio
do
    for emb_dim in 25 50 75 100
    do
        python3 single_graph_experiment.py --num_folds 5  --dataset $dataset --space mobius --device cuda:0 --emb_dim $emb_dim > ../final_logs/$dataset/mobius_$emb_dim.log 
        python3 single_graph_experiment.py --num_folds 5  --dataset $dataset --space euclidean --device cuda:0 --emb_dim $emb_dim > ../final_logs/$dataset/euclidean_$emb_dim.log 
        python3 single_graph_experiment.py --num_folds 5  --dataset $dataset --space hyperbolic --device cuda:0 --emb_dim $emb_dim > ../final_logs/$dataset/hyperbolic_$emb_dim.log 
        
        echo "---------------$emb_dim done---------------"
    done

    echo "--------------------$dataset done--------------------"
done
