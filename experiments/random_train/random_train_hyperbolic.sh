    
for train_size in 10 20 50 100 500 1000 2000 5000
do
    mkdir results/"$train_size"
    
    for i in {1..20}
    do
        python random_train.py --space hyperbolic --train_size $train_size --experiment_num $i
    done
    echo "hyp $train_size done"
done

echo "++++++++++++++HYP DONE++++++++++++++"
