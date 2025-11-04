python="/home/matthis/Documents/Studium/5 MLO/notebooks/.venv/bin/python"
train_model="/home/matthis/Documents/Studium/5 MLO/notebooks/train_moodel_2.py"

for c in 0.1; do
    echo "Trainiere LR mit C=$c"
    #"$python" "$train_model" --model logistic_regression --C "$c"
done

for depth in 3 10; do
    for criterion in "gini" "log_loss" ; do
        for min_samples_split in 5 15 30; do
            for min_samples_leaf in 5 10; do
                echo "Trainiere DT mit max_depth=$depth, criterion=$criterion, min_sammples_split=$min_samples_split, min_samples_leaf=$min_samples_leaf"
                "$python" "$train_model" --model decision_tree --max_depth "$depth" --criterion "$criterion" --min_samples_split "$min_samples_split --min_samples_leaf "$min_samples_leaf"
            done
        done
    done
done