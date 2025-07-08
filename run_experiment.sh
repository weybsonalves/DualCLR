DATASETS=("toxicity" "metabric-pam50" "lung" "prostate" "cll" "smk")

for DATASET in "${DATASETS[@]}"; do
    echo "Running experiment for dataset: $DATASET (without CRM)"
    python src/main.py \
        --model 'wpfs' \
        --dataset "$DATASET" \
        --use_best_hyperparams \
        --experiment_name "${DATASET}_original_v2" \
        --run_repeats_and_cv \
        --train_on_full_data \
        --max_steps 200

    echo "Running experiment for dataset: $DATASET (with CRM)"
    python src/main.py \
        --model 'wpfs' \
        --dataset "$DATASET" \
        --use_best_hyperparams \
        --experiment_name "${DATASET}_crm_v2" \
        --run_repeats_and_cv \
        --train_on_full_data \
        --max_steps 200 \
        --use_crm

    echo "Running experiment for dataset: $DATASET (with CRM and SupCon)"
    python src/main.py \
        --model 'wpfs' \
        --dataset "$DATASET" \
        --use_best_hyperparams \
        --experiment_name "${DATASET}_crm_supcon_v2" \
        --run_repeats_and_cv \
        --train_on_full_data \
        --max_steps 200 \
        --use_crm \
        --use_supcon

    echo "Running experiment for dataset: $DATASET (with CRM and SupCon)"
    python src/main.py \
        --model 'rf' \
        --dataset "$DATASET" \
        --use_best_hyperparams \
        --experiment_name "${DATASET}_crm_supcon_rf_v2" \
        --run_repeats_and_cv \
        --train_on_full_data \
        --max_steps 200 \
        --use_crm \
        --use_supcon
done