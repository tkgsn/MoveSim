
# dataset=peopleflow
# max_size=10000
# data_name=$max_size
# latlon_config=peopleflow.json
# location_threshold=0
# time_threshold=0
# n_bins=38
# seed_for_dataset=0
# training_data_name=bin38

dataset=test
max_size=1000
data_name=normal_variable
latlon_config=test.json
location_threshold=0
time_threshold=0
n_bins=1
seed_for_dataset=0
training_data_name=seed0_size$max_size

python3 make_raw_data.py --original_data_name $dataset --max_size $max_size --seed $seed_for_dataset --save_name $data_name
python3 data_pre_processing.py --latlon_config  $latlon_config --dataset $dataset --data_name $data_name --location_threshold $location_threshold --time_threshold $time_threshold --save_name $training_data_name --n_bins $n_bins

seed=0
batch_size=100
embed_dim=32

cuda_number=0
n_pre_training_epochs=1000
n_epochs=1000
n_generated=10000
lr=1e-3
patience=100

n_generated_for_discriminator=10000

save_name=test

# set the options
pre_training=False
add_residual=False

declare -A options=(
    ["pre_training"]=$pre_training
    ["add_residual"]=$add_residual
)

# make the option parameter
option=""
for key in "${!options[@]}"; do
    if [ "${options[$key]}" = True ]; then
        option="$option --$key"
    fi
done

python3 run.py --cuda_number $cuda_number --n_pre_training_epochs $n_pre_training_epochs --n_epochs $n_epochs --n_generated $n_generated --dataset $dataset --data_name $data_name --training_data_name $training_data_name --seed $seed --batch_size $batch_size --embed_dim $embed_dim --save_name $save_name --lr $lr --patience $patience --n_generated_for_discriminator $n_generated_for_discriminator $option