datapath=data/insplad-seg
datasets=('vari-grip')
dataset_flags=($(for dataset in "${datasets[@]}"; do echo '-d '"${dataset}"; done))

python main.py \
--gpu 0 \
--seed 0 \
--log_group insplad-seg \
--log_project insplad \
--results_path results \
--run_name run \
run \
-results_path wideresnet50 \

