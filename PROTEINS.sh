[ -z "${data}" ] && data="--batch_size 48 --max_node 512"
[ -z "${epoch}" ] && epoch="--split_epochs 50 --end_epochs 200 --retrain_epochs 200"
[ -z "${update}" ] && update="--warmup_updates 600 --tot_updates 5000"
[ -z "${dataset}" ] && dataset="--data_split $2"
[ -z "${archparam}" ] && archparam="--n_layers 4 --num_heads 4 --hidden_dim 32 --ffn_dim 32"

CUDA_VISIBLE_DEVICES=$1 python -u entry.py $data $epoch $update $dataset $archparam
