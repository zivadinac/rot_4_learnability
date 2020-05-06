
data_path=$1
out_path=${2:-"../hmm_models/"}
n_iter=${3:-300}
n_modes=${*:4}

if [ "$n_modes" == "" ]
then
    n_modes="4 8 16 32 64 128"
fi

echo $n_modes | xargs -P 10 -n 1 python src/fit_hmm.py $data_path $out_path --nIter $n_iter --nModes 
