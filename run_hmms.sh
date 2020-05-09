
data_path=$1
out_path=${2:-"../hmm_models/"}
n_iter=${3:-100}
n_modes=${*:4}

if [ "$n_modes" == "" ]
then
    n_modes="4 8 16 32 64 128"
fi

echo "Starting fit_hmm with parameters:"
echo -e '\t'$data_path
echo -e '\t'$out_path
echo -e "\t--nIter $n_iter"
echo -e "\t--nModes $n_modes"

echo "echo $n_modes | xargs -P 10 -n 1 python src/fit_hmm.py $data_path $out_path --nIter $n_iter --nModes "
echo $n_modes | xargs -P 10 -n 1 python src/fit_hmm.py $data_path $out_path --nIter $n_iter --nModes 
