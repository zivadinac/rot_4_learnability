
stimulus_path=${1:-"../data/nat_stim/nat_stim_256.mkv"}
out_path=${2:-"../data/my_simulations/"}
stimulus_repeats=${3:-10}
rf_size=${4:-64}
off_prob=${5:-0.5}
timesteps=${6:-16}
batch_size=${7:-1}
use_gpu=${8:-0}
save_rfs=${9:-0}
population_sizes=${*:10}

if [ "$population_sizes" == ""  ]
then
    population_sizes="32 64 128 256 512 1024"
fi

seed=None # if not None change calling line also

echo "Starting simulations with parameters:"
echo -e '\t'$stimulus_path
echo -e '\t'$out_path
echo -e "\t--stimulus_repeats $stimulus_repeats"
echo -e "\t--timesteps $timesteps"
echo -e "\t--batch_size $batch_size"
echo -e "\t--use_gpu $use_gpu"
echo -e "\t--save_rfs $save_rfs"
echo -e "\t--seed $seed"
echo -e "\t--rf_size $rf_size $rf_size"
echo -e "\t--off_prob $off_prob"
echo -e "\tpopulation sizes: $population_sizes"

echo "$population_sizes" | xargs -P 12 -n 1 python src/simulation.py $stimulus_path $out_path --stimulus_repeats $stimulus_repeats --timesteps $timesteps --batch_size $batch_size --use_gpu $use_gpu --save_rfs $save_rfs --rf_size $rf_size $rf_size --off_prob $off_prob --population_size 
#echo "$population_sizes" | xargs -P 12 -n 1 python src/simulation.py $stimulus_path $out_path --stimulus_repeats $stimulus_repeats --timesteps $timesteps --batch_size $batch_size --seed $seed --use_gpu $use_gpu --save_rfs $save_rfs --rf_size $rf_size $rf_size --off_prob $off_prob --population_size 
