
stimulus_path=${1:-"../data/nat_stim/nat_stim_256.mkv"}
out_path=${2:-"../data/my_simulations/"}
stimulus_repeats=${3:-10}
timesteps=${4:-16}
batch_size=${5:-1}
use_gpu=${6:-0}
seed=${7:-12345}
population_sizes=${*:8}

if [ "$population_sizes" == ""  ]
then
    population_sizes="32 64 128 256 512 1024"
fi
echo $population_sizes

echo "Starting simulations with parameters:"
echo -e '\t'$stimulus_path
echo -e '\t'$out_path
echo -e "\t--stimulus_repeats $stimulus_repeats"
echo -e "\t--timesteps $timesteps"
echo -e "\t--batch_size $batch_size"
echo -e "\t--use_gpu $use_gpu"
echo -e "\t--seed $seed"
echo -e "\tpopulation sizes: $population_sizes"

echo "$population_sizes" | xargs -P 10 -n 1 python src/simulation.py $stimulus_path $out_path --stimulus_repeats $stimulus_repeats --timesteps $timesteps --batch_size $batch_size --seed $seed --use_gpu $use_gpu --population_size 
