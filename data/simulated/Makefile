videos_dir=../nat_stim

max_jobs?=12
input_size?=128

stimulus_repeats?=1
timesteps?=16
batch_size?=1
use_gpu?=0
save_rfs?=0
rf_size?=64
off_prob?=0.0
seed?=$(shell date +%Y)$(shell date +%j)
population_sizes?=32 64 128 256 512 1024

#TODO maby use different targets and make -j for parallelization?
simulation_cmd_1 := echo $(population_sizes) | xargs -P $(max_jobs) -n 1 python ../../src/simulation.py 
simulation_cmd_2 := --stimulus_repeats $(stimulus_repeats) --timesteps $(timesteps) --batch_size $(batch_size) --use_gpu $(use_gpu) --save_rfs $(save_rfs) --rf_size $(rf_size) $(rf_size) --off_prob $(off_prob) --seed $(seed) --population_size 


.PHONY: clean clean_$(input_size)_rf_$(rf_size)

retina_$(input_size)_rf_$(rf_size):
	mkdir $@
	$(simulation_cmd_1) $(videos_dir)/nat_stim_$(input_size)_long.mkv $@ $(simulation_cmd_2)

clean_$(input_size)_rf_$(rf_size):
	rm -rf retina_$(input_size)_rf_$(rf_size)

clean:
	rm -rf */
	

