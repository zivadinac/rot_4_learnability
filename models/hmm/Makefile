sim_dir=../../data/simulated
sim_nc_dir=../../data/simulated_noise_corr
input_size?=128
rf_size?=64
noise_corr?=0.5

cross_val_folds?=2
n_iter?=500
eta?=0.002
seed?=$(shell date +%Y)$(shell date +%j)
max_jobs?=$(shell expr 12 / $(cross_val_folds))

population_sizes?=32 64 128 256 512 1024
sim_data_file=$(sim_dir)/retina_$(input_size)_rf_$(rf_size)/nat_stim_$(input_size)_long_ps
sim_nc_data_file=$(sim_nc_dir)/retina_$(input_size)_rf_$(rf_size)_nc_$(noise_corr)/nat_stim_$(input_size)_long_ps

corr_cmd_1 := xargs -P $(max_jobs) -n 1 python ../../src/fit_hmm.py 
corr_cmd_2 := --cross_val_folds $(cross_val_folds) --n_iter $(n_iter) --eta $(eta) --seed $(seed) --n_modes 

n_modes_32=1 2 5 15 25 35
n_modes_64=1 2 5 15 25 35 45 55 65
n_modes_128=1 2 5 15 25 35 45 55 65 75 85 95 105 115 125
n_modes_256=1 2 5 15 25 35 45 55 65 75 85 95 105 115 125

.PHONY: clean clean_$(input_size)_rf_$(rf_size) clean_prentice_et_al_2016


### simulated retina
retina_$(input_size)_rf_$(rf_size): retina_$(input_size)_rf_$(rf_size)/ps_32 retina_$(input_size)_rf_$(rf_size)/ps_64 retina_$(input_size)_rf_$(rf_size)/ps_128 retina_$(input_size)_rf_$(rf_size)/ps_256
	@echo "Fitted hmm models for $@ and population sizes $(population_sizes)"

retina_$(input_size)_rf_$(rf_size)/ps_32:
	mkdir -p $@
	echo $(n_modes_32) | $(corr_cmd_1) $(sim_data_file)_32.pck $@ $(corr_cmd_2)

retina_$(input_size)_rf_$(rf_size)/ps_64:
	mkdir -p $@
	echo $(n_modes_64) | $(corr_cmd_1) $(sim_data_file)_64.pck $@ $(corr_cmd_2)

retina_$(input_size)_rf_$(rf_size)/ps_128:
	mkdir -p $@
	echo $(n_modes_128) | $(corr_cmd_1) $(sim_data_file)_128.pck $@ $(corr_cmd_2)

retina_$(input_size)_rf_$(rf_size)/ps_256:
	mkdir -p $@
	echo $(n_modes_256) | $(corr_cmd_1) $(sim_data_file)_256.pck $@ $(corr_cmd_2)

retina_$(input_size)_rf_$(rf_size)/ps_512:
	mkdir -p $@
	echo $(n_modes_512) | $(corr_cmd_1) $(sim_data_file)_512.pck $@ $(corr_cmd_2)

retina_$(input_size)_rf_$(rf_size)/ps_1024:
	mkdir -p $@
	echo $(n_modes_1024) | $(corr_cmd_1) $(sim_data_file)_1024.pck $@ $(corr_cmd_2)

clean_$(input_size)_rf_$(rf_size):
	rm -rf retina_$(input_size)_rf_$(rf_size)


### simulated retina with noise correlation
retina_$(input_size)_rf_$(rf_size)_nc_$(noise_corr): retina_$(input_size)_rf_$(rf_size)_nc_$(noise_corr)/ps_32 retina_$(input_size)_rf_$(rf_size)_nc_$(noise_corr)/ps_64 retina_$(input_size)_rf_$(rf_size)_nc_$(noise_corr)/ps_128 retina_$(input_size)_rf_$(rf_size)_nc_$(noise_corr)/ps_256
	@echo "Fitted hmm models for $@ and population sizes $(population_sizes)"

retina_$(input_size)_rf_$(rf_size)_nc_$(noise_corr)/ps_32:
	mkdir -p $@
	echo $(n_modes_32) | $(corr_cmd_1) $(sim_nc_data_file)_32.pck $@ $(corr_cmd_2)

retina_$(input_size)_rf_$(rf_size)_nc_$(noise_corr)/ps_64:
	mkdir -p $@
	echo $(n_modes_64) | $(corr_cmd_1) $(sim_nc_data_file)_64.pck $@ $(corr_cmd_2)

retina_$(input_size)_rf_$(rf_size)_nc_$(noise_corr)/ps_128:
	mkdir -p $@
	echo $(n_modes_128) | $(corr_cmd_1) $(sim_nc_data_file)_128.pck $@ $(corr_cmd_2)

retina_$(input_size)_rf_$(rf_size)_nc_$(noise_corr)/ps_256:
	mkdir -p $@
	echo $(n_modes_256) | $(corr_cmd_1) $(sim_nc_data_file)_256.pck $@ $(corr_cmd_2)

retina_$(input_size)_rf_$(rf_size)_nc_$(noise_corr)/ps_512:
	mkdir -p $@
	echo $(n_modes_512) | $(corr_cmd_1) $(sim_nc_data_file)_512.pck $@ $(corr_cmd_2)

retina_$(input_size)_rf_$(rf_size)_nc_$(noise_corr)/ps_1024:
	mkdir -p $@
	echo $(n_modes_1024) | $(corr_cmd_1) $(sim_nc_data_file)_1024.pck $@ $(corr_cmd_2)

clean_$(input_size)_rf_$(rf_size)_nc_$(noise_corr):
	rm -rf retina_$(input_size)_rf_$(rf_size)_nc_$(noise_corr)


clean: clean_$(input_size)_rf_$(rf_size)
	rm -rf */
	

