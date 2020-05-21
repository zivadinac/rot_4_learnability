sim_dir=../simulated
sim_nc_dir=../simulated_noise_corr
max_jobs?=12
input_size?=128
rf_size?=64
noise_corr?=0.5

population_sizes?=32 64 128 256 512 1024
sim_data_files=$(patsubst %,$(sim_dir)/$@/nat_stim_$(input_size)_long_ps_%.pck, $(population_sizes))
sim_nc_data_files=$(patsubst %,$(sim_nc_dir)/$@/nat_stim_$(input_size)_long_ps_%.pck, $(population_sizes))

corr_cmd := python ../../src/correlation_matrix.py 


.PHONY: clean clean_$(input_size)_rf_$(rf_size) clean_prentice_et_al_2016

retina_$(input_size)_rf_$(rf_size):
	mkdir $@
	echo $(sim_data_files) | xargs -P $(max_jobs) -n 1 $(corr_cmd) --out_path $@ --data_path 

retina_$(input_size)_rf_$(rf_size)_nc_$(noise_corr):
	mkdir $@
	echo $(sim_nc_data_files) | xargs -P $(max_jobs) -n 1 $(corr_cmd) --out_path $@ --data_path 

prentice_et_al_2016:
	mkdir $@
	$(corr_cmd) --out_path $@ --data_path ../prentice_et_al_2016/unique_natural_movie/data.mat

clean_$(input_size)_rf_$(rf_size):
	rm -rf retina_$(input_size)_rf_$(rf_size)

clean_prentice_et_al_2016:
	rm -rf prentice_et_al_2016

clean: clean_prentice_et_al_2016 clean_$(input_size)_rf_$(rf_size)
	rm -rf */
	

