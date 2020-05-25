corr_mat_dir=../data/corr_mat
max_jobs?=12
input_size?=128
rf_size?=64
noise_corr?=0.5

#population_sizes?=32 64 128 256 512 1024
population_sizes?=32
sim_data_files=$(patsubst %, $(corr_mat_dir)/$@/corr_ps_%.pck, $(population_sizes))
pre_data_file=$(corr_mat_dir)/prentice_et_al_2016/corr_ps_152.pck

hmm_models_dir=../models/hmm

corr_cmd := python ../src/figure_correlations.py 
hmm_ll_cmd := python ../src/figure_hmms.py


.PHONY: clean clean_$(input_size)_rf_$(rf_size) clean_prentice_et_al_2016

define hmm_file
$(hmm_models_dir)/$1/nat_stim_$(input_size)_long_ps_$2_nModes_
endef

define hmm_ll_out_file
$1/hmms_ps_$2.png
endef

n_modes_32=2 5 15 25 35
n_modes_64=2 5 15 25 35 45 55 65
n_modes_128=2 5 15 25 35 45 55 65 75 85 95 105 115 125 135
n_modes_256=2 5 15 25 35 45 55 65 75 85 95 105 115 125 135 145 155 165

### simulated retina
retina_$(input_size)_rf_$(rf_size): retina_$(input_size)_rf_$(rf_size)_hmm_ll
	mkdir $@
	$(corr_cmd) --out_path $@ --data_paths $(pre_data_file) $(sim_data_files)

retina_$(input_size)_rf_$(rf_size)_hmm_ll: retina_$(input_size)_rf_$(rf_size)/ps_32 retina_$(input_size)_rf_$(rf_size)/ps_64 retina_$(input_size)_rf_$(rf_size)/ps_128 retina_$(input_size)_rf_$(rf_size)/ps_256 retina_$(input_size)_rf_$(rf_size)/ps_512 retina_$(input_size)_rf_$(rf_size)/ps_1024
	@echo "Plotted all HMM log li."

retina_$(input_size)_rf_$(rf_size)/ps_32:
	mkdir -p $@
	$(hmm_ll_cmd) $(call hmm_file,$@,32) $(call hmm_ll_out_file,$@,32) --n_modes $(n_modes_32)

retina_$(input_size)_rf_$(rf_size)/ps_64:
	mkdir -p $@
	$(hmm_ll_cmd) $(call hmm_file,$@,64) $(call hmm_ll_out_file,$@,64) --n_modes $(n_modes_64)

retina_$(input_size)_rf_$(rf_size)/ps_128:
	mkdir -p $@
	$(hmm_ll_cmd) $(call hmm_file,$@,128) $(call hmm_ll_out_file,$@,128) --n_modes $(n_modes_128)

retina_$(input_size)_rf_$(rf_size)/ps_256:
	mkdir -p $@
	$(hmm_ll_cmd) $(call hmm_file,$@,256) $(call hmm_ll_out_file,$@,256) --n_modes $(n_modes_256)

retina_$(input_size)_rf_$(rf_size)/ps_512:
	mkdir -p $@
	$(hmm_ll_cmd) $(call hmm_file,$@,512) $(call hmm_ll_out_file,$@,512) --n_modes $(n_modes_512)

retina_$(input_size)_rf_$(rf_size)/ps_1024:
	mkdir -p $@
	$(hmm_ll_cmd) $(call hmm_file,$@,1024) $(call hmm_ll_out_file,$@,1024) --n_modes $(n_modes_1024)

clean_$(input_size)_rf_$(rf_size):
	rm -rf retina_$(input_size)_rf_$(rf_size)

### simulated retina with noise correlation
retina_$(input_size)_rf_$(rf_size)_nc_$(noise_corr): retina_$(input_size)_rf_$(rf_size)_nc_$(noise_corr)_hmm_ll
	mkdir $@
	$(corr_cmd) --out_path $@ --data_paths $(pre_data_file) $(sim_data_files)

retina_$(input_size)_rf_$(rf_size)_nc_$(noise_corr)_hmm_ll: retina_$(input_size)_rf_$(rf_size)_nc_$(noise_corr)/ps_32 retina_$(input_size)_rf_$(rf_size)_nc_$(noise_corr)/ps_64 retina_$(input_size)_rf_$(rf_size)_nc_$(noise_corr)/ps_128 retina_$(input_size)_rf_$(rf_size)_nc_$(noise_corr)/ps_256 retina_$(input_size)_rf_$(rf_size)_nc_$(noise_corr)/ps_512 retina_$(input_size)_rf_$(rf_size)_nc_$(noise_corr)/ps_1024
	@echo "Plotted all HMM log li."

retina_$(input_size)_rf_$(rf_size)_nc_$(noise_corr)/ps_32:
	mkdir -p $@
	$(hmm_ll_cmd) $(call hmm_file,$@,32) $(call hmm_ll_out_file,$@,32) --n_modes $(n_modes_32)

retina_$(input_size)_rf_$(rf_size)_nc_$(noise_corr)/ps_64:
	mkdir -p $@
	$(hmm_ll_cmd) $(call hmm_file,$@,64) $(call hmm_ll_out_file,$@,64) --n_modes $(n_modes_64)

retina_$(input_size)_rf_$(rf_size)_nc_$(noise_corr)/ps_128:
	mkdir -p $@
	$(hmm_ll_cmd) $(call hmm_file,$@,128) $(call hmm_ll_out_file,$@,128) --n_modes $(n_modes_128)

retina_$(input_size)_rf_$(rf_size)_nc_$(noise_corr)/ps_256:
	mkdir -p $@
	$(hmm_ll_cmd) $(call hmm_file,$@,256) $(call hmm_ll_out_file,$@,256) --n_modes $(n_modes_256)

retina_$(input_size)_rf_$(rf_size)_nc_$(noise_corr)/ps_512:
	mkdir -p $@
	$(hmm_ll_cmd) $(call hmm_file,$@,512) $(call hmm_ll_out_file,$@,512) --n_modes $(n_modes_512)

retina_$(input_size)_rf_$(rf_size)_nc_$(noise_corr)/ps_1024:
	mkdir -p $@
	$(hmm_ll_cmd) $(call hmm_file,$@,1024) $(call hmm_ll_out_file,$@,1024) --n_modes $(n_modes_1024)

clean_$(input_size)_rf_$(rf_size)_nc_$(noise_corr)_hmm_ll:
	rm -rf retina_$(input_size)_rf_$(rf_size)_nc_$(noise_corr)/ps_*

clean_$(input_size)_rf_$(rf_size)_nc_$(noise_corr):
	rm -rf retina_$(input_size)_rf_$(rf_size)_nc_$(noise_corr)

clean: clean_prentice_et_al_2016 clean_$(input_size)_rf_$(rf_size) clean_$(input_size)_rf_$(rf_size)_nc_$(noise_corr)
	rm -rf */
	

