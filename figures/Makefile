corr_mat_dir=../data/corr_mat
max_jobs?=12
input_size?=128
rf_size?=64

population_sizes?=32 64 128 256 512 1024
sim_data_files=$(patsubst %, $(corr_mat_dir)/$@/corr_ps_%.pck, $(population_sizes))
pre_data_file=$(corr_mat_dir)/prentice_et_al_2016/corr_ps_152.pck

corr_cmd := python ../src/figure_correlations.py 


.PHONY: clean clean_$(input_size)_rf_$(rf_size) clean_prentice_et_al_2016

retina_$(input_size)_rf_$(rf_size):
	mkdir $@
	$(corr_cmd) --out_path $@ --data_paths $(pre_data_file) $(sim_data_files)

clean_$(input_size)_rf_$(rf_size):
	rm -rf retina_$(input_size)_rf_$(rf_size)

clean: clean_prentice_et_al_2016 clean_$(input_size)_rf_$(rf_size)
	rm -rf */
	

