seed?=$(shell date +%Y)$(shell date +%j)

all: nat_stim simulated corr_mat
	echo "Data generated."

.PHONY: nat_stim simulated corr_mat clean_all

### prentice_et_al_2016

prentice_et_al_2016:
	make -C prentice_et_al_2016

clean_prentice_et_al_2016:
	make -C prentice_et_al_2016 clean

### nat_stim

nat_stim:
	make -C nat_stim

clean_nat_stim:
	make -C nat_stim clean

### simulated

simulated: simulated/retina_64_rf_64 simulated/retina_128_rf_64 simulated/retina_256_rf_64
	echo "Generated simulated data."

simulated/retina_64_rf_64: nat_stim
	make -C simulated input_size=64 rf_size=64 seed=$(seed)

simulated/retina_128_rf_64: nat_stim
	make -C simulated input_size=128 rf_size=64 seed=$(seed)

simulated/retina_256_rf_64: nat_stim
	make -C simulated input_size=256 rf_size=64 seed=$(seed)

clean_simulated:
	make -C simulated clean

### corr_mat

corr_mat: corr_mat/prentice_et_al_2016 corr_mat/retina_64_rf_64 corr_mat/retina_128_rf_64 corr_mat/retina_256_rf_64
	echo "Computed all correlation matrices."

corr_mat/prentice_et_al_2016:
	make -C corr_mat input_size=64 rf_size=64 prentice_et_al_2016

corr_mat/retina_64_rf_64: simulated/retina_64_rf_64
	make -C corr_mat input_size=64 rf_size=64

corr_mat/retina_128_rf_64: simulated/retina_128_rf_64
	make -C corr_mat input_size=128 rf_size=64

corr_mat/retina_256_rf_64: simulated/retina_256_rf_64
	make -C corr_mat input_size=256 rf_size=64

corr_mat/prentice_et_al_2016: prentice_et_al_2016
	make -C corr_mat prentice_et_al_2016

clean_corr_mat:
	make -C corr_mat clean


clean_all: clean_nat_stim clean_simulated clean_prentice_et_al_2016
	echo "Cleaned all data."
