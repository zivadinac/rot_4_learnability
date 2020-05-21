seed?=$(shell date +%Y)$(shell date +%j)

all: nat_stim simulated simulated_noise_corr corr_mat
	@echo "Data generated."

.PHONY: nat_stim simulated simulated_noise_corr corr_mat clean_all

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
	@echo "Generated simulated data."

simulated/retina_64_rf_64: nat_stim
	make -C simulated input_size=64 rf_size=64 seed=$(seed) retina_64_rf_64

simulated/retina_128_rf_64: nat_stim
	make -C simulated input_size=128 rf_size=64 seed=$(seed) retina_128_rf_64

simulated/retina_256_rf_64: nat_stim
	make -C simulated input_size=256 rf_size=64 seed=$(seed) retina_256_rf_64

clean_simulated:
	make -C simulated clean

### simulated_noise_corr

simulated_noise_corr: simulated_noise_corr/retina_64_rf_64_nc_0.5 simulated_noise_corr/retina_128_rf_64_nc_0.5 simulated_noise_corr/retina_256_rf_64_nc_0.5
	@echo "Generated simulated_noise_corr data."

## noise_corr 0.5
simulated_noise_corr/retina_64_rf_64_nc_0.5: nat_stim
	make -C simulated_noise_corr input_size=64 rf_size=64 noise_corr=0.5 seed=$(seed) retina_64_rf_64_nc_0.5

simulated_noise_corr/retina_128_rf_64_nc_0.5: nat_stim
	make -C simulated_noise_corr input_size=128 rf_size=64 noise_corr=0.5 seed=$(seed) retina_128_rf_64_nc_0.5

simulated_noise_corr/retina_256_rf_64_nc_0.5: nat_stim
	make -C simulated_noise_corr input_size=256 rf_size=64 noise_corr=0.5 seed=$(seed) retina_256_rf_64_nc_0.5

clean_simulated_noise_corr:
	make -C simulated_noise_corr clean

### corr_mat

corr_mat: corr_mat/retina corr_mat/retina_nc_0.5
	@echo "Computed all correlation matrices."

## simulated retina
corr_mat/retina: corr_mat/retina_64_rf_64 corr_mat/retina_128_rf_64 corr_mat/retina_256_rf_64
	@echo "Computed correlation matrices for simulated retina."

corr_mat/retina_64_rf_64: simulated/retina_64_rf_64
	make -C corr_mat input_size=64 rf_size=64 retina_64_rf_64

corr_mat/retina_128_rf_64: simulated/retina_128_rf_64
	make -C corr_mat input_size=128 rf_size=64 retina_128_rf_64

corr_mat/retina_256_rf_64: simulated/retina_256_rf_64
	make -C corr_mat input_size=256 rf_size=64 retina_256_rf_64

## simulated retina with noise correlation 0.5
#corr_mat/retina_nc_0.5: corr_mat/retina_64_rf_64_nc_0.5 corr_mat/retina_128_rf_64_nc_0.5 corr_mat/retina_256_rf_64_nc_0.5
corr_mat/retina_nc_0.5: corr_mat/retina_64_rf_64_nc_0.5 corr_mat/retina_128_rf_64_nc_0.5
	@echo "Computed correlation matrices for simulated retina with noise corr 0.5."

corr_mat/retina_64_rf_64_nc_0.5: simulated_noise_corr/retina_64_rf_64_nc_0.5
	make -C corr_mat input_size=64 rf_size=64 noise_corr=0.5 retina_64_rf_64_nc_0.5

corr_mat/retina_128_rf_64_nc_0.5: simulated_noise_corr/retina_128_rf_64_nc_0.5
	make -C corr_mat input_size=128 rf_size=64 noise_corr=0.5 retina_128_rf_64_nc_0.5

corr_mat/retina_256_rf_64_nc_0.5: simulated_noise_corr/retina_256_rf_64_nc_0.5
	make -C corr_mat input_size=256 rf_size=64 noise_corr=0.5 retina_256_rf_64_nc_0.5

## experimental data from prentice_et_al_2016
corr_mat/prentice_et_al_2016: prentice_et_al_2016
	make -C corr_mat prentice_et_al_2016

clean_corr_mat:
	make -C corr_mat clean


clean_all: clean_nat_stim clean_simulated clean_prentice_et_al_2016
	@echo "Cleaned all data."
