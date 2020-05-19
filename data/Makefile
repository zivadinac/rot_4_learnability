seed?=$(shell date +%Y)$(shell date +%j)

all: nat_stim simulated
	echo "Data generated."

.PHONY: nat_stim simulated clean

nat_stim:
	make -C nat_stim

simulated: simulated/retina_64_rf_64 simulated/retina_128_rf_64 simulated/retina_256_rf_64
	echo "Generated simulated data."

simulated/retina_64_rf_64: nat_stim
	make -C simulated input_size=64 rf_size=64 seed=$(seed)

simulated/retina_128_rf_64: nat_stim
	make -C simulated input_size=128 rf_size=64 seed=$(seed)

simulated/retina_256_rf_64: nat_stim
	make -C simulated input_size=256 rf_size=64 seed=$(seed)

clean:
	make -C simulated clean
	make -C nat_stim clean
