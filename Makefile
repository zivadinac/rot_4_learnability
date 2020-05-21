.PHONY: data clean_data figures correlation_figures clean_figures clean

data:
	make -C data

clean_data:
	make -C data clean

### figures

figures: correlation_figures
	echo "Plotted all figures."

correlation_figures: data correlation_figures/retina correlation_figures/retina_nc_0.5
	echo "Plotted all correlation figures."

## simulated retina
correlation_figures/retina: correlation_figures/retina_64_rf_64 correlation_figures/retina_128_rf_64 correlation_figures/retina_256_rf_64
	@echo "Plotted correlation figures for simulated retina."

correlation_figures/retina_64_rf_64: data
	make -C figures input_size=64 rf_size=64 retina_64_rf_64

correlation_figures/retina_128_rf_64: data
	make -C figures input_size=128 rf_size=64 retina_128_rf_64

correlation_figures/retina_256_rf_64: data
	make -C figures input_size=256 rf_size=64 retina_256_rf_64

## simulated retina with noise correlation 0.5
#correlation_figures/retina_nc_0.5: correlation_figures/retina_64_rf_64_nc_0.5 correlation_figures/retina_128_rf_64_nc_0.5 correlation_figures/retina_256_rf_64_nc_0.5
correlation_figures/retina_nc_0.5: correlation_figures/retina_64_rf_64_nc_0.5 correlation_figures/retina_128_rf_64_nc_0.5
	@echo "Plotted correlation figures for simulated retina with noise correlation 0.5."

correlation_figures/retina_64_rf_64_nc_0.5: data
	make -C figures input_size=64 rf_size=64 noise_corr=0.5 retina_64_rf_64_nc_0.5

correlation_figures/retina_128_rf_64_nc_0.5: data
	make -C figures input_size=128 rf_size=64 noise_corr=0.5 retina_128_rf_64_nc_0.5

correlation_figures/retina_256_rf_64_nc_0.5: data
	make -C figures input_size=256 rf_size=64 noise_corr=0.5 retina_256_rf_64_nc_0.5

clean_figures:
	make -C figures clean

clean: clean_data
