.PHONY: data clean_data figures clean_figures clean

data:
	make -C data

clean_data:
	make -C data clean

### figures

figures: correlation_figures
	echo "Plotted all figures."

correlation_figures: data correlation_figures/retina_64_rf_64 correlation_figures/retina_128_rf_64 correlation_figures/retina_256_rf_64
	echo "Plotted all correlation figures."

correlation_figures/retina_64_rf_64: data
	make -C figures input_size=64 rf_size=64

correlation_figures/retina_128_rf_64: data
	make -C figures input_size=128 rf_size=64

correlation_figures/retina_256_rf_64: data
	make -C figures input_size=256 rf_size=64

clean_figures:
	make -C figures clean

clean: clean_data
