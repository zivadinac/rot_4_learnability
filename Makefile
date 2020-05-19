.PHONY: data clean clean_data

data:
	make -C data

clean: clean_data

clean_data:
	make -C data clean
