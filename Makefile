
install-cpu:
	pip install tensorflow
	pip install -e .

install-gpu:
	pip install tensorflow-gpu
	pip install -e . 

clean:
	find . -name "__pycache__" -delete
	rm -r *-egg-info

.PHONY: clean install-cpu install-gpu
