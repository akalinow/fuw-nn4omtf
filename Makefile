install-cpu:
	pip install -e . --global-option="--use-cpu"

install-gpu:
	pip install -e . --install-option="--use-gpu"

clean:
	find . -name "__pycache__" -delete
	rm -r *-egg-info

.PHONY: clean
