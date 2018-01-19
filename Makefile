install:
	pip install -e .

clean:
	find . -name "__pycache__" -delete
	rm -r *-egg-info

.PHONY: init clean
