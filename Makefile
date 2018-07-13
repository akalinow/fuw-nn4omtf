
tf-cpu:
	pip install tensorflow

tf-gpu:
	pip install tensorflow-gpu

nn4omtf:
	pip install -e . 

clean:
	find . -name "__pycache__" -delete
	rm -r *-egg-info

.PHONY: clean tf-cpu tf-gpu nn4omtf
