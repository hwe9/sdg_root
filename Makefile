PIP_COMPILE = pip-compile --resolver=backtracking
LOCK = constraints/constraints.txt

lock:
	$(PIP_COMPILE) --upgrade --generate-hashes -o $(LOCK) constraints/constraints.in

reqs: lock
	@for SVC in api auth data_processing content_extraction; do \
	  $(PIP_COMPILE) --generate-hashes -c $(LOCK) -o $$SVC/requirements.txt $$SVC/requirements.in; \
	done
