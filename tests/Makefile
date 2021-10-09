check_MODULES = $(foreach ck, $(wildcard ck_*.c), $(basename $(ck)).la)

check: $(check_MODULES)

ck_%.la : MODULE = $(basename $@)
ck_%.la : ck_%.c
	lux-build $< -o $@ -i sim
	rm $(MODULE).lo $(MODULE).la
	lux $(MODULE)

clean:
	rm -rf sim
