check_MODULES = $(foreach ck, $(wildcard ck_*.c), $(basename $(ck)).la)

check: $(check_MODULES)

ck_%.la : ck_%.c
	lux-build $< -o $@
	lux $(basename $@)

clean:
	rm *.so *.lo *.la
