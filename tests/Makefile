build: sample.la ck_sample

check: build
	for ck in $$(ls -1 ck_* | grep -v '\.'); do ./$$ck --debug; done

clean:
	for f in $$(ls *.rap  2> /dev/null); do rm -f $${f%.rap}_rap.h;   done
	for g in $$(ls *.opts 2> /dev/null); do rm -f $${g%.opts}_opts.h; done
	rm -fr .libs _libs
	rm -f  *.{lo,so,la}                  # remove modules
	rm -f  $$(ls -1 ck_* | grep -v '\.') # remove binaries

ck_%: %_rap.h ck_%.c
	lux-build $^ -o $@

%.la: %_rap.h %.c
	lux-build $^ -o $@

%_rap.h: %.rap
	lux-rapgen $<

%_opts.h: %.opts
	lux-optgen $<
