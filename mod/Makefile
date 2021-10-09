build: gray.la

install: gray.la
	lux-install gray.la +i sim

clean:
	for f in $$(ls *.rap  2> /dev/null); do rm -f $${f%.rap}_rap.h;   done
	for g in $$(ls *.opts 2> /dev/null); do rm -f $${g%.opts}_opts.h; done
	rm -fr {.,_}libs
	rm -f  *.{lo,so,la}

gray.la: gray.h gray.c gray_opts.h Kerr_opts.h infcam_opts.h

%.la:
	lux-build $^ -o $@

%_rap.h: %.rap
	lux-rapgen $<

%_opts.h: %.opts
	lux-optgen $<
