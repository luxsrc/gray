all: Kerr/ocl.la infcam.la gray.la

install: Kerr/ocl.la infcam.la gray.la
	lux-install Kerr/ocl.la +i sim/gray
	lux-install infcam.la   +i sim/gray
	lux-install gray.la     +i sim

clean:
	for f in $$(ls *.rap  2> /dev/null); do rm -f $${f%.rap}_rap.h;   done
	for g in $$(ls *.opts 2> /dev/null); do rm -f $${g%.opts}_opts.h; done
	rm -fr {.,_}libs
	rm -f  *.{lo,so,la}

# mod:       interfaces        sources    options etc
# v          v                 v          v
Kerr/ocl.la: Kerr_rap.h        Kerr/ocl.c Kerr_opts.h
infcam.la:   initcond.h        infcam.c   infcam_opts.h
gray.la:     gray.h initcond.h gray.c     gray_opts.h Kerr_opts.h infcam_opts.h
	lux-build $^ -f hdf5 -o $@

%.la:
	lux-build $^ -o $@

%_rap.h: %.rap
	lux-rapgen $<

%_opts.h: %.opts
	lux-optgen $<
