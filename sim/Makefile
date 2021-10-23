all: gray/Kerr/ocl.la gray/infcam.la gray.la

install: gray/Kerr/ocl.la gray/infcam.la gray.la
	lux-install $^ +i sim

clean:
	for f in $$(ls *.rap  2> /dev/null); do rm -f $${f%.rap}_rap.h;   done
	for g in $$(ls *.opts 2> /dev/null); do rm -f $${g%.opts}_opts.h; done
	rm -fr {.,_}libs
	rm -f  *.{lo,so,la}

# mod:            interfaces                   sources         options etc
# v               v                            v               v
gray/Kerr/ocl.la: Kerr_rap.h                   gray/Kerr/ocl.c Kerr_opts.h
gray/infcam.la:   initcond.h                   gray/infcam.c   infcam_opts.h
gray.la:          gray.h initcond.h Kerr_rap.h gray.c          gray_opts.h Kerr_opts.h infcam_opts.h
	lux-build $^ -f hdf5 -o $@

%.la:
	lux-build $^ -o $@

%_rap.h: %.rap
	lux-rapgen $<

%_opts.h: %.opts
	lux-optgen $<
