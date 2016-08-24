all: gray.la

install: gray.la
	lux-install *.cl gray.la +i sim

%.h: %.opts
	lux-optgen $<

gray.la: gray.h *.c icond.h param.h setup.h
	lux-build gray.h *.c -o gray.la

clean:
	-rm -fr .libs _libs
	-rm -f  *.lo gray.*so gray.la
	-rm -f  icond.h param.h setup.h
