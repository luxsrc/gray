all: gray.la

options.h: options.otab
	lux-optgen options.otab

gray.la: gray.h gray.c options.h
	lux-build gray.h gray.c -o gray.la

clean:
	-rm -fr .libs _libs
	-rm -f  gray.*so gray.l?
	-rm -f  options.h
