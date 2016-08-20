all: gray.la

gray.la: gray.h gray.c
	lux-build gray.h gray.c -o gray.la

clean:
	-rm -fr .libs _libs
	-rm -f  gray.*so gray.l?
