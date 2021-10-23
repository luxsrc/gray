default: doc

all:

doc: html/index.html latex/refman.pdf

check:

install:

clean:

html/index.html:
	doxygen

latex/refman.pdf:
	if [ ! -f latex/refman.tex ]; then doxygen; fi
	cd latex; pdflatex refman; pdflatex refman; cd ..

clean:
	rm -rf html/ latex/
