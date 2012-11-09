CUDA = $(subst /bin/nvcc,,$(shell which nvcc))
NVCC = $(CUDA)/bin/nvcc

ifneq ($(wildcard $(CUDA)/lib64/libcuda*),)
	RPATH = $(CUDA)/lib64
else
	RPATH = $(CUDA)/lib
endif

CPPFLAGS = -Isrc/$@

ifneq ($(shell uname),Darwin)
	LDFLAGS = $(addprefix -Xlinker ,\
	            -rpath $(RPATH)) -lglut -lglu -lgl
else
	LDFLAGS = $(addprefix -Xlinker ,\
	            -rpath $(RPATH) -framework Glut -framework OpenGL)
endif

CFLAGS   = $(addprefix --compiler-options ,-Wall) -O3

help:
	@echo 'The follow settings are avilable:'
	@echo
	@c=0; for F in src/*/; do  \
	   f=$${F##src/};          \
	   c=`expr $$c + 1`;       \
	   echo "  $$c. $${f%%/}"; \
	 done
	@echo
	@echo 'Use `make NAME` and `bin/NAME` to compile and run geode.'

%:
	@if [ ! -d src/$@ ]; then                                \
	   echo 'The setting "$@" is not available.';            \
	   echo 'Use `make help` to obtain a list of settings.'; \
	   false;                                                \
	 fi

	@mkdir -p bin
	@echo -n 'Compiling... '
	@$(NVCC) src/*.{cu,cpp} $(CPPFLAGS) $(CFLAGS) $(LDFLAGS) -o bin/$@
	@if [ -f bin/$@ ]; then              \
	   echo 'DONE';                      \
	   echo 'Use `bin/$@` to run geode'; \
	 else                                \
	   echo 'FAIL!!!';                   \
	 fi

clean:
	@echo -n 'Clean up... '
	@for F in src/*/; do   \
	   f=$${F##src/};      \
	   rm -f bin/$${f%%/}; \
	 done
	@rm -f src/*~ src/*/*~
	@if [ -z "`ls bin 2>&1`" ]; then rmdir bin; fi
	@echo 'DONE'
