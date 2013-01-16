ifeq ($(DOUBLE),1) # use `make <prob> DOUBLE=1` to compile in double precision
	CPPFLAGS += -DOUBLE
	CFLAGS   += -arch sm_13
endif

ifeq ($(GL),0) # use `make <prob> GL=0` to disable OpenGL visualization
	CPPFLAGS += -DISABLE_GL
else
	ifeq ($(shell uname),Darwin)
		LDFLAGS += $(addprefix -Xlinker ,\
		             -framework Glut -framework OpenGL)
	else
		LDFLAGS += -lglut -lglu -lgl
	endif
endif

ifneq ($(IO),0)
	CPPFLAGS += -DUMP
endif

CUDA = $(subst /bin/nvcc,,$(shell which nvcc))
NVCC = $(CUDA)/bin/nvcc

ifeq ($(wildcard $(CUDA)/lib64/libcuda*),)
	LDFLAGS += $(addprefix -Xlinker ,-rpath $(CUDA)/lib)
else
	LDFLAGS += $(addprefix -Xlinker ,-rpath $(CUDA)/lib64)
endif

CPPFLAGS += -Isrc/$@
CFLAGS   += $(addprefix --compiler-options ,-Wall) -O3

help:
	@echo 'The follow problems are avilable:'
	@echo
	@c=0; for F in src/*/; do  \
	   f=$${F##src/};          \
	   c=`expr $$c + 1`;       \
	   echo "  $$c. $${f%%/}"; \
	 done
	@echo
	@echo \
'Use `make <prob> [DOUBLE=1] [GL=0] [IO=0]` and `bin/<prob>` to compile and'
	@echo \
'run geode.  The option DOUBLE=1 enforces double precision, while GL=0 disable'
	@echo \
'OpenGL and IO=0 disable IO.'

%:
	@if [ ! -d src/$@ ]; then                                \
	   echo 'The problem "$@" is not available.';            \
	   echo 'Use `make help` to obtain a list of problems.'; \
	   false;                                                \
	 fi

	@mkdir -p bin
	@echo -n 'Compiling $@... '
	@$(NVCC) src/*.{cu,cc} $(CPPFLAGS) $(LDFLAGS) $(CFLAGS) -o bin/$@
	@if [ -f bin/$@ ]; then                      \
	   echo 'DONE.  Use `bin/$@` to run geode.'; \
	 else                                        \
	   echo 'FAIL!!!';                           \
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
