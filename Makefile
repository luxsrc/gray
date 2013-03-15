CUDA = $(subst /bin/nvcc,,$(shell which nvcc))
NVCC = $(CUDA)/bin/nvcc
ONI  = /usr/local/OpenNI/OpenNI-2.1.0
NITE = /usr/local/NiTE/NiTE-2.0.0

ifeq ($(DEBUG),1) # use `make <prob> DEBUG=1` to enable debug messages
	CPPFLAGS += -DEBUG
endif

ifeq ($(DETAILS),1) # use `make <prob> DETAILS=1` to show the ptxas info
	CFLAGS += -v --ptxas-options=-v
endif

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

ifneq ($(IO),0) # use `make <prob> IO=0` to disable IO
	CPPFLAGS += -DUMP
endif

ifeq ($(wildcard $(CUDA)/lib64/libcuda*),)
	LDFLAGS += $(addprefix -Xlinker ,-rpath $(CUDA)/lib)
else
	LDFLAGS += $(addprefix -Xlinker ,-rpath $(CUDA)/lib64)
endif

CPPFLAGS += -Isrc/$@ -I$(NITE)/Include -I$(ONI)/Include
LDFLAGS  += -L. -lNiTE2 -lOpenNI2
CFLAGS   += $(addprefix --compiler-options ,-Wall) -m64 -O3

help:
	@echo 'The follow problems are avilable:'
	@echo
	@c=0; for F in src/*/; do  \
	   f=$${F##src/};          \
	   c=`expr $$c + 1`;       \
	   echo "  $$c. $${f%%/}"; \
	 done
	@echo
	@echo "\
Use \`make <prob> [DEBUG=1] [DETAILS=1] [DOUBLE=1] [GL=0] [IO=0]\` and\n\
\`bin/GRay-<prob>\` to compile and run GRay.  The option DEBUG=1 turns on\n\
debugging messages, DETAILS=1 prints ptxas information, DOUBLE=1 enforces\n\
double-precision, while GL=0 disables OpenGL and IO=0 disables IO.\n\
\n\
To compile and link with OpenNI and NiTE, one needs to set the paths for\n\
them at the beginning of the \"Makefile\" and copy the files in Redist/ to\n\
the working directory."

%:
	@if [ ! -d src/$@ ]; then                                \
	   echo 'The problem "$@" is not available.';            \
	   echo 'Use `make help` to obtain a list of problems.'; \
	   false;                                                \
	 fi

	@mkdir -p bin
	@echo -n 'Compiling $@... '
	@$(NVCC) src/*.{cu,cc} $(CPPFLAGS) $(LDFLAGS) $(CFLAGS) -o bin/GRay-$@
	@if [ -f bin/GRay-$@ ]; then                     \
	   echo 'DONE.  Use `bin/GRay-$@` to run GRay.'; \
	 else                                            \
	   echo 'FAIL!!!';                               \
	 fi

clean:
	@echo -n 'Clean up... '
	@for F in src/*/; do        \
	   f=$${F##src/};           \
	   rm -f bin/GRay-$${f%%/}; \
	 done
	@rm -f src/*~ src/*/*~
	@if [ -z "`ls bin 2>&1`" ]; then rmdir bin; fi
	@echo 'DONE'
