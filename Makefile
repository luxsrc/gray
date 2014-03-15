CUDA_PATH = $(subst /bin/nvcc,,$(shell which nvcc))
NITE_PATH = /usr/local/NiTE/2.2
OPNI_PATH = /usr/local/OpenNI/2.2
LEAP_PATH = /usr/local/LeapSDK

NVCC = $(CUDA_PATH)/bin/nvcc

ifeq ($(DEBUG),1) # use `make <prob> DEBUG=1` to enable debug messages
	CPPFLAGS += -DEBUG
endif

ifeq ($(DETAILS),1) # use `make <prob> DETAILS=1` to show the ptxas info
	CFLAGS += -v --ptxas-options=-v
endif

ifeq ($(DOUBLE),1)      # use `make <prob> DOUBLE=1` for double precision
	CPPFLAGS += -DOUBLE=1
	CFLAGS   += -arch sm_13
else ifeq ($(SINGLE),1) # use `make <prob> SINGLE=1` for single precision
	CPPFLAGS += -DOUBLE=0
else                    # mixed precision otherwise
	CFLAGS   += -arch sm_13
endif

ifeq ($(GL),0) # use `make <prob> GL=0` to disable OpenGL visualization
	CPPFLAGS += -DISABLE_GL
else
	OPT += src/optional/{ctrl,shaders,texture,vis}.cc
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

ifneq ($(NITE),1) # use `make <prob> NITE=1 to enable natural interaction
	CPPFLAGS += -DISABLE_NITE
else
	CPPFLAGS += -I$(NITE_PATH)/Include \
	            -I$(OPNI_PATH)/Include
	LDFLAGS  += -L$(NITE_PATH)/Redist  \
	            -L$(OPNI_PATH)/Redist  \
	            -lNiTE2 -lOpenNI2
endif

ifneq ($(LEAP),1) # use `make <prob> LEAP=1 to enable natural interaction
	CPPFLAGS += -DISABLE_LEAP
else
	CPPFLAGS += -I$(LEAP_PATH)/include
	LDFLAGS  += -L$(LEAP_PATH)/lib -lLeap
endif

ifeq ($(wildcard $(CUDA)/lib64/libcuda*),)
	LDFLAGS += $(addprefix -Xlinker ,-rpath $(CUDA)/lib)
else
	LDFLAGS += $(addprefix -Xlinker ,-rpath $(CUDA)/lib64)
endif

CPPFLAGS += -Isrc/$@
CFLAGS   += $(addprefix --compiler-options ,-Wall) -m64 -O3

help:
	@echo 'The follow problems are avilable:'
	@echo
	@c=0; for F in src/[[:upper:]]*/; do \
	   f=$${F##src/};                    \
	   c=`expr $$c + 1`;                 \
	   echo "  $$c. $${f%%/}";           \
	 done
	@echo
	@echo "\
Use \`make <prob> [DEBUG=1] [DETAILS=1] [DOUBLE/SINGLE=1] [GL=0] [IO=0]\`\n\
and \`bin/GRay-<prob>\` to compile and run GRay.  The option DEBUG=1 turns\n\
on debugging messages, DETAILS=1 prints ptxas information, DOUBLE=1 enforces\n\
double-precision, while GL=0 disables OpenGL and IO=0 disables IO.\n\
\n\
To enable PrimeSense or LeapMotion sensor, one needs to pass in the flag\n\
NITE=1 or LEAP=1 and manually set the header and library search paths at\n\
the beginning of this \"Makefile\"."

%:
	@if [ ! -d src/$@ ]; then                                \
	   echo 'The problem "$@" is not available.';            \
	   echo 'Use `make help` to obtain a list of problems.'; \
	   false;                                                \
	 fi

	@mkdir -p bin
	@echo -n 'Compiling $@... '
	@$(NVCC) src/*.{cu,cc} $(OPT) $(CPPFLAGS) $(LDFLAGS) $(CFLAGS) \
	   -o bin/GRay-$@

ifeq ($(NITE),1)
	@install_name_tool -change libNiTE2.dylib \
	       $(NITE_PATH)/Redist/libNiTE2.dylib \
	       bin/GRay-Kerr
	@install_name_tool -change libOpenNI2.dylib \
	       $(OPNI_PATH)/Redist/libOpenNI2.dylib \
	       bin/GRay-Kerr
endif

ifeq ($(LEAP),1)
	@install_name_tool -change @loader_path/libLeap.dylib \
	                       $(LEAP_PATH)/lib/libLeap.dylib \
	                       bin/GRay-Kerr
endif

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
