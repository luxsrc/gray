SUBDIRS = tests sim doc

ifeq ($(MAKECMDGOALS),)
	GOALS = all
else
	GOALS = $(MAKECMDGOALS)
endif

all: recursive

check: recursive

install: recursive

clean: recursive

recursive:
	list='$(SUBDIRS)'; goals='$(GOALS)'; for subdir in $$list; do \
	  test "$$subdir" = . || (cd $$subdir && make $$goals); \
	done
