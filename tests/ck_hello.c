#include <lux.h>
#include <lux/job.h>
#include <lux/mangle.h>
#include <lux/zalloc.h>

static int
conf(Lux_job *ego, const char *restrict opts)
{
	lux_print("conf(%p, %p);\n", ego, opts);
	return 0;
}

static int
init(Lux_job *ego)
{
	lux_print("init(%p);\n", ego);
	return 0;
}

static int
exec(Lux_job *ego)
{
	lux_print("exec(%p);\n", ego);
	lux_print("\n");
	lux_print("hello, world\n");
	return 0;
}

void *
LUX_MKMOD(const void *opts)
{
	Lux_job *ego;

	ego = zalloc(sizeof(Lux_job));
	if(ego) {
		ego->conf = conf;
		ego->init = init;
		ego->exec = exec;
	}
	return ego;
}

void
LUX_RMMOD(void *ego)
{
	lux_debug("GRay2: destructing instance %p\n", ego);

	free(ego);
}
