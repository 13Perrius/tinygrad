.ONESHELL:
SHELL := /bin/sh

.PHONY: clean_pickles clean_benchmarks clean_dumps clean run new_run test

PYTHON := venv/bin/python
BDIR := rangeify_perf

clean_pickles:
	rm -f ${BDIR}/pickles/*

clean_benchmarks:
	rm -f ${BDIR}/benchmarks/*

clean_dumps:
	rm -f ${BDIR}/dumps/*

clean:
	gmake clean_pickles clean_benchmarks 

run:
	if [ "$${OUT}" != "" ]
	then
	  touch ${BDIR}/dumps/$${OUT}
	  ${PYTHON} ${BDIR}/run_benchmark.py > ${BDIR}/dumps/$${OUT}
	else
	  ${PYTHON} ${BDIR}/run_benchmark.py
	fi

new_run:
	gmake clean
	gmake run

