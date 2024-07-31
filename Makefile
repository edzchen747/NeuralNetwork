THISDIR = $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
MAKEDIR = $(THISDIR)
default: all

all: 
	rm -f main
	gcc main.c data_format.c neural_network.c -std=c99 -lm -o main
	./main


clean:
	rm -f main
	
run:
	./main

compile:
	gcc -o main main.c
