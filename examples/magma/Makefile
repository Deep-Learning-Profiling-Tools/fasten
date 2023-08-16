CC=nvcc

all: main

main: main.cu
	$(CC) -lmagma -o $@ $<

clean:
	rm main
