mat-vecO0 : mat-vec.c
	gcc -O0  mat-vec.c -o mat-vecO0

mat-vecO1 : mat-vec.c
	gcc -O1  mat-vec.c -o mat-vecO1

mat-vecO2 : mat-vec.c
	gcc -O2  mat-vec.c -o mat-vecO2

mat-vecO3 : mat-vec.c
	gcc -O3  mat-vec.c -o mat-vecO3

mat-vecOfast : mat-vec.c
	gcc -Ofast  mat-vec.c -o mat-vecOfast

mat-vec-manual-vectorizationO0: mat-vec-manual-vectorization.c
	gcc -o mat-vec-manual-vectorizationO0 mat-vec-manual-vectorization.c  -O0 -std=c11 -mavx2

mat-vec-manual-vectorizationO1: mat-vec-manual-vectorization.c
	gcc -o mat-vec-manual-vectorizationO1 mat-vec-manual-vectorization.c  -O1 -std=c11 -mavx2

mat-vec-manual-vectorizationO2: mat-vec-manual-vectorization.c
	gcc -o mat-vec-manual-vectorizationO2 mat-vec-manual-vectorization.c  -O2 -std=c11 -mavx2

mat-vec-manual-vectorizationO3: mat-vec-manual-vectorization.c
	gcc -o mat-vec-manual-vectorizationO3 mat-vec-manual-vectorization.c  -O3 -std=c11 -mavx2

mat-vec-parO0: mat-vec-par.c
	gcc -fopenmp mat-vec-par.c -O0  -o mat-vec-parO0

mat-vec-parO1: mat-vec-par.c
	gcc -fopenmp mat-vec-par.c -O1  -o mat-vec-parO1

mat-vec-parO2: mat-vec-par.c
	gcc -fopenmp mat-vec-par.c -O2  -o mat-vec-parO2

mat-vec-parO3: mat-vec-par.c
	gcc -fopenmp mat-vec-par.c -O3  -o mat-vec-parO3

mat-vec-par-vectO0 : mat-vec-par-vect.c
	gcc -fopenmp -O0 -march=native mat-vec-par-vect.c -o mat-vec-par-vectO0

mat-vec-par-vectO1 : mat-vec-par-vect.c
	gcc -fopenmp -O1 -march=native mat-vec-par-vect.c -o mat-vec-par-vectO1

mat-vec-par-vectO2 : mat-vec-par-vect.c
	gcc -fopenmp -O2 -march=native mat-vec-par-vect.c -o mat-vec-par-vectO2

mat-vec-par-vectO3 : mat-vec-par-vect.c
	gcc -fopenmp -O3 -march=native mat-vec-par-vect.c -o mat-vec-par-vectO3

clean : 
	rm -f mat-vecO0 mat-vecO1 mat-vecO2 mat-vecO3 mat-vec-Ofast mat-vec-manual-vectorizationO0 mat-vec-manual-vectorizationO1 mat-vec-manual-vectorizationO2 mat-vec-manual-vectorizationO3 mat-vec-parO0 mat-vec-parO1 mat-vec-parO2 mat-vec-parO3 mat-vec-par-vectO0 mat-vec-par-vectO1 mat-vec-par-vectO2 mat-vec-par-vectO3
