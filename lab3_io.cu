#include "lab3_io.h"

void read_matrix (const char* input_filename, int* M, int* N, double** D){
	FILE *fin = fopen(input_filename, "r");
	int i;

	fscanf(fin, "%d%d", M, N);
	
	int num_elements = (*M) * (*N);
	*D = (double*) malloc(sizeof(double)*(num_elements));
	
	for (i = 0; i < num_elements; i++){
		fscanf(fin, "%lf", (*D + i));
	}
	fclose(fin);
}

void write_result (int M, 
		int N, 
		double* D, 
		double* U, 
		double* SIGMA, 
		double* V_T,
		int K, 
		double* D_HAT,
		double computation_time){
	// Will contain output code

	// printf("---------------------------------------------------------------------------------------------\n");
	for(int i=0;i<M;i++){
		for(int j=0;j<(M);j++){
			printf("%f |", V_T[i*M + j]);
		}
		printf("\n");
	}
	// printf("---------------------------------------------------------------------------------------------\n");

	// printf("%d\n", K);
	// printf("Time taken: %f\n", computation_time);




}
