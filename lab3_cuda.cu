#include "lab3_cuda.h"
#include <iostream>
#include <cmath>
#include <malloc.h>
#include <fstream>
#include <bits/stdc++.h>
#include <cuda.h>


#define pb push_back

using namespace std;

#define TOLERANCE 0.001
#define JACOBI_UPDATE_TOLERANCE 0.001
#define FILENAME1 "testcase_1000_300"
#define FILENAME2 "iris_stndardized"
#define samples 150
#define features 4
#define BLOCK_SIZE 16

double **S; //Symmetric matrix (input)
double  *e; //eigenvalues
double **E; //eigenvectors
int  *ind;
bool *changed;
int  state;
int  N;

void read_file(char* filename, int num_samples, int num_features, double** A) {
    ifstream ifile;
    ifile.open(filename, ios::in);

    double tmp;
    for (int i=0; i<num_samples; i++) {
        for (int j=0; j<num_features; j++){
            ifile >> tmp;
            A[i][j] = tmp;
        }
    }

    ifile.close();
}

double** mat_transpose(double** A, int Am, int An) {
    double **B;
    B = (double**)malloc(__SIZEOF_POINTER__*An);
    for (int i=0; i<An; i++)
        B[i] = (double*)malloc(__SIZEOF_DOUBLE__*Am);

    for (int i=0; i<Am; i++){
        for (int j=0; j<An; j++){
            B[j][i] = A[i][j];
        }
    }

    return B;
}

double** mat_mul(double** A, int Am, int An, 
                 double** B, int Bm, int Bn){
    double **C;
    C = (double**)malloc(__SIZEOF_POINTER__*Am);
    for (int i=0; i<Am; i++)
        C[i] = (double*)malloc(__SIZEOF_DOUBLE__*Bn);

    for (int i=0; i<Am; i++){
        for (int j=0; j<Bn; j++){
            C[i][j] = 0;
            for (int k=0; k<An; k++){
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    return C;
}

// dim3 dimGrid((k + BLOCK_SIZE - 1) / BLOCK_SIZE, (m + BLOCK_SIZE - 1) / BLOCK_SIZE);
// dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);


int maxind(int k) {
    int m = k+1;

    for (int i = k+2; i < N; i++){
        if (fabs(S[k][i]) > fabs(S[k][m])){
            m = i;
        }
    }

    return m;
}

void update(int k, double t) {
    double ek_prev = e[k];
    e[k] = ek_prev + t;

    if (e[k] < 0) e[k] = 0;

    if (changed[k] && (ek_prev - e[k]) < JACOBI_UPDATE_TOLERANCE) {
        changed[k] = false;
        state = state - 1;
    }
    else if ((! changed[k]) && (ek_prev - e[k]) > JACOBI_UPDATE_TOLERANCE) {
        changed[k] = true;
        state = state + 1;
    }
}

void rotate(int k, int l, int i, int j, double c, double s,
            bool eigenvectors){
    double** mat1;
    double** mat2;
    double** mat3;

    mat1 = (double**)malloc(__SIZEOF_POINTER__*2);
    mat1[0] = (double*)malloc(__SIZEOF_DOUBLE__*2);
    mat1[1] = (double*)malloc(__SIZEOF_DOUBLE__*2);
    mat1[0][0] = c; mat1[0][1] = -s;
    mat1[1][0] = s; mat1[1][1] = c;

    mat2 = (double**)malloc(__SIZEOF_POINTER__*2);
    mat2[0] = (double*)malloc(__SIZEOF_DOUBLE__*1);
    mat2[1] = (double*)malloc(__SIZEOF_DOUBLE__*1);
    if (eigenvectors){
        mat2[0][0] = E[i][k];
        mat2[1][0] = E[i][l];
    }
    else {
        mat2[0][0] = S[k][l];
        mat2[1][0] = S[i][j];
    }

    mat3 = mat_mul(mat1, 2, 2, mat2, 2, 1);

    if (eigenvectors){
        E[i][k] = mat3[0][0];
        E[i][l] = mat3[1][0];
    }
    else{
        S[k][l] = mat3[0][0];
        S[i][j] = mat3[1][0];
    }

    free(mat1[0]);
    free(mat1[1]);
    free(mat1);
    free(mat2[0]);
    free(mat2[1]);
    free(mat2);
    free(mat3[0]);
    free(mat3[1]);
    free(mat3);
}

void print_matrix(double** A, int Am, int An) {
    cout << "[";
    for (int i=0; i<Am; i++){
        if (i>0)
            cout<<" ";
        cout<<"[";
        for (int j=0; j<An-1; j++){
            cout << A[i][j] << ", ";
        }
        if (i < Am-1)
            cout << A[i][An-1] << "]" << endl;
    }
    cout << A[Am-1][An-1] << "]]" << endl;
}

void print_vector(double* A, int An) {
    cout << "[";
    for(int i=0; i<An-1; i++)
        cout << A[i] << ",";
    cout << A[An-1] << "]" << endl;
}

void init_jacobi() {
    E = (double**)malloc(__SIZEOF_POINTER__*N);
    for (int i=0; i<N; i++){
        E[i] = (double*)malloc(__SIZEOF_DOUBLE__*N);
        for (int j=0; j<N; j++){
            E[i][j] = 0;
        }
        E[i][i] = 1;
    }

    state = N;

    e = (double*)malloc(__SIZEOF_DOUBLE__*N);
    ind = (int*)malloc(__SIZEOF_INT__*N);
    changed = (bool*)malloc(sizeof(bool)*N);

    for (int k=0; k<N; k++){
        ind[k]     = maxind(k);
        e[k]       = S[k][k];
        changed[k] = true;
    }
}

void Jacobi(double **input_matrix, int n, 
            double **eigenvalues, double ***eigenvectors) {
    N = n;
    S = input_matrix;

    init_jacobi();

    while(state != 0){
        int m = 0;

        for (int k=1; k<N-1; k++){
            if (fabs(S[k][ind[k]]) > fabs(S[m][ind[m]])){
                m = k;
            }
        }

        int k = m;
        int l = ind[m];
        double p = S[k][l];
        double y = (e[l] - e[k]) / 2.0;
        double d = fabs(y) + sqrt(p*p + y*y);
        double r = sqrt(p*p + d*d);
        double c = d / r;
        double s = p / r;
        double t = (p*p) / d;

        if (y < 0.0) { s = -s; t = -t; }

        S[k][l] = 0.0;
        update(k, -t);
        update(l, t);

        for (int i=0; i<k; i++)  { rotate(i, k, i, l, c, s, false); }
        for (int i=k+1; i<l; i++){ rotate(k, i, i, l, c, s, false); }
        for (int i=l+1; i<N; i++)  { rotate(k, i, l, i, c, s, false); }

        for (int i=0; i<N; i++){
            rotate(k, l, i, i, c, s, true);
        }

        ind[k] = maxind(k);
        ind[l] = maxind(l);
    }

    *eigenvalues = e;
    *eigenvectors = E;
}

// int main(){
//     double **D, **D_T;
//     double **prod, *eigenvalues, **eigenvectors;

//     D = (double**)malloc(sizeof(double*)*samples);
//     for (int i=0; i<samples; i++)
//         D[i] = (double*)malloc(sizeof(double)*features);

//     read_file((char*)FILENAME1, samples, features, D);

//     D_T = mat_transpose(D, samples, features);

//     prod = mat_mul(D_T, features, samples, D, samples, features);
//     Jacobi(prod, features, &eigenvalues, &eigenvectors);

//     cout << "\neigenvalues:" << endl;
//     print_vector(eigenvalues, features);

//     cout << "\neigenvectors:" << endl;
//     print_matrix(eigenvectors, features, features);

//     return 0;
// }

// /*
// 	*****************************************************
// 		TODO -- You must implement this function
// 	*****************************************************
// */

// void SVD_and_PCA (int M, 
void SVD_and_PCA (int M, 
        int N, 
        double* D, 
        double** U, 
        double** SIGMA, 
        double** V_T, 
        double** D_HAT, 
        int *K,
        int retention) {
    // write your code here

	double **d;
	double **d_t;
    double **product, *eigenvalues, **eigenvectors;
    // double **v;


    d = (double**)malloc(sizeof(double*)*M);
    for (int i=0; i<M; i++)
        d[i] = (double*)malloc(sizeof(double)*N);


    for(int i=0;i<M;i++){
    	for(int j=0;j<N;j++) d[i][j] = D[i*N+j];
    }

    d_t = mat_transpose(d, M, N);



    product = mat_mul(d_t, N, M, d, M, N);


    float computation_time1;
    cudaEvent_t start1, stop1;
    cudaEventCreate(&start1);
    cudaEventCreate(&stop1);

    cudaEventRecord(start1);

    Jacobi(product, N, &eigenvalues, &eigenvectors);

    cudaEventRecord(stop1);
    cudaEventSynchronize(stop1);
    cudaEventElapsedTime(&computation_time1, start1, stop1);
    printf("Time taken for Jacobi: %f\n", computation_time1);   


    // for(int i=0;i<N;i++) printf("%f\n", eigenvalues[i]);

    // vector<double> eigenvals;
    // for(int i=0; i<N; i++) eigenvals.pb(eigenvalues[i]);

    vector<pair<double, int> > eigenv_index;
	

	for(int i=0; i<N; i++){
		eigenv_index.pb(make_pair(eigenvalues[i],i));
	}

    // for(int i=0; i<N; i++){
    //     // eigenv_index[i] = {eigenvalues[i],i};
    //     eigenv_index[i] = make_pair(eigenvalues[i],i);
    //     // eigenv_index[i].first = eigenvalues[i];
    //     // eigenv_index[i].second = i;
    //     // eigenv_index[i] = make_pair(eigenvalues[i],i);
    //     cout << eigenvalues[i] << ' ' << eigenv_index[i].first << ' ' << i << ' ' << eigenv_index[i].second << endl;

    //     // printf("%f %d\n", eigenv_index[i].first, eigenv_index[i].second);

    // }

    // for(int i=0;i<N;i++) printf("%f %d\n", eigenv_index[i].first, eigenv_index[i].second);


	sort(eigenv_index.begin(), eigenv_index.end());

    // for(int i=0;i<N;i++) printf("%f\n", eigenv_index[i].first);

	int e = eigenv_index.size()-1;
	
 	for(int i=0;i<N;i++){
        // printf("%f\n", sqrt(eigenv_index[e].first));
		(*SIGMA)[i] = sqrt(eigenv_index[e].first);
		e--;
	}


    // for(int i=0;i<N;i++) printf("%f\n", (*SIGMA)[i]);

	double **u = (double**)malloc(sizeof(double*)*N);
    for (int i=0; i<N; i++)
        u[i] = (double*)malloc(sizeof(double)*N);


	e = eigenv_index.size()-1;	
	for(int j=0;j<N;j++){
		int index = eigenv_index[e].second;
		for(int i=0;i<N;i++){
			u[i][j] = eigenvectors[i][index];
		}
		e--;
	}

	for(int j=0;j<N;j++){
		for(int i=0;i<N;i++){
			(*U)[i*N+j] = u[i][j];
		}
	}


	// for(int j=0;j<N;j++){
	// 	for(int i=0;i<N;i++){
	// 		printf("%f ", (*U)[i*N+j]);
	// 	}
	// 	printf("\n");
	// }


	// size N*M
    double **sigma_invT = (double**)malloc(sizeof(double*)*N);
    for (int i=0; i<N; i++)
        sigma_invT[i] = (double*)malloc(sizeof(double)*M);

	for(int i=0; i<N; i++){
		for(int j=0; j<M; j++) sigma_invT[i][j]=0;
	}

	e = eigenv_index.size()-1;

	for(int i=0; i<N;i++){
		if(eigenv_index[e].first<1e-5){
			sigma_invT[i][i]= 0;
		}
		else{
			sigma_invT[i][i]= 1/sqrt(eigenv_index[e].first);
		}
		e--;	
	}

	double **temp = mat_mul(d, M, N, u, N, N);
	double **v = mat_mul(temp, M, N, sigma_invT, N, M);
	double **v_t = mat_transpose(v, M, M);

	// for(int i=0; i<M; i++){
	// 	for(int j=0; j<M; j++) printf("%f ", v_t[i][j]);
	// 	printf("\n");
	// }


	for(int i=0; i<M; i++){
		for(int j=0; j<M; j++) (*V_T)[i*M+j] = v_t[i][j];
	}

	// for(int i=0; i<M; i++){
	// 	for(int j=0; j<M; j++) printf("%f ", V_T[i][j]);
	// 	printf("\n");
	// }


	double num=0;
	double sigmasqsum=0;
	for(int i=0; i<N; i++){
		sigmasqsum += (*SIGMA)[i]*(*SIGMA)[i];
	}

    // printf("\n%f\n", sigmasqsum);

    int k=0;
	for(k=0; k<N; k++){
		num += ((*SIGMA)[k]*(*SIGMA)[k])/sigmasqsum;
		if(num >= retention/100.0){
			break;
		}
	}
    
    *K = k+1;

    // double **newU;
	// double **newU = (double**)malloc(sizeof(double*)*N*(k+1));
    double **newU = (double**)malloc(sizeof(double*)*N);
    for (int i=0; i<N; i++)
        newU[i] = (double*)malloc(sizeof(double)*(k+1));


    for(int i=0; i<N; i++){
    	for(int j=0;j<k+1;j++){
    		newU[i][j] = (u)[i][j];
    	}
    }


	// for(int i=0; i<N; i++){
	// 	for(int j=0; j<(k+1); j++) printf("%f ", newU[i][j]);
	// 	printf("\n");
	// }

    double **d_hat = (double**)malloc(sizeof(double*)*M);
    for (int i=0; i<(k+1); i++)
        d_hat[i] = (double*)malloc(sizeof(double)*(k+1));



    d_hat = mat_mul(d, M, N, newU, N, (k+1));

	*D_HAT = (double*) malloc(sizeof(double) * M*(k+1));


	for(int i=0; i<M; i++){
    	for(int j=0;j<k+1;j++){
    		(*D_HAT)[i*(k+1)+j] = d_hat[i][j];
    	}
    }


}

