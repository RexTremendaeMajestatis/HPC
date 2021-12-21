#include <mpi.h>
#include <stdio.h>
#include <cstdlib>
#include <string.h>
#include "time.h"

size_t index(size_t i, size_t j, size_t cols)
{
    return (i * cols) + j;
}

double* get_matrix(size_t rows, size_t cols, double val)
{
    size_t length = rows * cols;
    double* A = (double*)malloc(length * sizeof(double));
    for (size_t i = 0; i < length; i++)
    {
        A[i] = val + i;
    }

    return A;
}

double* get_vector(size_t length, double val)
{
    double* v = (double*)malloc(length * sizeof(double));
    for (size_t i = 0; i < length; i++)
    {
        v[i] = val + i;
    }

    return v;
}

void print_matrix(double* A, size_t rows, size_t cols)
{
    for (size_t i = 0; i < rows; i++)
    {
        for (size_t j = 0; j < cols; j++)
        {
            printf("%f ", A[index(i, j, cols)]);
        }
        printf("\n\r");
    }
    printf("\n\r");
}

void print_vector(double* v, size_t length)
{
    for (size_t i = 0; i < length; i++)
    {
        printf("%f ", v[i]);
    }
    printf("\n\r");
}

double* concat(double* a, size_t a_size, double* b, size_t b_size)
{
    double* res = (double*)malloc((a_size + b_size) * sizeof(double));
    for (size_t i = 0; i < a_size; i++)
    {
        res[i] = a[i];
    }
    for (size_t i = 0; i < b_size; i++)
    {
        res[a_size + i] = b[i];
    }

    return res;
}

size_t cols = 3000, rows = 3000;

int main(int argc, char* argv[])
{
    int ProcNum, ProcRank, tmp;
    MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);
    MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
    if (ProcRank == 0)
    {
        double* A = get_matrix(rows, cols, 1.0);
        double* v = get_vector(cols, 1.0);
        double* result = (double*)malloc(cols * sizeof(double));
        clock_t begin = clock();
        for (size_t row = 0; row < rows; row++)
        {
            size_t proc = (row % (ProcNum - 1)) + 1;
            double *to_send = concat(A + (cols * row), cols, v, cols);
            //MPI_Send(A + (cols * row), cols, MPI_DOUBLE, proc, 0, MPI_COMM_WORLD);
            MPI_Send(to_send, 2 * cols, MPI_DOUBLE, proc, 0, MPI_COMM_WORLD);
        }

        for (size_t row = 0; row < rows; row++)
        {
            double res = 0;
            size_t proc = (row % (ProcNum - 1)) + 1;
            MPI_Recv(&res, 1, MPI_DOUBLE, proc, row, MPI_COMM_WORLD, &status);
            result[row] = res;
        }
        clock_t end = clock();
        double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
        printf("[size: %d x %d, threads: %d, time: %f]\n\r", rows, cols, ProcNum, time_spent);
    }
    else
    {
        int compute_processes = ProcNum - 1;
        int count = ProcRank <= (rows % compute_processes) ? (rows / compute_processes) + 1 : (rows / compute_processes);
        for (int i = 0; i < count; i++)
        {
            size_t row = (compute_processes * i + ProcRank) - 1;
            double* buf = (double*)malloc(2 * cols * sizeof(double));
            MPI_Recv(buf, 2 * cols, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
            double res = 0;
            for (size_t i = 0; i < cols; i++)
            {
                res += buf[i] * buf[i + cols];
            }

            MPI_Send(&res, 1, MPI_DOUBLE, 0, row, MPI_COMM_WORLD);
        }
    }
    MPI_Finalize();
    return 0;
}