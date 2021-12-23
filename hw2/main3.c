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
    int process_cnt;
    int process_rank;
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &process_cnt);
    MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);

    int compute_process_cnt = process_cnt - 1;

    if (process_rank == 0)
    {
        double* A = get_matrix(rows, cols, 1.0);
        double* v = get_vector(cols, 1.0);
        double* result = (double*)malloc(cols * sizeof(double));
        clock_t begin = clock();
        for (int rank = 1; rank < process_cnt; rank++)
        {
            size_t process_first_row = (rows / compute_process_cnt) * (rank - 1);
            size_t process_rows_cnt = (rank == process_cnt - 1) ? (rows / compute_process_cnt) + (rows % compute_process_cnt) : (rows / compute_process_cnt);
            size_t to_send_cnt = process_rows_cnt * cols;

            double* to_send = A + process_first_row * cols;

            MPI_Send(v, cols, MPI_DOUBLE, rank, 0, MPI_COMM_WORLD);
            MPI_Send(to_send, to_send_cnt, MPI_DOUBLE, rank, rank, MPI_COMM_WORLD);
        }

        for (int rank = 1; rank < process_cnt; rank++)
        {
            size_t process_first_row = (rows / compute_process_cnt) * (rank - 1);
            size_t process_rows_cnt = (rank == process_cnt - 1) ? (rows / compute_process_cnt) + (rows % compute_process_cnt) : (rows / compute_process_cnt);
            MPI_Recv(result + process_first_row, process_rows_cnt, MPI_DOUBLE, rank, 0, MPI_COMM_WORLD, &status);
        }
        clock_t end = clock();
        double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
        printf("[size: %d x %d, threads: %d, time: %f]\n\r", rows, cols, compute_process_cnt, time_spent);
        //print_vector(result, cols);
    }
    else
    {
        size_t process_rows_cnt = (process_rank == process_cnt - 1) ? (rows / compute_process_cnt) + (rows % compute_process_cnt) : (rows / compute_process_cnt);
        size_t to_receive_cnt = process_rows_cnt * cols;
        size_t process_first_row = (rows / compute_process_cnt) * (process_rank - 1);
        double* v = (double*)malloc(cols * sizeof(double));
        double* process_rows = (double*)malloc(to_receive_cnt * sizeof(double));
        double* result = (double*)malloc(process_rows_cnt * sizeof(double));

        MPI_Recv(v, cols, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
        MPI_Recv(process_rows, to_receive_cnt, MPI_DOUBLE, 0, process_rank, MPI_COMM_WORLD, &status);

        for (size_t row = 0; row < process_rows_cnt; row++)
        {
            result[row] = 0.0;
            for (size_t col = 0; col < cols; col++)
            {
                result[row] += process_rows[cols * row + col] * v[col];
            }
        }

        MPI_Send(result, process_rows_cnt, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }
    MPI_Finalize();
    return 0;
}