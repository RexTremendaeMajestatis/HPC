#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include "time.h"

size_t index(size_t i, size_t j, size_t cols)
{
    return (i * cols) + j;
}

double *get_matrix(size_t rows, size_t cols, double val)
{
    size_t length = rows * cols;
    double *A = (double *)malloc(length * sizeof(double));
    for (size_t i = 0; i < length; i++)
    {
        A[i] = val + i;
    }

    return A;
}

double *get_vector(size_t length, double val)
{
    double *v = (double *)malloc(length * sizeof(double));
    for (size_t i = 0; i < length; i++)
    {
        v[i] = val + i;
    }

    return v;
}

void print_matrix(double *A, size_t rows, size_t cols)
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

void print_vector(double *v, size_t length)
{
    for (size_t i = 0; i < length; i++)
    {
        printf("%f ", v[i]);
    }
    printf("\n\r");
}

size_t rows = 3000;
size_t cols = 3000;

int main(int argc, char *argv[])
{
    int process_cnt;
    int process_rank;
    MPI_Win win;
    MPI_Win matrix_window;
    MPI_Win result_window;
    MPI_Win vector_window;
    int *data;
    double *A;
    double *v;
    double *result;
    clock_t begin;
    clock_t end;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &process_cnt);
    MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);

    // Подготовка к вычислениям: хотим чтобы входные данные находились только в нулевом ранге
    // Как сделать так, чтобы окно создалось ТОЛЬКО для нулевого процесса - я не разобрался
    if (process_rank == 0)
    {
        A = get_matrix(rows, cols, 1.0);
        v = get_vector(cols, 1.0);
    }
    else
    {
        A = (double *)calloc(rows * cols, sizeof(double));
        v = (double *)calloc(cols, sizeof(double));
    }
    result = (double *)calloc(cols, sizeof(double));

    MPI_Win_create(A, rows * cols * sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &matrix_window);
    MPI_Win_fence(0, matrix_window);

    MPI_Win_create(v, cols * sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &vector_window);
    MPI_Win_fence(0, vector_window);

    MPI_Win_create(result, cols * sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &result_window);
    MPI_Win_fence(0, result_window);

    // Получили порцию матрицы для обработки
    if (process_rank == 0)
    {
        begin = clock();
    }
    size_t process_rows_cnt = (process_rank == process_cnt - 1) ? (rows / process_cnt) + (rows % process_cnt) : (rows / process_cnt);
    size_t portion_offset = process_rank * (rows / process_cnt) * cols;
    double *A_portion = (double *)calloc(process_rows_cnt * cols, sizeof(double));
    MPI_Get(A_portion, process_rows_cnt * cols, MPI_DOUBLE, 0, portion_offset, process_rows_cnt * cols, MPI_DOUBLE, matrix_window);
    MPI_Win_fence(0, matrix_window);

    // Получили вектор для скалярного умножения
    double *v_local = (double *)calloc(cols, sizeof(double));
    MPI_Get(v_local, cols, MPI_DOUBLE, 0, 0, cols, MPI_DOUBLE, vector_window);
    MPI_Win_fence(0, vector_window);

    // Получили часть результирующего вектора
    double *result_local = (double *)calloc(process_rows_cnt, sizeof(double));
    for (size_t row = 0; row < process_rows_cnt; row++)
    {
        for (size_t col = 0; col < cols; col++)
        {
            result_local[row] += v_local[col] * A_portion[row * cols + col];
        }
    }

    // Записали части вектора в результат
    size_t result_offset = process_rank * (rows / process_cnt);
    MPI_Put(result_local, process_rows_cnt, MPI_DOUBLE, 0, result_offset, process_rows_cnt, MPI_DOUBLE, result_window);
    MPI_Win_fence(0, result_window);

    if (process_rank == 0)
    {
        end = clock();
        double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
        printf("[size: %d x %d, threads: %d, time: %f]\n\r", rows, cols, process_cnt, time_spent);
        // print_vector(result, rows);
    }

    MPI_Win_free(&matrix_window);
    MPI_Win_free(&vector_window);
    MPI_Win_free(&result_window);
    MPI_Finalize();
    return 0;
}