#include "stdio.h"
#include "stdlib.h"
#include "math.h"
#include "conio.h"
#include "omp.h"
#include "time.h"


#define EPS 0.001
#define CUTOFF 200

size_t li(size_t i, size_t j, size_t size)
{
    return (i * size) + j;
}

double *transpose(double *X, size_t size)
{
    double *result = (double *)malloc(size * size * sizeof(double));

    for (size_t i = 0; i < size; i++)
    {
        for (size_t j = 0; j < size; j++)
        {
            result[li(i, j, size)] = X[li(j, i, size)];
        }
    }

    return result;
}

double *dot_matrix_array(double *A, double *b, size_t size)
{
    double *result = (double *)malloc(size * sizeof(double));

    for (size_t i = 0; i < size; i++)
    {
        double acc = 0;
        for (size_t j = 0; j < size; j++)
        {
            acc += A[li(i, j, size)] * b[j];
        }

        result[i] = acc;
    }

    return result;
}

double *dot_matrix(double *A, double *B, size_t size)
{
    double *result = (double *)malloc(size * size * sizeof(double));

    for (size_t i = 0; i < size; i++)
    {
        for (size_t j = 0; j < size; j++)
        {
            result[li(i, j, size)] = 0;
            for (size_t k = 0; k < size; k++)
            {
                result[li(i, j, size)] += A[li(i, k, size)] * B[li(k, j, size)];
            }
        }
    }

    return result;
}

double RMSE(double *a, double *b, size_t size)
{
    double acc = 0;
    for (size_t i = 0; i < size; i++)
    {
        double temp = a[i] - b[i];
        acc += temp * temp;
    }

    return sqrt(acc);
}

double *get_vector(size_t size, double val)
{
    double *v = (double *)malloc(size * size * sizeof(double));

    for (size_t i = 0; i < size; i++)
    {
        v[i] = val;
    }

    return v;
}

void print_m(double *A, size_t size)
{
    for (size_t i = 0; i < size; i++)
    {
        for (size_t j = 0; j < size; j++)
        {
            printf("%f ", A[li(i, j, size)]);
        }
        printf("\n\r");
    }
    printf("\n\r");
}

void print_v(double *v, size_t size)
{
    for (size_t i = 0; i < size; i++)
    {
        printf("%f ", v[i]);
    }
    printf("\n\r");
}

double *get_random_matrix(size_t size)
{
    double *X = (double *)malloc(size * size * sizeof(double));

    srand(41);
    for (size_t i = 0; i < size; i++)
    {
        for (size_t j = 0; j < size; j++)
        {
            X[li(i, j, size)] = 0;

            if (j == i)
            {
                int r = rand() % 1000;
                X[li(i, j, size)] = (double)(r);
            }

            if (j < i)
            {
                int r = rand() % 10;
                X[li(i, j, size)] = (double)(r);
            }
        }
    }

    double *XT = transpose(X, size);
    double *result = dot_matrix(XT, X, size);
    free(X);
    free(XT);
    return result;
}

/***
 *       _____ _           _           _
 *      / ____| |         | |         | |
 *     | |    | |__   ___ | | ___  ___| | ___   _
 *     | |    | '_ \ / _ \| |/ _ \/ __| |/ / | | |
 *     | |____| | | | (_) | |  __/\__ \   <| |_| |
 *      \_____|_| |_|\___/|_|\___||___/_|\_\\__, |
 *                                           __/ |
 *                                          |___/
 */

double *get_L(double *A, size_t size)
{
    double *L = (double *)malloc(size * size * sizeof(double));
    for (size_t i = 0; i < size * size; i++)
    {
        L[i] = 0;
    }

    for (size_t i = 0; i < size; i++)
    {
        double diag = A[li(i, i, size)];
        double acc1 = 0;

        for (size_t p = 0; p < i; p++)
        {
            double temp = L[li(i, p, size)];
            acc1 += temp * temp;
        }

        L[li(i, i, size)] = sqrt(diag - acc1);

        for (size_t j = 0; j < size; j++)
        {
            double acc2 = 0;
            for (size_t p = 0; p < i; p++)
            {
                acc2 += L[li(i, p, size)] * L[li(j, p, size)];
            }

            L[li(j, i, size)] = (A[li(j, i, size)] - acc2) / L[li(i, i, size)];
        }
    }

    return L;
}

double *solve_lt(double *L, double *v, size_t size)
{
    double *result = (double *)malloc(size * sizeof(double));

    for (size_t i = 0; i < size; i++)
    {
        double acc = 0;
        for (size_t j = 0; j < i; j++)
        {
            acc += L[li(i, j, size)] * result[j];
        }

        result[i] = (v[i] - acc) / L[li(i, i, size)];
    }

    return result;
}

double *solve_ut(double *L, double *v, size_t size)
{
    double *result = (double *)malloc(size * sizeof(double));

    for (size_t ii = 0; ii < size; ii++)
    {
        size_t i = (size - 1) - ii;
        double acc = 0;
        for (size_t jj = 0; (size - 1) - jj > i; jj++)
        {
            size_t j = (size - 1) - jj;
            acc += L[li(i, j, size)] * result[j];
        }

        result[i] = (v[i] - acc) / L[li(i, i, size)];
    }

    return result;
}

double *cholesky(double *A, double *b, size_t size)
{
    double *L = get_L(A, size);
    double *y = solve_lt(L, b, size);
    double *LT = transpose(L, size);
    double *x = solve_ut(LT, y, size);

    free(L);
    free(LT);
    free(y);

    return x;
}

/***
 *       _____ _           _           _
 *      / ____| |         | |         | |
 *     | |    | |__   ___ | | ___  ___| | ___   _
 *     | |    | '_ \ / _ \| |/ _ \/ __| |/ / | | |
 *     | |____| | | | (_) | |  __/\__ \   <| |_| |
 *      \_____|_| |_|\___/|_|\___||___/_|\_\\__, |
 *                                           __/ |
 *                                          |___/
 */

int main(int argc, char *argv[])
{
    size_t n = 3500;
    double *X = get_random_matrix(n);
    double *v = get_vector(n, 1);
    clock_t begin = clock();
    double *res = cholesky(X, v, n);
    clock_t end = clock();
    double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("%f\n\r", time_spent);
    print_v(dot_matrix_array(X, res, n), n);
    return 0;
}