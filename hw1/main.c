#include "stdio.h"
#include "stdlib.h"
#include "math.h"

#define EPS 0.001

size_t li(size_t i, size_t j, size_t size)
{
    return (i * size) + j;
}

double *get_L(double *A, size_t size)
{
    double *L = (double *)malloc(size * size * sizeof(double));

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

double *dot(double *A, double *b, size_t size)
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

int test()
{
    int result = -1.0;
    double A[9] = {81.0, -45.0, 45.0, -45.0, 50.0, -15.0, 45.0, -15.0, 38};
    double v[3] = {1.0, 1.0, 1.0};
    double *x = cholesky(A, v, 3);
    double *expected = dot(A, x, 3);

    result = RMSE(v, expected, 3) < EPS;

    free(expected);
    free(x);

    return result;
}

int main(int argc, char *argv[])
{
    printf("test: %d", test());
}