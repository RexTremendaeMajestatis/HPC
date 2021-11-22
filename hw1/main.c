#include "stdio.h"
#include "stdlib.h"
#include "math.h"
#include "omp.h"

#define EPS 0.001
#define CUTOFF 10

size_t li(size_t i, size_t j, size_t size)
{
    return (i * size) + j;
}

double serial_sum1(double *A, size_t row, size_t start, size_t end, size_t size)
{
    size_t length = end - start + 1;
    if (length == 0)
    {
        return 0;
    }

    if (length == 1)
    {
        double temp = A[li(row, start, size)];
        return temp * temp;
    }

    size_t middle = (end - start) / 2;

    return serial_sum1(A, row, start, middle, size) + serial_sum1(A, row, middle + 1, end, size);
}

double serial_sum2(double *A, size_t row1, size_t row2, size_t start, size_t end, size_t size)
{
    size_t length = end - start + 1;
    if (length == 0)
    {
        return 0;
    }

    if (length == 1)
    {
        double temp = A[li(row1, start, size)] * A[li(row2, start, size)];
        return temp;
    }

    size_t middle = (end - start) / 2;

    return serial_sum2(A, row1, row2, start, middle, size) + serial_sum2(A, row1, row2, middle + 1, end, size);
}

double parallel_sum1(double *A, size_t row, size_t start, size_t end, size_t size)
{
    // printf("thread: %d\n\r", omp_get_thread_num());
    size_t length = end - start + 1;
    if (length < CUTOFF)
    {
        return serial_sum1(A, row, start, end, size);
    }

    double left;
    double right;
    size_t middle = (end - start) / 2;
#pragma omp task shared(left)
    left = parallel_sum1(A, row, start, middle, size);
#pragma omp task shared(right)
    right = parallel_sum1(A, row, middle + 1, end, size);
#pragma omp taskwait
    left += right;
    return left;
}

double parallel_sum2(double *A, size_t row1, size_t row2, size_t start, size_t end, size_t size)
{
    // printf("thread: %d\n\r", omp_get_thread_num());
    size_t length = end - start + 1;
    if (length < CUTOFF)
    {
        return serial_sum2(A, row1, row2, start, end, size);
    }

    double left;
    double right;
    size_t middle = (end - start) / 2;
#pragma omp task shared(left)
    left = parallel_sum2(A, row1, row2, start, middle, size);
#pragma omp task shared(right)
    right = parallel_sum2(A, row1, row2, middle + 1, end, size);
#pragma omp taskwait
    left += right;
    return left;
}

double array_sum1(double *A, size_t row, size_t start, size_t end, size_t size)
{
    float result;
#pragma omp parallel
#pragma omp single nowait
    result = parallel_sum1(A, row, start, end, size);
    return result;
}

double array_sum2(double *A, size_t row1, size_t row2, size_t start, size_t end, size_t size)
{
    float result;
#pragma omp parallel
#pragma omp single nowait
    result = parallel_sum2(A, row1, row2, start, end, size);
    return result;
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

double *parallel_get_L(double *A, size_t size)
{
    double *L = (double *)malloc(size * size * sizeof(double));

    for (size_t i = 0; i < size; i++)
    {
        double diag = A[li(i, i, size)];
        double acc1 = array_sum1(L, i, 0, i - 1, size);
        L[li(i, i, size)] = sqrt(diag - acc1);

        for (size_t j = 0; j < size; j++)
        {
            double acc2 = array_sum2(L, i, j, 0, i - 1, size);

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

double *parallel_cholesky(double *A, double *b, size_t size)
{
    double *L = parallel_get_L(A, size);
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
    int result = -1;
    double A[9] = {81.0, -45.0, 45.0, -45.0, 50.0, -15.0, 45.0, -15.0, 38};
    double v[3] = {1.0, 1.0, 1.0};
    double *x = cholesky(A, v, 3);
    double *expected = dot(A, x, 3);

    result = RMSE(v, expected, 3) < EPS;

    free(expected);
    free(x);

    return result;
}

int parallel_test()
{
    int result = -1;
    double A[9] = {81.0, -45.0, 45.0, -45.0, 50.0, -15.0, 45.0, -15.0, 38};
    double v[3] = {1.0, 1.0, 1.0};
    double *x = parallel_cholesky(A, v, 3);

    double *expected = dot(A, x, 3);

    result = RMSE(v, expected, 3) < EPS;

    free(expected);
    free(x);

    return result;
}

int test_sum()
{
    int result = -1;
    double A[9] = {81.0, -45.0, 45.0, -45.0, 50.0, -15.0, 45.0, -15.0, 38};
    double actual = serial_sum1(A, 0, 0, 2, 3);
    double expected = 0;
    for (size_t i = 0; i < 3; i++)
    {
        double temp = A[li(0, i, 3)];
        expected += temp * temp;
    }

    result = (actual - expected) < EPS;
    return result;
}

int parallel_test_sum()
{
    int result = -1;
    double A[9] = {81.0, -45.0, 45.0, -45.0, 50.0, -15.0, 45.0, -15.0, 38};
    double actual = parallel_sum1(A, 0, 0, 2, 3);
    double expected = 0;
    for (size_t i = 0; i < 3; i++)
    {
        double temp = A[li(0, i, 3)];
        expected += temp * temp;
    }

    result = (actual - expected) < EPS;
    return result;
}

int array_sum()
{
    int result = -1;
    double A[9] = {81.0, -45.0, 45.0, -45.0, 50.0, -15.0, 45.0, -15.0, 38};
    double actual = array_sum1(A, 0, 0, 2, 3);
    double expected = 0;
    for (size_t i = 0; i < 3; i++)
    {
        double temp = A[li(0, i, 3)];
        expected += temp * temp;
    }

    result = (actual - expected) < EPS;
    return result;
}

int main(int argc, char *argv[])
{
    printf("main thread: %d\n\r", omp_get_thread_num());
    printf("test sum: %d\n\r", test_sum());
    printf("parallel test sum: %d\n\r", parallel_test_sum());
    printf("array sum: %d\n\r", array_sum());
    printf("algo test: %d\n\r", test());
    printf("parallel algo test: %d\n\r", parallel_test());
}