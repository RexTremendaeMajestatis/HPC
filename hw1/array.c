#include "stdio.h"
#include "stdlib.h"
#include "math.h"
#include "conio.h"
#include "omp.h"

double serial_sum(double *A, size_t start, size_t end, size_t size)
{
    if (end - start + 1 <= 0)
    {
        return 0;
    }

    if (end - start + 1 == 1)
    {
        return A[start];
    }

    size_t middle = (end - start) / 2;
    return serial_sum(A, start, start + middle, size) + serial_sum(A, start + middle + 1, end, size);
}

double parallel_sum(double *A, size_t start, size_t end, size_t size)
{
    if (end - start + 1 <= 10)
    {
        return serial_sum(A, start, end, size);
    }

    size_t middle = (end - start) / 2;
    double left, right;
#pragma omp task shared(left)
    {
        left = parallel_sum(A, start, start + middle, size);
    }
#pragma omp task shared(right)
    {
        right = parallel_sum(A, start + middle + 1, end, size);
    }
#pragma omp taskwait
    left += right;
    return left;
}

int main(int argc, char *argv[])
{
    omp_set_num_threads(4);

    size_t n = 1000000000;
    double *A = (double *)malloc(n * sizeof(double));
    for (size_t i = 0; i < n; i++)
    {
        A[i] = i;
    }
#pragma omp parallel
#pragma omp single nowait
    {
        printf("%f\n\r", parallel_sum(A, 0, n - 1, n));
    }
    return 0;
}