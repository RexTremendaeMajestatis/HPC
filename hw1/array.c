#include "stdio.h"
#include "stdlib.h"
#include "math.h"
#include "conio.h"
#include "omp.h"
#include "time.h"
#include <stdint.h>

double sum1(double *v, size_t row, size_t start, size_t end, size_t size)
{
    double res = 0;

    for (size_t i = start; i < end; i++)
    {
        res += v[(row * size) + i];
    }

    return res;
}

double parallel_sum1(double *v, size_t row, size_t start, size_t end, size_t size)
{
    if (end - start + 1 < 100000)
    {
        return sum1(v, row, start, end, size);
    }

    size_t middle = (end - start) / 2;
    double left, right;
    printf("%d \n\r", middle);
#pragma omp task shared(left)
    left = parallel_sum1(v, row, start, start + middle, size);
#pragma omp task shared(right)
    right = parallel_sum1(v, row, start + middle + 1, end, size);
#pragma omp taskwait
    left += right;
    return left;
}

double array_sum1(double *v, size_t row, size_t start, size_t end, size_t size)
{
    double total;
#pragma omp parallel
#pragma omp single nowait
    total = parallel_sum1(v, row, start, end, size);
    return total;
}

int main(int argc, char *argv[])
{
    // omp_set_num_threads(4);

    for (int64_t j = -1; j >= 0; j--)
    {
        printf("%d \n\r", j);
    }
    return 0;
}