#include "stdio.h"
#include "stdlib.h"
#include "math.h"
#include "conio.h"
#include "omp.h"

size_t li(size_t i, size_t j, size_t size)
{
    return (i * size) + j;
}

double sum(double *A, size_t size, size_t start, size_t end)
{
    size_t length = end - start + 1;

    if (length < 0)
    {
        return 0;
    }

    
}

int main(int argc, char *argv[])
{

    return 0;
}