/**
 * @author XPPGX
 * @date 2023/07/15
*/
#ifndef COMMON
#define COMMON
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#endif

//content
#define CUDA

#ifndef ADJLIST
#define ADJLIST
#include "../AdjList/AdjList.h"
#endif

struct CSR{
    int* csrV;
    int* csrE;
    int csrVSize;
    int csrESize;
    int startAtZero;
};

struct CSR* createCSR(struct Graph* _adjlist);
void showCSR(struct CSR* csr);