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
    int csrVSize; //結尾會多一個格子，放總共的edge數，當作最後的offset。
    int csrESize; //如果是無向圖，csrESize就是原本dataset的兩倍。
    int startAtZero;
};

struct CSR* createCSR(struct Graph* _adjlist);
void showCSR(struct CSR* csr);