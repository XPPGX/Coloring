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

#ifndef QQueue
#define QQueue
#include "../qQueue/qQueue.h"
#endif

struct CSR{
    int* csrV;
    int* csrE;
    int csrVSize; //結尾會多一個格子，放總共的edge數，當作最後的offset。
    int csrESize; //如果是無向圖，csrESize就是原本dataset的兩倍。
    int* csrNodesDegree; //紀錄csr的各個node的degree。
    struct qQueue* degreeOneNodesQ; //紀錄誰是degreeOne的Queue。
    int foldedDegreeOneCount;
    int startAtZero;
};

struct CSR* createCSR(struct Graph* _adjlist);
void swap(int* _val1, int* _val2);
void degreeOneFolding(struct CSR* _csr);
void showCSR(struct CSR* csr);