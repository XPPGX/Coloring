#ifndef COMMON
#define COMMON
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#endif

#ifndef coloring
#define coloring
#include "colorMethod.h"
#endif

#define _DEBUG_
#ifdef _DEBUG_
#include <stdarg.h> //有這個library才可以定義可變長度參數
static int DebugPrintf(const char* format, ...){
    va_list argPtr;
    int count;
    va_start(argPtr, format);
    fflush(stderr);
    count = vfprintf(stderr, format, argPtr);
    va_end(argPtr);
}
#else
static inline int DebugPrintf(const char* format, ...){

}
#endif

int main(int argc, char* argv[]){
    char* datasetPath = argv[1];
    struct Graph* adjList = buildGraph(datasetPath);
    struct CSR* csr = createCSR(adjList);
    struct colorInfo* result = (struct colorInfo*)malloc(sizeof(struct colorInfo));
    // showCSR(csr);
    // degreeOneFolding(csr);

    // Firstfit(csr, result);
    // checkAns(csr, result, "FF");
    // standardError(result);

    FFDP(csr, result);
    // checkAns(csr, result, "FF_D_P");
    // standardError(result);
}