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
static int DebugPrintf(FILE* _fptr, const char* format, ...){
    va_list argPtr;
    int count;
    va_start(argPtr, format);
    // fflush(stderr);
    count = vfprintf(_fptr, format, argPtr);
    va_end(argPtr);
}
#else
static inline int DebugPrintf(const char* format, ...){

}
#endif

#ifndef tTime
#define tTime
#include "../Lib/tTime/tTime.h"
#endif

int main(int argc, char* argv[]){
    char* datasetPath = argv[1];
    struct Graph* adjList = buildGraph(datasetPath);
    struct CSR* csr = createCSR(adjList);
    struct colorInfo* result = (struct colorInfo*)malloc(sizeof(struct colorInfo));
    // showCSR(csr);
    // degreeOneFolding(csr);
    double startTime = 0;
    double time1 = 0;
    double time2 = 0;
    double time3 = 0;

    // struct algoInfo algo1;
    // startTime = seconds();
    // Firstfit(csr, result);
    // time1 = seconds();
    // checkAns(csr, result, "FF");
    // standardDeviation(result, &algo1);
    // printf("[Execution time] : %2f\n", time1 - startTime);
    // sprintf(algo1.name, "%s", "FF");
    // algo1.Time = time1 - startTime;
    // printf("algo1 = {name = %s, colorNum = %d, MaxMinDiffer = %d, avg = %f, SD = %f, time = %f}\n", algo1.name, algo1.colorNum, algo1.MaxMinDiffer, algo1.Avg, algo1.SD, algo1.Time);

    // free(result->nodeColorArr);
    // free(result);

    // result = (struct colorInfo*)malloc(sizeof(struct colorInfo));


    struct algoInfo algo2;
    time2 = seconds();
    FFDP(csr, result);
    time3 = seconds();
    checkAns(csr, result, "FF_D_P");
    standardDeviation(result, &algo2);
    printf("[Execution time] : %2f\n", time3 - time2);
    sprintf(algo2.name, "%s", "FFDP");
    algo2.Time = time3 - time2;
    printf("algo2 = {name = %s, colorNum = %d, MaxMinDiffer = %d, avg = %f, SD = %f, time = %f}\n", algo2.name, algo2.colorNum, algo2.MaxMinDiffer, algo2.Avg, algo2.SD, algo2.Time);
    
    struct algoInfo* algoPtr = &algo2;
    FILE *recordFptr;
    recordFptr = fopen("metrics.csv", "a");
    if(recordFptr == NULL){
        printf("Open File Error\n");
        exit(1);
    }
    // fprintf(recordFptr, "faker\n");
    fprintf(recordFptr, "%d,%d,%f,%f,%f\n", algoPtr->colorNum, algoPtr->MaxMinDiffer, algoPtr->Avg, algoPtr->SD, algoPtr->Time);
    fclose(recordFptr);
    return 0;
}