#ifndef COMMON
#define COMMON
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#endif

#ifndef PriorityQ
#define PrioirtyQ
#include "PriorityQueue.h"
#endif

#define TEST
#ifdef TEST
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

int main(){
    struct minPQ* minPQ_Data = (struct minPQ*)malloc(sizeof(struct minPQ*));
    minPQ_Init(minPQ_Data);
    // int a = 0;
    // DebugPrintf("[Error] : node %d is not degree one\n", a);
    int fullFlag = minPQ_IsFull(minPQ_Data);
    DebugPrintf("[line 35][Log] : fullFlag = %d\n", fullFlag);
    minPQ_Insert(minPQ_Data, 10, 20);
    DebugPrintf("[line 37][Log] : color[%d] = %d\n", minPQ_Data->heapNodes[1].element, minPQ_Data->heapNodes[1].key);
    minPQ_Insert(minPQ_Data, 2, 40);
    minPQ_Insert(minPQ_Data, 4, 60);
    minPQ_Insert(minPQ_Data, 1, 70);
    minPQ_Insert(minPQ_Data, 6, 10);
    minPQ_Insert(minPQ_Data, 13, 20);
    minPQ_Insert(minPQ_Data, 14, 30);
    minPQ_Insert(minPQ_Data, 14, 2);
    minPQ_Insert(minPQ_Data, 14, 5);
    minPQ_Insert(minPQ_Data, 14, 8);
    minPQ_Insert(minPQ_Data, 14, 1);
    minPQ_Insert(minPQ_Data, 14, 6);
    DebugPrintf("minPQ nodes :\n");
    DebugPrintf("before heapifying...\n");
    for(int nodeID = 1 ; nodeID <= minPQ_Data->tail ; nodeID ++){
        DebugPrintf("nodeID[%d] = {element = %d, key = %d}\n", nodeID, minPQ_Data->heapNodes[nodeID].element, minPQ_Data->heapNodes[nodeID].key);
    }
}