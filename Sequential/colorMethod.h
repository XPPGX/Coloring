#ifndef COMMON
#define COMMON
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#endif

#ifndef PQ
#define PQ
#include "../Lib/PriorityQueue/PriorityQueue.h"
#endif

#ifndef cCSR
#define cCSR
#include "../Lib/CSR/CSR.h"
#endif

#include <math.h>

struct algoInfo{
    char name[1000];
    int colorNum;
    double Avg;
    double SD;
    double Time;
    int MaxMinDiffer;
};

struct colorInfo{
    int* nodeColorArr; //紀錄每個node的顏色的array
    int totalColorClassNum; //紀錄總共有多少個color class
    int startWith; //紀錄nodeID從0或1開始
    int NodeNum; //總node數
};


/**
 * @brief 普通的Firstfit
*/
void Firstfit(struct CSR* _csr, struct colorInfo* _result);

/**
 * @brief BFS-like FF
*/
void BFS_FF(struct CSR* _csr, int startNodeID, struct colorInfo* _result);

/**
 * @brief Firstfit + 摺疊degree 1 + PriorityQueue
*/
void FFDP(struct CSR* _csr, struct colorInfo* _result);

/**
 * @brief 從起始點(任意點)，往degree大的neighbor走，使的neighbor
*/
// void FFDP_MDN(struct CSR* _csr);

/**
 * @brief 檢查是否有不合法的顏色
*/
void checkAns(struct CSR* _csr, struct colorInfo* _result, const char* _methodName);

/**
 * @brief 算顏色使用數量的變異數與標準差並回傳變異數
*/
void standardDeviation(struct colorInfo* _result, struct algoInfo* _algo);

/**
 * @brief 把算出來的metric顯示在螢幕上，或寫入csv file
*/
void metric(struct colorInfo* _result);