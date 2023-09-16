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

/**
 * @brief 普通的Firstfit
*/
void Firstfit(struct CSR* _csr);

/**
 * @brief Firstfit + 摺疊degree 1 + PriorityQueue
*/
void FFDP(struct CSR* _csr);