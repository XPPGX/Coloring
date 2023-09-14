/**
 * @author XPPGX
 * @date 2023/07/15
*/
#include "CSR.h"

//#define _DEBUG_

struct CSR* createCSR(struct Graph* _adjlist){
    printf("==============================\n");
    printf("CreateCSR...\n");
    int* CSR_V;
    int* CSR_E;
    int nodeID = 0;
    int tempNodeNum = 0;
    if(_adjlist->startAtZero == 1){
        // printf("Graph startAtZero = yes\n");
        tempNodeNum = _adjlist->nodeNum + 1;
        CSR_V = (int*)malloc(sizeof(int) * tempNodeNum);
        nodeID = 0;
    }
    else{
        // printf("Graph startAtZero = no\n");
        tempNodeNum = _adjlist->nodeNum + 2;
        CSR_V = (int*)malloc(sizeof(int) * tempNodeNum);
        CSR_V[0] = -1;
        nodeID = 1;
    }
    CSR_E = (int*)malloc(sizeof(int) * _adjlist->edgeNum * 2);
    int indexCount = -1;
    // printf("nodeID = %d, tempNodeNum = %d, indexCount = %d\n", nodeID, tempNodeNum, indexCount);
    for(; nodeID < tempNodeNum - 1 ; nodeID ++){
        CSR_V[nodeID] = indexCount + 1;

        #ifdef _DEBUG_
        printf("CSR_V[%d] = %d\n", nodeID, CSR_V[nodeID]);
        #endif
        
        for(int neighborIndex = 0 ; neighborIndex <= _adjlist->vertices[nodeID].neighbors->tail ; neighborIndex ++){
            indexCount ++;
            CSR_E[indexCount] = _adjlist->vertices[nodeID].neighbors->dataArr[neighborIndex];
        }
    }
    // printf("nodeID = %d, tempNodeNum = %d, indexCount = %d\n", nodeID, tempNodeNum, indexCount);
    CSR_V[nodeID] = _adjlist->edgeNum * 2;

    struct CSR* csr = (struct CSR*)malloc(sizeof(struct CSR));
    csr->csrV = CSR_V;
    csr->csrE = CSR_E;
    csr->csrVSize = tempNodeNum;
    csr->csrESize = _adjlist->edgeNum * 2;
    csr->startAtZero = _adjlist->startAtZero;
    printf("csr->csrVSize = %d\n", csr->csrVSize);
    printf("csr->csrESize = %d\n", csr->csrESize);
    printf("csr->startAtZero = %d\n", csr->startAtZero);
    printf("[Success] CreateCSR finish\n");
    printf("==============================\n");
    return csr;
}

void showCSR(struct CSR* csr){
    printf("show CSR...\n");
    int nodeID = 0;
    if(csr->startAtZero == 0){
        nodeID ++;
    }
    int neighborNum = 0;
    for(; nodeID < csr->csrVSize - 1 ; nodeID ++){
        neighborNum = csr->csrV[nodeID + 1] - csr->csrV[nodeID];
        printf("neighborNum = %d, neighbor[%d] = {", neighborNum, nodeID);
        for(int i = 0 ; i < neighborNum ; i ++){
            printf("%d, ", csr->csrE[csr->csrV[nodeID] + i]);
        }
        printf("}\n");
    }
    printf("Success : show CSR Finish.\n");
}