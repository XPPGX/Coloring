#include "colorMethod.h"

#define RESULT
#define PRINT

void Firstfit(struct CSR* _csr){
    int nodeID = 0;
    if(_csr->startAtZero == 0){ //如果node從1開始
        nodeID = 1;
    }
    // 顏色候選陣列(bit array)初始化
    int* candidateColorArr = (int*)calloc(_csr->csrVSize, sizeof(int));
    int* nodeColorArr = (int*)malloc(sizeof(int) * _csr->csrVSize);
    // 初始化每個node的顏色
    for(int i = 0 ; i < _csr->csrVSize ; i++){
        nodeColorArr[i] = -1;
    }
    // 開始塗色
    int neighborNum = 0;
    int neighbor = -1;
    int color = -1;
    for(; nodeID < _csr->csrVSize - 1 ; nodeID ++){
        neighborNum = _csr->csrV[nodeID + 1] - _csr->csrV[nodeID];
        // 透過訪問neighbors取得candidateColorArr資訊
        for(int neighborIndex = 0 ; neighborIndex < neighborNum ; neighborIndex ++){
            neighbor = _csr->csrE[_csr->csrV[nodeID] + neighborIndex];
            color = nodeColorArr[neighbor];
            if(color != -1){
                candidateColorArr[color] = 1;
            }
        }
        // 對nodeID塗編號最小的顏色
        for(int colorIndex = 0 ; colorIndex < _csr->csrVSize - 1 ; colorIndex ++){
            if(candidateColorArr[colorIndex] == 0){
                nodeColorArr[nodeID] = colorIndex;
                break;
            }
        }
        // 重置candidateColorArr成0
        memset(candidateColorArr, 0, sizeof(int) * _csr->csrVSize);
    }
    #ifdef RESULT
    nodeID = 0;
    if(_csr->startAtZero == 0){
        nodeID = 1;
    }
    #ifdef PRINT
    
    for(; nodeID < _csr->csrVSize - 1 ; nodeID++){
        printf("node[%d] = {color = %d}\n", nodeID, nodeColorArr[nodeID]);
    }
    #else
    FILE* fptr;
    fptr = fopen("nodesColor.csv", "at+");
    if(fptr == NULL){
        printf("[Fail] open new file\n");
        return;
    }
    else{
        fprintf(fptr, "node,color");
        for(; nodeID < _csr->csrVSize - 1 ; nodeID ++){
            fprintf(fptr, "\n%d,%d", nodeID, nodeColorArr[nodeID]);
        }
    }
    #endif //PRINT
    #endif //RESULT
}

void FFDP(struct CSR* _csr){
    
}