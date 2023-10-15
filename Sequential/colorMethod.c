#include "colorMethod.h"

// #define _DEBUG_
// #define RESULT
// #define PRINT

void Firstfit(struct CSR* _csr, struct colorInfo* _result){
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
    int maxColor = -1; //紀錄Number of color class
    for(; nodeID < _csr->csrVSize - 1 ; nodeID ++){
        neighborNum = _csr->csrNodesDegree[nodeID];
        
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

                // 如果有新顏色，則maxColor更新
                if(maxColor < colorIndex){
                    maxColor = colorIndex;
                }
                
                break;
            }
        }
        // 重置candidateColorArr成0
        memset(candidateColorArr, 0, sizeof(int) * _csr->csrVSize);
    }
    // 把結果送進_result
    _result->nodeColorArr = nodeColorArr;
    _result->totalColorClassNum = maxColor + 1; //FF的color class從0開始

    //顯示結果或將結果輸出至csv
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
    fclose(fptr);
    #endif //PRINT
    #endif //RESULT
}

// First fit with {degree 1 folding and priority queue for colors}
void FFDP(struct CSR* _csr, struct colorInfo* _result){
    degreeOneFolding(_csr);

    struct minPQ* colorPQ = (struct minPQ*)malloc(sizeof(struct minPQ));
    minPQ_Init(colorPQ);
    // 插入color 1，數量0
    int nodeID = 0;
    if(_csr->startAtZero == 0){
        nodeID = 1;
    }
    printf("Vsize = %d\n", _csr->csrVSize);
    // 顏色候選陣列(bit array)初始化
    int* candidateColorArr = (int*)calloc(_csr->csrVSize, sizeof(int));
    int* nodeColorArr = (int*)malloc(sizeof(int) * (_csr->csrVSize - 1));
    // 初始化每個node的顏色
    for(int i = 0 ; i < _csr->csrVSize - 1 ; i++){
        nodeColorArr[i] = -1;
    }
    // 開始塗色
    int neighborNum = 0;
    int neighbor = -1;
    int usedColor = -1;
    int optimalColor = -1;
    for(; nodeID < _csr->csrVSize - 1 ; nodeID ++){
        neighborNum = _csr->csrNodesDegree[nodeID];

        if(neighborNum > 0){
            for(int neighborIndex = 0 ; neighborIndex < neighborNum ; neighborIndex ++){
                neighbor = _csr->csrE[_csr->csrV[nodeID] + neighborIndex];
                usedColor = nodeColorArr[neighbor];
                if(usedColor != -1){
                    candidateColorArr[usedColor] = 1;
                }
            }
            // 如果在colorPQ中找到可以塗的就塗，不能的話就要引入新顏色
            for(int ColorIndex = 1 ; ColorIndex <= colorPQ->tail ; ColorIndex ++){
                optimalColor = colorPQ->heapNodes[ColorIndex].element;
                if(candidateColorArr[optimalColor] == 0){
                    nodeColorArr[nodeID] = optimalColor;
                    minPQ_increaseKey(colorPQ, optimalColor, 1);
                    break;
                }
            }
            // 如果colorPQ沒有可以用的顏色，則引入新顏色。
            if(nodeColorArr[nodeID] == -1){

                #ifdef _DEBUG_
                printf("\tnodeID = %d, use new color %d\n", nodeID, colorPQ->tail + 1);
                #endif

                minPQ_Insert(colorPQ, colorPQ->tail + 1, 1); //minPQ_Insert完之後，tail已經 + 1
                nodeColorArr[nodeID] = colorPQ->tail;
            }
            // printf("finished nodeID = %d\n", nodeID);
            memset(candidateColorArr, 0, sizeof(int) * _csr->csrVSize);
        }
    }
    
    // 開始塗 folded Node
    int noColoredD1Node = -1;
    int noColoredD1NodeNeighbor = -1;
    for(int d1NodeIndex = 0 ; d1NodeIndex < _csr->foldedDegreeOneCount ; d1NodeIndex ++){
        // 取得d1Node
        noColoredD1Node = _csr->degreeOneNodesQ->dataArr[d1NodeIndex];
        // 把d1Node的csrV offset復原
        _csr->csrV[noColoredD1Node] = _csr->csrV[noColoredD1Node] - _csr->oriCsrNodesDegree[noColoredD1Node];
        // 訪問d1Node的neighbors
        for(int d1NodeNeighborIndex = 0 ; d1NodeNeighborIndex < _csr->oriCsrNodesDegree[noColoredD1Node] ; d1NodeNeighborIndex ++){
            noColoredD1NodeNeighbor = _csr->csrE[_csr->csrV[noColoredD1Node] + d1NodeNeighborIndex];
            usedColor = nodeColorArr[noColoredD1NodeNeighbor];
            if(usedColor != -1){
                candidateColorArr[usedColor] = 1;
            }
        }

        for(int colorIndex = 1 ; colorIndex <= colorPQ->tail ; colorIndex ++){
            optimalColor = colorPQ->heapNodes[colorIndex].element;
            if(candidateColorArr[optimalColor] == 0){
                nodeColorArr[noColoredD1Node] = optimalColor;
                minPQ_increaseKey(colorPQ, optimalColor, 1);
                break;
            }
        }
        memset(candidateColorArr, 0, sizeof(int) * _csr->csrVSize);
    }

    //FREE
    free(colorPQ->elementIndexArr);
    free(colorPQ->heapNodes);
    free(colorPQ);

    free(candidateColorArr);

    // 把結果放入_result
    _result->nodeColorArr = nodeColorArr;
    _result->totalColorClassNum = colorPQ->tail;
    #ifdef _DEBUG_
    nodeID = 0;
    if(_csr->startAtZero == 0){
        nodeID = 1;
    }
    int count = 0;
    for(; nodeID < _csr->csrVSize - 1 ; nodeID ++){
        if(nodeColorArr[nodeID] == -1){
            printf("uncolored nodeID = %d\n", nodeID);
            count ++;
        }
    }
    printf("colored -1 node count = %d\n", count);
    
    for(int i = 1 ; i <= colorPQ->tail ; i ++){
        printf("colorNum[%d] = %d\n", colorPQ->heapNodes[i].element, colorPQ->heapNodes[i].key);
    }
    #endif

    //顯示結果或將結果輸出至csv
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

// FFDP在traverse時會盡量往degree大的neighbor走，目標是要找到所有需要使用的顏色
// void FFDP_MDN(struct CSR* _csr){

// }

void checkAns(struct CSR* _csr, struct colorInfo* _result, const char* _methodName){
    printf("==============================\n");
    printf("Checking answer(%s)...\t", _methodName);
    int nodeID = 0;
    _result->startWith = 0;
    _result->NodeNum = _csr->csrVSize - 1;
    if(_csr->startAtZero == 0){
        nodeID = 1;
        _result->startWith = 1;
        _result->NodeNum--;
    }
    int currentNodeColor = -1;
    int neighborNode = -1;
    int neighborColor = -1;
    for(; nodeID < _csr->csrVSize - 1 ; nodeID ++){
        // printf("neighbor[%d] = {", nodeID);
        currentNodeColor = _result->nodeColorArr[nodeID];
        
        for(int neighborIndex = 0 ; neighborIndex < _csr->oriCsrNodesDegree[nodeID] ; neighborIndex ++){
            neighborNode = _csr->csrE[_csr->oriCsrV[nodeID] + neighborIndex];
            neighborColor = _result->nodeColorArr[neighborNode];
            if(currentNodeColor == neighborColor){
                printf("[Ans] Wrong (╯°□°）╯︵ ┻━┻\n");
                return;
            }
            // printf("%d, ", neighborNode);
        }
        // printf("}\n");
    }
    printf("[Ans] Correct (⌐■_■)\n");
}

void standardDeviation(struct colorInfo* _result, struct algoInfo* _algo){
    printf("startwith %d, nodeNum = %d, colorClassNum = %d\n", _result->startWith, _result->NodeNum, _result->totalColorClassNum);
    /****Bug在這****/
    double* numPerColorClass = (double*)calloc(_result->totalColorClassNum, sizeof(double)); //Bug "malloc(): invalid size(unsorted)"，原因在這
    // int* numPerColorClass = (int*)calloc(_result->totalColorClassNum, sizeof(int));
    // float* numPerColorClass = (float*)calloc(_result->totalColorClassNum, sizeof(float));
    /***************/
    int nodeID = 0;
    if(_result->startWith == 1){
        nodeID = 1;
    }
    printf("nodeID = %d\n", nodeID);
    int color = -1;
    //BUG!!!!!!!
    for(int i = 0 ; i < _result->NodeNum ; i ++, nodeID ++){
        color = _result->nodeColorArr[nodeID];
        numPerColorClass[color] = numPerColorClass[color] + 1;
        // printf("nodeID = %d, color = %d, numPercolorClass[%d] = %d, addr = %p\n", nodeID, color, color, numPerColorClass[color], &numPerColorClass[color]);
    }
    // for(; nodeID <= _result->NodeNum ; nodeID ++){
    //     color = _result->nodeColorArr[nodeID];
    //     numPerColorClass[color] = numPerColorClass[color] + 1;
    //     printf("nodeID = %d, color = %d, numPercolorClass[%d] = %d, addr = %p\n", nodeID, color, color, numPerColorClass[color], &numPerColorClass[color]);
    //     // if(nodeID == 334863){
    //     //     return;
    //     // }
    // }
    
    printf("nodeID = %d, _result->NodeNum = %d\n", nodeID, _result->NodeNum);
    /*************/
    double NodeNum = _result->NodeNum;
    double totalColorClassNum = _result->totalColorClassNum;
    int colorIndex = 0;
    if(numPerColorClass[0] == 0){
        colorIndex = 1;
    }
    printf("totalColorClassNum = %f\n", totalColorClassNum);
    double classAvgNodeNum = NodeNum / totalColorClassNum; //average number of nodes for color classes
    double classVarNodeNum = 0; //variance number of nodes for color classes
    printf("avg = %2f\n", classAvgNodeNum);
    
    
    //算最大最小差距
    int maxUsedColorNum = -1;
    int minUsedColorNum = __INT_MAX__;
    if(numPerColorClass[0] == 0){
        colorIndex = 1;
    }
    for(; colorIndex <= _result->totalColorClassNum ; colorIndex ++){
        printf("numPerColorClass[%d] = %f\n", colorIndex, numPerColorClass[colorIndex]);
        if((maxUsedColorNum < numPerColorClass[colorIndex]) && (numPerColorClass[colorIndex] >= 1)){
            maxUsedColorNum = numPerColorClass[colorIndex];
        }
        if((minUsedColorNum > numPerColorClass[colorIndex]) && (numPerColorClass[colorIndex] >= 1)){
            // printf("Find min %f, and minUsedColorNum = %d, swap!\n", numPerColorClass[colorIndex], minUsedColorNum);
            minUsedColorNum = numPerColorClass[colorIndex];
        }
    }
    printf("maxUsedColorNum = %d, minUsedColorNum = %d\n", maxUsedColorNum, minUsedColorNum);

    // 算Variance
    colorIndex = 0;
    if(numPerColorClass[0] == 0){
        colorIndex = 1;
    }
    for(; colorIndex <= _result->totalColorClassNum ; colorIndex ++){
        // printf("colorIndex = %d\n", colorIndex);
        classVarNodeNum += (((double)numPerColorClass[colorIndex] - classAvgNodeNum) * ((double)numPerColorClass[colorIndex] - classAvgNodeNum));
    }
    classVarNodeNum /= totalColorClassNum;

    // 算Standard deviation(標準差, S.D.)
    double SD = sqrt(classVarNodeNum);
    printf("Standard Deviation = %2f\n", SD);

    _algo->Avg = classAvgNodeNum;
    _algo->SD = SD;
    _algo->colorNum = totalColorClassNum;
    _algo->MaxMinDiffer = maxUsedColorNum - minUsedColorNum;
}