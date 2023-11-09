/**
 * @author XPPGX
 * @remark
 * 1. Bitmap 的某些operation還沒搞定            (pend)
*/

#ifndef COMMON
#define COMMON
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#endif

extern "C"{
    #ifndef cCSR
    #define cCSR
    #include "../Lib/CSR/CSR.h"
    #endif

    #ifndef tTime
    #define tTime
    #include "../Lib/tTime/tTime.h"
    #endif
}

#include <cuda_runtime.h>

//define
#define _DEBUG_

//用於紀錄CSRInfo
struct CSRInfo{
    int startAtZero;
    int nodeSize;
    int startNodeID;
    int endNodeID;
    int maxDegree;
    int nodeBitmapIntNum;
};

//用於紀錄DAG的offset
struct NodeInfo{
    //For the offset of DAG_E
    int nodeStartOffset;
    int nodeEndOffset;
    
    //For the bitmap        
    int possColorBitLength;     //[variable]    紀錄bitmap長度
    int bitmapStartOffset;      //[variable]    紀錄在bitmap中的offset，代表從bitmap中的哪一格開始
    int bestColor;              //[variable]    紀錄當前最好的顏色
    int worstColor;             //[variable]    紀錄當前最差的顏色    
};

int reAssignNodeSize(CSR* _csr);
CSRInfo getCsrInfo(CSR* _csr, int _nodeSize);
void checkDevice();
void checkOneNodeBitmap(struct NodeInfo* _hostNodeInfo, struct CSR* _csr, unsigned int* _hostNodeBitmap, int* _hostDAG_E, int _nodeID);

void ECL_GC_Init(   int* _cudaCsrV, int* _cudaCsrE, int* _cudaWorkList,
                    int* _cudaWorkListNowIndex, int* _cudaCsrDegree, int* _cudaColorArr,
                    int* _cudaDAG_E, unsigned int* _cudaNodeBitmap, struct NodeInfo* _cudaNodeInfo,
                    struct CSRInfo* _deviceCsrInfo, dim3 _block, dim3 _grid);

__device__ void BitmapSetOperation(int nodeID, unsigned int** _cudaNodeBitmap, struct NodeInfo* _cudaNodeInfo, struct CSRInfo* _deviceCsrInfo);

__global__ void Init(   int* _cudaCsrV, int* _cudaCsrE, int* _cudaWorkList,
                        int* _cudaWorkListNowIndex, int* _cudaCsrDegree, int* _cudaColorArr,
                        int* _cudaDAG_E, unsigned int* _cudaNodeBitmap, struct NodeInfo* _cudaNodeInfo,
                        struct CSRInfo* _deviceCsrInfo);

__global__ void Worklist_DAG_Init(int* _cudaWorkList,   int* _cudaWorkListLength,    int* _cudaCsrDegree,
                                  int* _cudaCsrV,       int* _cudaCsrE,             struct NodeInfo* _cudaNodeInfo,
                                  int* _cudaDAG_E,      unsigned int* _cudaNodeBitmap,  struct CSRInfo* _deviceCsrInfo);

__device__ void bitmapRemoveOnebit( int _nodeID, unsigned int** _cudaNodeBitmap, struct NodeInfo* _cudaNodeInfo,
                                    int targetBitLocation, struct CSRInfo* _deviceCsrInfo);

__device__ int bitmapGetBestColor(  int _nodeID, unsigned int* _cudaNodeBitmap, struct NodeInfo* _cudaNodeInfo,
                                    struct CSRInfo* _deviceCsrInfo);

__device__ int bitmapGetWorstColor( int _nodeID, unsigned int* _cudaNodeBitmap, struct NodeInfo* _cudaNodeInfo,
                                    struct CSRInfo* _deviceCsrInfo);

__global__ void HighDegreeColoring( int* _cudaWorkList, int* _cudaWorkListLength, int* _cudaColorArr, int* _cudaDAG_E, struct NodeInfo* _cudaNodeInfo,
                                    int* _againFlag,    unsigned int* _cudaNodeBitmap, struct CSRInfo* _deviceCsrInfo);

void ECL_GC_HighDegree_Vertex_Coloring( int* _cudaWorkList, int* _cudaWorkListLength, int* _cudaColorArr, int* _cudaDAG_E,
                                        struct NodeInfo* _cudaNodeInfo, unsigned int* _cudaNodeBitmap, struct CSRInfo* _deviceCsrInfo,
                                        dim3 _block, dim3 _grid);

__global__ void ECL_GC_InitSub( int _nodeID,            int* _cudaCsrV,     int* _cudaCsrE,
                                int* _cudaCsrDegree,    int* _cudaDAG_E,    int* _subKernelFlag,
                                int* _cudaDAG_nodeEndOffset, struct NodeInfo* _cudaDAG_E_Offset);

int main(int argc, char* argv[]){
    char* datasetPath   = argv[1];
    
    Graph* adjlist = buildGraph(datasetPath);
    CSR* csr = createCSR(adjlist);

    #ifdef _DEBUG_
    // showCSR(csr);
    #endif
    int nodeSize                = reAssignNodeSize(csr);        //取得真正的node數
    CSRInfo hostCsrInfo         = getCsrInfo(csr, nodeSize);    //取得Csr的一些資訊(可擴充)
    printf("MaxDegree = %d, node int element num = %d\n", csr->maxDegree, (csr->maxDegree + 32) / 32);
    int colorBitmapIntElementNum   = csr->csrVSize * ((csr->maxDegree + 32) / 32); //紀錄CSR形式的colorMap要用多少個Unsigned Int組成
    //因為每個node的Bitmap長度都是((csr->maxDegree+32) / 32)，以Dblp為例，每個node都有大約11個int的長度。大概300多bit。
    checkDevice();

#pragma region threadlayout
    dim3 block(32,1);
    dim3 grid((csr->csrVSize + block.x - 1) / block.x, 1);
#pragma endregion

#pragma region copyData
    //Define DevicePointer
    int* cudaCsrV;                              //[Array]   以 CSR 的方式紀錄 nodes
    int* cudaCsrE;                              //[Array]   以 CSR 的方式紀錄 edges
    int* cudaCsrDegree;                         //[Array]   紀錄每個 node 的 degree
    int* cudaWorkList;                          //[Array]   紀錄 degree > 32 的node
    int* cudaColorArr;                          //[Array]   紀錄每個 node 的 color     

    int* cudaWorkListNowIndex;                  //[Variable]紀錄在kernel中的 WorkList 的 Index，用於Atomic operation 將 node 放入 WorkList
    struct CSRInfo* deviceCsrInfo;              //[Variable]紀錄 CSR 的資訊如，從0開始、node數量 
    int* cudaDAG_E;                             //[Array]   以CSR的方式紀錄每個node的鄰居，有哪些是比自己更優先的，「offset一開始照舊用csr->CsrV且如果碰到-1則停下，後續則用cudaDAG_V_offset」
    struct NodeInfo* cudaNodeInfo;              //[Array]   紀錄每個node，自己DAG的start offset與end offset，以及當前最好的顏色、可使用的顏色長度。
    unsigned int* cudaNodeBitmap;               //[Array]   紀錄每個node，自己的Bitmap。
    //Malloc device memory space for DevicePointer
    cudaMalloc((void**)&cudaCsrV, sizeof(int) * csr->csrVSize);
    cudaMalloc((void**)&cudaCsrE, sizeof(int) * csr->csrESize);
    cudaMalloc((void**)&cudaCsrDegree, sizeof(int) * csr->csrVSize);
    cudaMalloc((void**)&deviceCsrInfo, sizeof(CSRInfo));
    cudaMalloc((void**)&cudaWorkList, sizeof(int) * csr->csrVSize);
    cudaMalloc((void**)&cudaColorArr, sizeof(int) * csr->csrVSize);
    cudaMalloc((void**)&cudaWorkListNowIndex, sizeof(int));
    cudaMalloc((void**)&cudaDAG_E, sizeof(int) * csr->csrESize);
    cudaMalloc((void**)&cudaNodeInfo, sizeof(NodeInfo) * csr->csrVSize);
    cudaMalloc((void**)&cudaNodeBitmap, sizeof(unsigned int) * colorBitmapIntElementNum);
    //Copy data from host to device
    printf("[Execution][Copy Data : Host To Device]...\n");
    cudaMemcpy(cudaCsrV, csr->csrV, sizeof(int) * csr->csrVSize, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaCsrE, csr->csrE, sizeof(int) * csr->csrESize, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaCsrDegree, csr->csrNodesDegree, sizeof(int) * csr->csrVSize, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceCsrInfo, &hostCsrInfo, sizeof(CSRInfo), cudaMemcpyHostToDevice);
    cudaMemset(cudaWorkList, 0, sizeof(int) * csr->csrVSize);
    cudaMemset(cudaColorArr, -1, sizeof(int) * csr->csrVSize);
    cudaMemset(cudaWorkListNowIndex, 0, sizeof(int));
    cudaMemset(cudaDAG_E, -1, sizeof(int)* csr->csrESize);  //cudaDAG_E如果有cell = -1，代表那格是空的
    cudaMemset(cudaNodeBitmap, 0, sizeof(unsigned int) * colorBitmapIntElementNum); //0代表不能用的，1代表可以用
    printf("[Finish][Copy Data : Host To Device]~\n");
#pragma endregion

#pragma region Algo
    ECL_GC_Init(cudaCsrV, cudaCsrE, cudaWorkList, cudaWorkListNowIndex, cudaCsrDegree, cudaColorArr,
                cudaDAG_E, cudaNodeBitmap, cudaNodeInfo, deviceCsrInfo, block, grid);

    ECL_GC_HighDegree_Vertex_Coloring(  cudaWorkList, cudaWorkListNowIndex, cudaColorArr,
                                        cudaDAG_E, cudaNodeInfo, cudaNodeBitmap, deviceCsrInfo,
                                        block, grid);
#pragma endregion //Algo
    int* hostCsrDegree = (int*)malloc(sizeof(int) * csr->csrVSize);
    cudaMemcpy(hostCsrDegree, cudaCsrDegree, sizeof(int) * csr->csrVSize, cudaMemcpyDeviceToHost);
    #ifdef _DEBUG_
    for(int i = hostCsrInfo.startNodeID ; i <= hostCsrInfo.endNodeID ; i ++){
        // printf("Degree[%d] = {host = %d, device = %d}\n", i, csr->csrNodesDegree[i], hostCsrDegree[i]);
        if(csr->csrNodesDegree[i] != hostCsrDegree[i]){
            printf("node[i] degree record wrong!!\n");
        }
        
    }
    #endif

    int* hostWorkList                   = (int*)malloc(sizeof(int) * csr->csrVSize);
    int* hostWorkListNowIndex           = (int*)malloc(sizeof(int));
    int* hostDAG_E                      = (int*)malloc(sizeof(int) * csr->csrESize);
    NodeInfo* hostNodeInfo              = (NodeInfo*)malloc(sizeof(NodeInfo) * csr->csrVSize);
    unsigned int* hostNodeBitmap        = (unsigned int*)malloc(sizeof(unsigned int) * colorBitmapIntElementNum);
    int* hostColorArr                   = (int*)malloc(sizeof(int) * csr->csrVSize);

    printf("[Execution][Copy Data : Device To Host]...\n");
    cudaMemcpy(hostWorkList, cudaWorkList, sizeof(int) * nodeSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(hostWorkListNowIndex, cudaWorkListNowIndex, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(hostDAG_E, cudaDAG_E, sizeof(int) * csr->csrESize, cudaMemcpyDeviceToHost);
    cudaMemcpy(hostNodeInfo, cudaNodeInfo, sizeof(NodeInfo) * csr->csrVSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(hostNodeBitmap, cudaNodeBitmap, sizeof(unsigned int) * colorBitmapIntElementNum, cudaMemcpyDeviceToHost);
    cudaMemcpy(hostColorArr, cudaColorArr, sizeof(int) * csr->csrVSize, cudaMemcpyDeviceToHost);
    printf("[Finish][Copy Data : Device To Host]~\n");
    #ifdef _DEBUG_
    int count = 0;
    printf("=============In Host==============\n");
    // printf("[ColorArr Checking] : \n");
    for(int i = hostCsrInfo.startNodeID ; i <= hostCsrInfo.endNodeID ; i ++){
        // printf("node[%d] = {startOffset = %d, endOffset = %d}\n", i, hostNodeInfo[i].nodeStartOffset, hostNodeInfo[i].nodeEndOffset);
        if(csr->csrNodesDegree[i] >= 32){
            count ++;
        }
        // printf("node[%d].color = %d\n", i, hostColorArr[i]);
        // printf("%d\n", i);
    }
    //檢查WorkList中
    for(int i = 0 ; i < *hostWorkListNowIndex ; i ++){
        int nodeID          = hostWorkList[i];
        int nodeColor       = hostColorArr[nodeID];
        int flag            = 0;
        for(int neighborIndex = csr->csrV[nodeID] ; neighborIndex < csr->csrV[nodeID + 1] ; neighborIndex ++){
            int neighborID = csr->csrE[neighborIndex];
            int neighborColor = hostColorArr[neighborID];
            if(nodeColor == neighborColor){
                printf("\n\nnode[%d].color = %d, neighbor[%d].color = %d!!!!!!!!!!!!!!!!!\n\n", nodeID, nodeColor, neighborID, neighborColor);
                flag = 1;
                break;
            }
        }
        if(flag == 1){break;}
    }
    //亂數
    time_t t;
    srand(time(&t));
    // int arbitraryNode = random() % nodeSize + 1;
    // int arbitraryNode = 104410;
    // int arbitraryNode = 96457; //LiveJournal中，bitmap用量較多的一個node
    // int arbitraryNode = 3984674;

   
    // for(int nodeIter = hostCsrInfo.startNodeID ; nodeIter <= hostCsrInfo.endNodeID ; nodeIter ++){
    //     if(hostNodeInfo[nodeIter].possColorBitLength == 34){
    //         arbitraryNode = nodeIter;
    //     }
    // }

    //hostDAG_E有被建立好!
    // int node1 = 553341;
    // int node2 = 1193315;
    // int node3 = 2274;
    // checkOneNodeBitmap(hostNodeInfo, csr, hostNodeBitmap, hostDAG_E, node1);
    // checkOneNodeBitmap(hostNodeInfo, csr, hostNodeBitmap, hostDAG_E, node2);
    // checkOneNodeBitmap(hostNodeInfo, csr, hostNodeBitmap, hostDAG_E, node3);
    // printf("[DAG_E  Checking] : ");
    // printf("node[%d] = {", arbitraryNode);
    // for(int offset = hostNodeInfo[arbitraryNode].nodeStartOffset ; offset < hostNodeInfo[arbitraryNode].nodeEndOffset ; offset ++){
    //     printf("%d, ", hostDAG_E[offset]);
    // }
    // printf("}\n");
    // printf("[Bitmap Checking] : node[%d].length = %d", arbitraryNode, hostNodeInfo[arbitraryNode].possColorBitLength);
    // if(hostNodeInfo[arbitraryNode].possColorBitLength == 0)
    //     printf(", degree = %d\n", csr->csrNodesDegree[arbitraryNode]);
    // else
    //     printf("\n");
    // printf("node[%d] = {", arbitraryNode);
    // for(int intIter = 0 ; intIter < ((csr->maxDegree + 32) / 32) ; intIter ++){
    //     printf("%u, ", hostNodeBitmap[hostNodeInfo[arbitraryNode].bitmapStartOffset + intIter]);
    // }
    // printf("}\n");

    printf("\nWorkListNowIndex = %u, HostCount = %d\n", *hostWorkListNowIndex, count);
    #endif
    
}


int reAssignNodeSize(CSR* _csr){
    int nodeSize = 0;
    if(_csr->startAtZero){
        nodeSize = _csr->csrVSize - 1;
        // printf("Start at 0, nodeSize = %d\n", nodeSize);
    }
    else{
        nodeSize = _csr->csrVSize - 2;
        // printf("Start at 1, nodeSize = %d\n", nodeSize);
    }
    return nodeSize;
}

CSRInfo getCsrInfo(CSR* _csr, int _nodeSize){
    CSRInfo hostCsrInfo;
    hostCsrInfo.nodeSize            = _nodeSize;                            //紀錄總共有幾個Node
    hostCsrInfo.startAtZero         = _csr->startAtZero;
    hostCsrInfo.startNodeID         = !hostCsrInfo.startAtZero;             //紀錄起始的NodeID
    hostCsrInfo.endNodeID           = _nodeSize - hostCsrInfo.startAtZero; //紀錄結束的NodeID
    hostCsrInfo.maxDegree           = _csr->maxDegree;
    hostCsrInfo.nodeBitmapIntNum    = (_csr->maxDegree + 32) / 32;
    printf("hostCsrInfo = {startNodeID = %d, endNodeID = %d, nodeSize = %d, maxDegree = %d}\n", hostCsrInfo.startNodeID, hostCsrInfo.endNodeID, hostCsrInfo.nodeSize, hostCsrInfo.maxDegree);
    return hostCsrInfo;
}

void checkDevice(){
    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("device %d : %s\n", dev, deviceProp.name);
    cudaDeviceReset();
}

void checkOneNodeBitmap(struct NodeInfo* _hostNodeInfo, struct CSR* _csr, unsigned int* _hostNodeBitmap, int* _hostDAG_E, int _nodeID){
    printf("[DAG_E  Checking] : ");
    printf("node[%d] = {", _nodeID);
    for(int offset = _hostNodeInfo[_nodeID].nodeStartOffset ; offset < _hostNodeInfo[_nodeID].nodeEndOffset ; offset ++){
        printf("%d, ", _hostDAG_E[offset]);
    }
    printf("}\n");
    printf("[Bitmap Checking] : node[%d].length = %d", _nodeID, _hostNodeInfo[_nodeID].possColorBitLength);
    if(_hostNodeInfo[_nodeID].possColorBitLength == 0)
        printf(", degree = %d\n", _csr->csrNodesDegree[_nodeID]);
    else
        printf("\n");
    printf("node[%d] = {", _nodeID);
    for(int intIter = 0 ; intIter < ((_csr->maxDegree + 32) / 32) ; intIter ++){
        printf("%u, ", _hostNodeBitmap[_hostNodeInfo[_nodeID].bitmapStartOffset + intIter]);
    }
    printf("}\n\n");
}

#pragma region ECL_GC_Init
__device__ void BitmapSetOperation(int _nodeID, unsigned int** _cudaNodeBitmap, struct NodeInfo* _cudaNodeInfo, struct CSRInfo* _deviceCsrInfo)
{
    int eachNodeIntElementSize  = (_deviceCsrInfo->maxDegree + 32) / 32;
    _cudaNodeInfo[_nodeID].bitmapStartOffset = _nodeID * eachNodeIntElementSize;
    
    int possBitOccupyIntNum     = _cudaNodeInfo[_nodeID].possColorBitLength / 32;
    int possBitRemainLength     = _cudaNodeInfo[_nodeID].possColorBitLength % 32;
    // if(_nodeID == 96457){
    //     printf("node[%d].PossBitOccupyIntNum = %d\n", _nodeID, possBitOccupyIntNum);
    // }
    //把一次超過32bit的1，用for loop assign
    int intIter;
    for(intIter = 0 ; intIter < possBitOccupyIntNum ; intIter ++){
        (*_cudaNodeBitmap)[_cudaNodeInfo[_nodeID].bitmapStartOffset + intIter] = 0xffffffff;
        // if(_nodeID == 96457){
        //     printf("intIter = %d, bitmap = %u\n", intIter, (*_cudaNodeBitmap)[_cudaNodeInfo[_nodeID].bitmapStartOffset + intIter]);
        // }
    }

    //把剩下的bit也assign進去
    unsigned int remainBit = 0;
    for(int i = 0 ; i < possBitRemainLength ; i ++){
        remainBit |= (1 << i);
    }
    (*_cudaNodeBitmap)[_cudaNodeInfo[_nodeID].bitmapStartOffset + intIter] |= remainBit;
    
    
    if(_nodeID == 307071){
        int best = 0;
        int worst = 0;
        best = bitmapGetBestColor(_nodeID, *_cudaNodeBitmap, _cudaNodeInfo, _deviceCsrInfo);
        worst = bitmapGetWorstColor(_nodeID, *_cudaNodeBitmap, _cudaNodeInfo, _deviceCsrInfo);
        printf("\t\tnode[%d] = {bestColor = %d, WorstColor = %d}\n", _nodeID, best, worst);
    }
    // if(_nodeID == 3984674){
    //     bitmapRemoveOnebit(_nodeID, _cudaNodeBitmap, _cudaNodeInfo, 1, _deviceCsrInfo);
    // }
    // if(_nodeID == 96457){ //在LiveJournal中的奇怪case
    //     printf("node[%d] = {bitmapStartOffset = %d, BitLength = %d, Bitmap = %u}\n", _nodeID, _cudaNodeInfo[_nodeID].bitmapStartOffset, _cudaNodeInfo[_nodeID].possColorBitLength, (*_cudaNodeBitmap)[_cudaNodeInfo[_nodeID].bitmapStartOffset + intIter]);
    // }
}

__global__ void Init(int* _cudaCsrV,             int* _cudaCsrE,         int* _cudaWorkList,
                     int* _cudaWorkListNowIndex, int* _cudaCsrDegree,    int* _cudaColorArr,
                     int* _cudaDAG_E,            unsigned int* _cudaNodeBitmap,
                     struct NodeInfo* _cudaNodeInfo, struct CSRInfo* _deviceCsrInfo)
{
                                
    int tid             = threadIdx.x + blockIdx.x * blockDim.x;
    int nodeID          = tid;
    int nodeDegree      = _cudaCsrDegree[nodeID];
    
    if(_deviceCsrInfo->startNodeID <= nodeID && nodeID <= _deviceCsrInfo->endNodeID){
        // _cudaColorArr的初始化已經用cudaMemset做了
        // _cudaColorArr[nodeID] = 0;
        
        if(_cudaCsrDegree[nodeID]  >= 32){
            int NowIndex            = atomicAdd(_cudaWorkListNowIndex, 1);
            _cudaWorkList[NowIndex] = nodeID;
            // printf("NowIndex = %d, Node = %d\n", NowIndex, nodeID);
        }
        else{
            int neighborID      = -1;
            int neighborDegree  = -1;
            _cudaNodeInfo[nodeID].nodeStartOffset   = _cudaCsrV[nodeID];
            _cudaNodeInfo[nodeID].nodeEndOffset     = _cudaCsrV[nodeID];
            for(int offsetIter = _cudaCsrV[nodeID] ; offsetIter < _cudaCsrV[nodeID + 1] ; offsetIter ++){
                neighborID      = _cudaCsrE[offsetIter];
                neighborDegree  = _cudaCsrDegree[neighborID];
                if(nodeDegree < neighborDegree){
                    _cudaDAG_E[_cudaNodeInfo[nodeID].nodeEndOffset] = neighborID;
                    _cudaNodeInfo[nodeID].nodeEndOffset ++;
                    // printf("node[%d] = {nowOffset = %d, neighbor = %d}\n", nodeID, _cudaNodeInfo[nodeID].nodeEndOffset, neighborID);
                    //用以下方式使用offset
                    /*
                    for(int offset = _cudaNodeInfo[nodeID].nodeStartOffset ; offset < _cudaNodeInfo[nodeID].nodeEndOffset ; offset ++){
                        //do something
                    }
                    */
                }
                else if(nodeDegree == neighborDegree){ //Breaktie by nodeID and neighborID
                    if(nodeID > neighborID){
                        _cudaDAG_E[_cudaNodeInfo[nodeID].nodeEndOffset] = neighborID;
                        _cudaNodeInfo[nodeID].nodeEndOffset ++;
                    }
                }
            }
            #ifdef _DEBUG_
            // printf("node[%d] = {startOffset = %d, endOffset = %d}\n", nodeID, _cudaNodeInfo[nodeID].nodeStartOffset, _cudaNodeInfo[nodeID].nodeEndOffset);
            #endif
            _cudaNodeInfo[nodeID].possColorBitLength    = _cudaNodeInfo[nodeID].nodeEndOffset - _cudaNodeInfo[nodeID].nodeStartOffset + 1;
            _cudaNodeInfo[nodeID].bestColor             = 0;
            _cudaNodeInfo[nodeID].worstColor            = _cudaNodeInfo[nodeID].possColorBitLength;
            BitmapSetOperation(nodeID, &_cudaNodeBitmap, _cudaNodeInfo, _deviceCsrInfo);
            //初始化_cudaDAG_E已在cudaMemcpy完成
            //Call ECL_GC_InitSub的時候，設置1個block = 32個thread，1個grid = 1個block就可以了
            // ECL_GC_InitSub<<<1, 32>>>(nodeID, _cudaCsrV, _cudaCsrE, _cudaCsrDegree, _cudaDAG_E, &subKernelFlag, _cudaDAG_nodeEndOffset, _cudaNodeInfo);
            // //這裡卡不住
            // printf("node[%d] = {startOffset = %d, endOffset = %d}\n", nodeID, _cudaNodeInfo[nodeID].nodeStartOffset, _cudaDAG_nodeEndOffset[nodeID]);
        }
    }
}

__global__ void Worklist_DAG_Init(int* _cudaWorkList,   int* _cudaWorkListLength,    int* _cudaCsrDegree,
                                  int* _cudaCsrV,       int* _cudaCsrE,             struct NodeInfo* _cudaNodeInfo,
                                  int* _cudaDAG_E,      unsigned int* _cudaNodeBitmap,  struct CSRInfo* _deviceCsrInfo)
{   
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid < *_cudaWorkListLength){
        int nodeID          = _cudaWorkList[tid];
        int nodeDegree      = _cudaCsrDegree[nodeID];
        
        int neighborID      = -1;
        int neighborDegree  = -1;
        
        _cudaNodeInfo[nodeID].nodeStartOffset   = _cudaCsrV[nodeID];
        _cudaNodeInfo[nodeID].nodeEndOffset     = _cudaCsrV[nodeID];

        for(int offsetIter = _cudaCsrV[nodeID] ; offsetIter < _cudaCsrV[nodeID + 1] ; offsetIter ++){
            neighborID      = _cudaCsrE[offsetIter];
            neighborDegree  = _cudaCsrDegree[neighborID];

            if(nodeDegree < neighborDegree){
                _cudaDAG_E[_cudaNodeInfo[nodeID].nodeEndOffset] = neighborID;
                _cudaNodeInfo[nodeID].nodeEndOffset ++;
            }
            else if(nodeDegree == neighborDegree){
                if(nodeID > neighborID){
                    _cudaDAG_E[_cudaNodeInfo[nodeID].nodeEndOffset] = neighborID;
                    _cudaNodeInfo[nodeID].nodeEndOffset ++;
                }
            }
        }
        _cudaNodeInfo[nodeID].possColorBitLength    = _cudaNodeInfo[nodeID].nodeEndOffset - _cudaNodeInfo[nodeID].nodeStartOffset + 1;
        _cudaNodeInfo[nodeID].bestColor             = 0;
        _cudaNodeInfo[nodeID].worstColor            = _cudaNodeInfo[nodeID].possColorBitLength;
        BitmapSetOperation(nodeID, &_cudaNodeBitmap, _cudaNodeInfo, _deviceCsrInfo);
    }
}

void ECL_GC_Init(   int* _cudaCsrV, int* _cudaCsrE, int* _cudaWorkList,
                    int* _cudaWorkListNowIndex, int* _cudaCsrDegree, int* _cudaColorArr,
                    int* _cudaDAG_E, unsigned int* _cudaNodeBitmap, struct NodeInfo* _cudaNodeInfo,
                    struct CSRInfo* _deviceCsrInfo, dim3 _block, dim3 _grid)
{
    printf("=============In Device============\n");
    printf("[Execution][ECL_GC_Init]...\n");

    printf("\t[Execution][Step1]Init...\n");
    Init<<<_grid, _block>>>(  _cudaCsrV,   _cudaCsrE,   _cudaWorkList,
                            _cudaWorkListNowIndex,   _cudaCsrDegree, _cudaColorArr,
                            _cudaDAG_E,  _cudaNodeBitmap, _cudaNodeInfo,   _deviceCsrInfo);
    cudaDeviceSynchronize();

    printf("\t[Finish][Step1]Init~\n");
    printf("\n");

    printf("\t[Execution][Step2]Worklist_DAG_Init...\n");
    Worklist_DAG_Init<<<_grid, _block>>>( _cudaWorkList,   _cudaWorkListNowIndex,   _cudaCsrDegree,  _cudaCsrV,   _cudaCsrE, 
                                        _cudaNodeInfo,   _cudaDAG_E,  _cudaNodeBitmap, _deviceCsrInfo);
    cudaDeviceSynchronize();
    printf("\t[Finish][Step2]Worklist_DAG_Init~\n");

    printf("[Finish][ECL_GC_Init]~\n\n");
}
#pragma endregion //ECL_GC_Init

__device__ void bitmapRemoveOnebit( int _nodeID, unsigned int** _cudaNodeBitmap, struct NodeInfo* _cudaNodeInfo,
                                    int targetBitLocation, struct CSRInfo* _deviceCsrInfo)
{
    int intLocation             = (targetBitLocation - 1) / 32;
    int bitLocationInSingleInt  = (targetBitLocation - 1) % 32;

    unsigned int removeBit      = 1 << bitLocationInSingleInt;
    (*_cudaNodeBitmap)[_cudaNodeInfo[_nodeID].bitmapStartOffset + intLocation] &= ~removeBit;
}

__device__ int bitmapGetBestColor(  int _nodeID, unsigned int* _cudaNodeBitmap, struct NodeInfo* _cudaNodeInfo,
                                    struct CSRInfo* _deviceCsrInfo)
{
    int intIter                 = 0;
    int bestColor               = 0;
    unsigned int unsignValue    = 0;
    for(intIter = 0 ; intIter < _deviceCsrInfo->nodeBitmapIntNum ; intIter ++){
        unsignValue = _cudaNodeBitmap[_cudaNodeInfo[_nodeID].bitmapStartOffset + intIter];
        bestColor   = __ffsll(unsignValue); //取得該int的 Least set one 的 index， "1010"會取到 2
        if(bestColor != 0){break;} 
    }
    bestColor      += intIter * 32;
    return bestColor;
}

__device__ int bitmapGetWorstColor( int _nodeID, unsigned int* _cudaNodeBitmap, struct NodeInfo* _cudaNodeInfo,
                                    struct CSRInfo* _deviceCsrInfo)
{
    int intIter                 = (_cudaNodeInfo[_nodeID].possColorBitLength - 1) / 32;
    int worstColor              = 0;
    unsigned int unsignValue    = 0;
    for(; intIter >= 0 ; intIter --){
        unsignValue     = _cudaNodeBitmap[_cudaNodeInfo[_nodeID].bitmapStartOffset + intIter];
        //Get most significant bit 1
        unsignValue    |= (unsignValue >> 1);
        unsignValue    |= (unsignValue >> 2);
        unsignValue    |= (unsignValue >> 4);
        unsignValue    |= (unsignValue >> 8);
        unsignValue    |= (unsignValue >> 16);

        unsignValue     = ((unsignValue + 1) >> 1) | (unsignValue & (1 << 31));
        worstColor      = __ffsll(unsignValue); //取得該int的Least set one的index，但這裡會取到 worst index，因為上面的處理
        if(worstColor != 0){break;}
    }
    worstColor += intIter * 32;
    return worstColor;
}

__global__ void HighDegreeColoring( int* _cudaWorkList, int* _cudaWorkListLength, int* _cudaColorArr, int* _cudaDAG_E, struct NodeInfo* _cudaNodeInfo,
                                    int* _againFlag,    unsigned int* _cudaNodeBitmap, struct CSRInfo* _deviceCsrInfo)
{
    int tid         = threadIdx.x + blockIdx.x * blockDim.x;
    int shortcut    = 0;    //紀錄是否可用shortcut，如果shortcut = 1則代表可用shortcut
    int done        = 0;    //紀錄全部的鄰居是否都已塗色
    // if(tid == 0){
    //     printf("tid = %d, cudaWorkListLength = %d\n", tid, *_cudaWorkListLength);
    // }
    if(tid < *_cudaWorkListLength){
        int nodeID = _cudaWorkList[tid];
        if(nodeID == 3976); //bug 有可能會塗到相同顏色
        if(nodeID == 3650); //bug 有可能會塗到相同顏色
        if(_cudaColorArr[nodeID] == -1){    //如果node還沒塗色
            shortcut    = 1;
            done        = 1;

            int nodeBestColor       = _cudaNodeInfo[nodeID].bestColor;
            // printf("node[%d] = {bestcolor = %d}\n", nodeID, nodeBestColor);

            int neighborID          = -1;
            int neighborBestColor   = -1;
            int neighborWorstColor  = -1;

            //Checkout neighbors' color
            for(int offsetIter = _cudaNodeInfo[nodeID].nodeStartOffset ; offsetIter < _cudaNodeInfo[nodeID].nodeEndOffset ; offsetIter ++){
                neighborID  = _cudaDAG_E[offsetIter];
                if(_cudaColorArr[neighborID] != -1){    //如果neighbor已塗色
                    if(_cudaColorArr[neighborID] == nodeBestColor){
                        shortcut = 0;
                    }
                    //remove a bit of possible color of nodeID
                    bitmapRemoveOnebit(nodeID, &_cudaNodeBitmap, _cudaNodeInfo, _cudaColorArr[neighborID], _deviceCsrInfo);
                    nodeBestColor = bitmapGetBestColor(nodeID, _cudaNodeBitmap, _cudaNodeInfo, _deviceCsrInfo);
                }
                else{
                    done                = 0;
                    neighborBestColor   = bitmapGetBestColor(neighborID, _cudaNodeBitmap, _cudaNodeInfo, _deviceCsrInfo);
                    neighborWorstColor  = bitmapGetWorstColor(neighborID, _cudaNodeBitmap, _cudaNodeInfo, _deviceCsrInfo);

                    nodeBestColor       = bitmapGetBestColor(nodeID, _cudaNodeBitmap, _cudaNodeInfo, _deviceCsrInfo);
                    if((neighborBestColor <= nodeBestColor) && (nodeBestColor <= neighborWorstColor)){
                        shortcut        = 0;
                    }
                }
            }

            // _cudaNodeInfo[nodeID].bestColor = bitmapGetBestColor(nodeID, _cudaNodeBitmap, _cudaNodeInfo, _deviceCsrInfo);
            nodeBestColor               = bitmapGetBestColor(nodeID, _cudaNodeBitmap, _cudaNodeInfo, _deviceCsrInfo);

            if(done || shortcut){
                _cudaColorArr[nodeID]   = nodeBestColor;
                // _cudaColorArr[nodeID] = _cudaNodeInfo[nodeID].bestColor;
                // if(nodeID == 553341 || nodeID == 1193315){
                //     printf("nodeID = %d, nodeBestColor = %d, cudaNodeInfo.bestColor = %d, neighborID = %d, neighborBestColor = %d, neighborWorstColor = %d\n", nodeID, nodeBestColor, _cudaNodeInfo[nodeID].bestColor, neighborID, neighborBestColor, neighborWorstColor);
                // }
                // printf("node[%d].colored = %d\n", nodeID, _cudaColorArr[nodeID]);
            }
            else{
                *_againFlag = 1;
            }
        }
    }
}

void ECL_GC_HighDegree_Vertex_Coloring( int* _cudaWorkList, int* _cudaWorkListLength, int* _cudaColorArr, int* _cudaDAG_E,
                                        struct NodeInfo* _cudaNodeInfo, unsigned int* _cudaNodeBitmap, struct CSRInfo* _deviceCsrInfo,
                                        dim3 _block, dim3 _grid)
{
    printf("[Execution][ECL_GC_HighDegree_Vertex_Coloring]...\n");
    int* hostAgainFlag = (int*)malloc(sizeof(int));

    int* cudaAgainFlag;
    cudaMalloc((void**)&cudaAgainFlag, sizeof(int));
    cudaMemset(cudaAgainFlag, 0, sizeof(int));
    
    int ECL_GC_HighDegree_Vertex_Coloring_Count = 0;
    do
    {
        *hostAgainFlag = 0;
        cudaMemset(cudaAgainFlag, 0, sizeof(int));

        HighDegreeColoring<<<_grid, _block>>>(  _cudaWorkList, _cudaWorkListLength, _cudaColorArr,
                                                _cudaDAG_E, _cudaNodeInfo, cudaAgainFlag, 
                                                _cudaNodeBitmap, _deviceCsrInfo);
        cudaDeviceSynchronize();

        cudaMemcpy(hostAgainFlag, cudaAgainFlag, sizeof(int), cudaMemcpyDeviceToHost);

        ECL_GC_HighDegree_Vertex_Coloring_Count ++;
    } while (*hostAgainFlag == 1);
    printf("\tcount = %d\n", ECL_GC_HighDegree_Vertex_Coloring_Count);
    cudaDeviceSynchronize();

    printf("[Finish][ECL_GC_HighDegree_Vertex_Coloring]~\n\n");
}

__global__ void ECL_GC_InitSub( int _nodeID,            int* _cudaCsrV,     int* _cudaCsrE,
                                int* _cudaCsrDegree,    int* _cudaDAG_E,    int* _subKernelFlag,
                                int* _cudaDAG_nodeEndOffset, struct NodeInfo* _cudaDAG_E_Offset)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    // degree 18
    // tid    0~31 只有 0~17有用，tid >= 18的都可以滾，只要tid < 18的做事就好
    if(tid < _cudaCsrDegree[_nodeID]){


        int neighborIndex   = tid;
        int neighborID      = _cudaCsrE[_cudaCsrV[_nodeID] + neighborIndex];
        
        int nodeDegree      = _cudaCsrDegree[_nodeID];
        int neighborDegree  = _cudaCsrDegree[neighborID];
        printf("node[%d] = {neighborIndex = %d, neighborID = %d, nodeDegree = %d, neighborDegree = %d}\n", _nodeID, neighborIndex, neighborID, nodeDegree, neighborDegree);
        
        if(tid == 0){
            //設置node在DAG_E的起始offset
            _cudaDAG_E_Offset[_nodeID].nodeStartOffset  = _cudaCsrV[_nodeID];
            //設置nodeEndOffset成起始的offset，且它之後會一直atomicAdd
            _cudaDAG_nodeEndOffset[_nodeID]             = _cudaCsrV[_nodeID]; 
        }
        // printf("cudaDAG_E_Offset[%d] = {startOffset = %d, EndOffset = %d}\n", _nodeID, _cudaDAG_E_Offset[_nodeID].nodeStartOffset, _cudaDAG_E_Offset[_nodeID].nodeEndOffset);
        //比較優先權(比較nodeDegree跟neighborDegree)
        if(nodeDegree < neighborDegree){
            //atomicAdd有問題，在這裡有+，但是不會去改到原本記憶體的值
            int nowOffset           = atomicAdd(&(_cudaDAG_nodeEndOffset[_nodeID]), 1);
            _cudaDAG_E[nowOffset]   = neighborID;
            printf("node[%d] = {nowOffset = %d, neighbor = %d}\n", _nodeID, nowOffset, neighborID);
        }
        else if(nodeDegree == neighborDegree){ //break tie by nodeID and neighborID
            // if nodeID > neighborID，則代表neighbor會先塗，故neighbor優先於node，所以node的DAG要記錄neighbor
            if(_nodeID > neighborID){
            //atomicAdd有問題，在這裡有+，但是不會去改到原本記憶體的值
                int nowOffset           = atomicAdd(&(_cudaDAG_nodeEndOffset[_nodeID]), 1);
                _cudaDAG_E[nowOffset]   = neighborID;
                printf("node[%d] = {nowOffset = %d, neighbor = %d}\n", _nodeID, nowOffset, neighborID);
            }
            // else if nodeID < neighborID，則代表node會先塗，故node優先於neighbor，所以不用記錄東西
        }
    }

    if(tid == 0){
        printf("============node[%d] Finish\n", _nodeID);
        // *_subKernelFlag = 1;
    }
}
