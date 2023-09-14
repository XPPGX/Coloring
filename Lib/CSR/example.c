#include "CSR.h"

int main(int argc, char* argv[]){
    char* datasetPath = argv[1];
    struct Graph* adjList = buildGraph(datasetPath);
    struct CSR* csr = createCSR(adjList);
    showCSR(csr);
    int nodeID = 0;
    if(csr->startAtZero == 0){nodeID = 1;}
    for( ; nodeID < csr->csrVSize ; nodeID ++){
        printf("vertex = %d, startIndex = %d, endIndex = %d, neighborNum = %d\n", nodeID, csr->csrV[nodeID], csr->csrV[nodeID + 1] - 1, csr->csrV[nodeID + 1] - csr->csrV[nodeID]);
    }
    // showAdjList(adjList);
}