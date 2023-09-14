/**
 * @author XPPGX
 * @date 2023/07/15
*/

#include "AdjList.h"

//default the graph is undirected graph

#define _DEBUG_
#define rowSize 2000

struct Graph* buildGraph(char* _datasetPath){
    struct Graph* graph = (struct Graph*)malloc(sizeof(struct Graph));
    FILE* fptr = fopen(_datasetPath, "r");
    if(fptr == NULL){
        printf("Error: Open file\n");
        exit(1);
    }
    char row[rowSize];
    int val1, val2 = 0;
    fgets(row, rowSize, fptr);
    
    getRowData(row, &val1, &val2);

    #ifdef _DEBUG_
    printf("nodeNum = %d, edgeNum = %d\n", val1, val2);
    #endif

    graph->nodeNum = val1;
    graph->edgeNum = val2;
    
    //read first edge to check 0 or 1 is start index.
    fgets(row, rowSize, fptr);
    getRowData(row, &val1, &val2);
    if(val1 == 0 || val2 == 0){
        #ifdef _DEBUG_
        printf("Graph start with 0\n");
        #endif

        graph->startAtZero = 1;
        graph->vertices = (struct adjList*)malloc(sizeof(struct adjList) * graph->nodeNum);
        for(int i = 0 ; i < graph->nodeNum ; i ++){
            graph->vertices[i].neighbors = InitvVector();
        }
    }
    else{
        #ifdef _DEBUG_
        printf("Graph start with 1\n");
        #endif

        graph->startAtZero = 0;
        graph->vertices = (struct adjList*)malloc(sizeof(struct adjList) * (graph->nodeNum + 1));
        for(int i = 0 ; i < graph->nodeNum + 1 ; i ++){
            graph->vertices[i].neighbors = InitvVector();
        }
    }

    //append first edge into adjlist
    vAppend(graph->vertices[val1].neighbors, val2);
    vAppend(graph->vertices[val2].neighbors, val1);
    //run over all remain edges
    for(int i = 1 ; i < graph->edgeNum ; i ++){
        fgets(row, rowSize, fptr);
        getRowData(row, &val1, &val2);
        vAppend(graph->vertices[val1].neighbors, val2);
        vAppend(graph->vertices[val2].neighbors, val1);
    }
    printf("Success : Build AdjList\n");
    printf("==============================\n");
    printf("dataset = %s\n", _datasetPath);
    printf("nodeNum = %d\n", graph->nodeNum);
    printf("edgeNum = %d\n", graph->edgeNum);
    switch(graph->startAtZero){
        case 0:
            printf("graph start with 1\n");
            break;
        case 1:
            printf("graph start with 0\n");
            break;
    }
    printf("==============================\n");
    return graph;
}

void showAdjList(struct Graph* _graph){
    if(_graph->startAtZero == 1){
        printf("Graph start at zero\n");
        for(int i = 0 ; i < _graph->nodeNum ; i ++){
            printf("neighbor[%d] = {", i);
            for(int neighbor_index = 0 ; neighbor_index <= _graph->vertices[i].neighbors->tail ; neighbor_index++){
                printf("%d, ", _graph->vertices[i].neighbors->dataArr[neighbor_index]);
            }
            printf("}\n");
        }
    }
    else{
        printf("Graph start at one\n");
        for(int i = 1 ; i < _graph->nodeNum + 1 ; i ++){
            printf("neighbor[%d] = {", i);
            for(int neighbor_index = 0 ; neighbor_index <= _graph->vertices[i].neighbors->tail ; neighbor_index++){
                printf("%d, ", _graph->vertices[i].neighbors->dataArr[neighbor_index]);
            }
            printf("}\n");
        }
    }
}