/**
 * @author XPPGX
 * @date 2023/07/15
*/

#ifndef COMMON
#define COMMON

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#endif

//content
#ifndef VVECTOR
#define VVECTOR
#include "../vVector/vVector.h"
#endif

#ifndef FILEREADER
#define FILEREADER
#include "../FileReader/FileReader.h"
#endif

#ifndef QQueue
#define QQueue
#include "../qQueue/qQueue.h"
#endif
struct adjList{
    struct vVector* neighbors;
};

struct Graph{
    int nodeNum;
    int edgeNum;
    int startAtZero;
    int* nodeDegrees;
    struct adjList* vertices;
    struct qQueue* degreeOneQueue;
};

struct Graph* buildGraph(char* _datasetPath);
void quicksort(struct vVector* _neighbors, int* _nodeDegrees, int _left, int _right);
void showAdjList(struct Graph* _graph);
