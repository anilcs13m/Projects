 #include <sys/time.h>
#include <iostream>
#include <limits.h>
#include <stdlib.h>
 
// Number of vertices in the graph
#define V 50
using namespace std;
 
// A utility function to find the vertex with minimum distance value, from
// the set of vertices not yet included in shortest path tree
int minDistance(int dist[], bool sptSet[])
{
   int min = INT_MAX, min_index;
 
   for (int v = 0; v < V; v++)
     if (sptSet[v] == false && dist[v] <= min)
         min = dist[v], min_index = v;
 
   return min_index;
}
 


void updateAllDijkstra(bool allDijkstra[][V], int parent[], int terminal[], int numberOfTerminals,int src)
{
	for(int i=0; i<V; ++i)
		cout<<parent[i]<<" ";cout<<endl;
	for(int i=0; i < numberOfTerminals; ++i)
	{
		if( terminal[i] != src )
		{
			int j=terminal[i];
			while(parent[j] != -1 )
			{
				allDijkstra[parent[j]][j] = 1;
				allDijkstra[j][parent[j]] = 1;
				j=parent[j];//cout<<"sud"<<endl;
			}
		}
	}
}
 
// Funtion that implements Dijkstra's single source shortest path algorithm
int* dijkstra(int graph[V][V], int src)
{
     int dist[V];     
                      
     //int parent[V];
     int *parent;
     parent = (int*)malloc(V*sizeof(int));
     parent[src] = -1;
 
     bool sptSet[V]; 
                    
 
     
     for (int i = 0; i < V; i++)
        dist[i] = INT_MAX, sptSet[i] = false;
 
     
     dist[src] = 0;
 
     
     for (int count = 0; count < V-1; count++)
     {
      
       int u = minDistance(dist, sptSet);
 
       sptSet[u] = true;
 
       for (int v = 0; v < V; v++)
 
         if (!sptSet[v] && (graph[u][v] != 0) && dist[u] != INT_MAX 
                                       && dist[u]+graph[u][v] < dist[v])
         {
            dist[v] = dist[u] + graph[u][v];
            parent[v] = u;
         }
     }
 
     return parent;
}
 
// driver program to test above function
int main()
{
   /* Let us create the example graph discussed above */
  string str;
	cin>>str;
	cout<<str;
	while(str != "Nodes")
	{
		cin>>str;
		cout<<str<<endl;
	}
	int numberOfNodes;
	cin>>numberOfNodes;
	cout<<numberOfNodes<<endl;
	int numberOfEdges;
	cin>>str;
	cout<<str<<endl;
	cin>>numberOfEdges;
	cout<<numberOfEdges<<endl;
	//vector<vector<int>> graph(numberOfNodes, vector<int>(numberOfNodes));
	//int graph[numberOfNodes][numberOfNodes];
	int graph[V][V]={0};
	cin>>str;
	cout<<str;
	int from, to, weight;
	while(str == "E")
	{
		cin>>from>>to>>weight;
		cout<<" "<<from<<" "<<to<<" "<<weight<<endl;
		graph[from-1][to-1] = weight;
		graph[to-1][from-1] = weight;
		cin>>str;
		//cout<<str;
	}
	int numberOfTerminals;
	while(str != "Terminals")
	{
		cin>>str;
		cout<<str<<endl;
	}
	cin>>numberOfTerminals;
	cout<<" "<<numberOfTerminals<<endl;
	//vector<int> terminals;
	int terminals[numberOfTerminals];
	int terminal;
	int i=-1;
	cin>>str;
	cout<<str;
	while(str == "T")
	{
		cin>>terminal;
		cout<<" "<<terminal<<endl;
		//terminals.push_back(terminal);
		terminals[++i] = terminal-1;
		cin>>str;
		cout<<str;
	}
	cout<<"Done!\n";

  // for(int i=0; i < V; ++i)
  // {
  //   for(int j=0; j < V; ++j)
  //   cout<<graph[i][j]<<" ";
  //   cout<<endl;
  // }
  // cout<<"done"<<endl;
 	// V = numberOfVertices;
 	bool allDijkstra[V][V]={0};
   // cout<<"Sudheer"<<endl;
 	 // cout<<"Terminals::";
     struct timeval start, present;
     gettimeofday(&start,NULL);
     double start_time = start.tv_sec + start.tv_usec/1000000.0 ;

    for(int i=0; i < numberOfTerminals; ++i) // find all dijkstra paths and combines.
    {
    	int *parent = dijkstra(graph,terminals[i]); // finding single source shortest path.
    	updateAllDijkstra(allDijkstra,parent,terminals,numberOfTerminals,i); // merging the above line resultunt graph with previous resultant paths.
    	free(parent);
    }
    
    for(int i=0; i < V; ++i) // updating actual graph( here we will remove the nodes which doesn't involve in finding all single source paths for all terminals.
    for(int j=0; j < V; ++j)
    {
    	if(allDijkstra[i][j] == 0)
    	{
    		graph[i][j] = 0;
    	}
    }
   
   
   // for(int i=0; i < V; ++i)// just printing.
   //  for(int j=0; j < V; ++j)
   //  	if(graph[i][j] != 0)
   //  		cout<<graph[i][j]<<endl;
    		
   int *parent = dijkstra(graph,terminals[0]);//returns array with spanning tree in the form of parent values.
   bool temp[V][V] = {0};
   for(int i=0; i < V; ++i)
   	if( parent[i] != -1)
   	{
   		temp[parent[i]][i] = 1;	
   		temp[i][parent[i]] = 1; 
   	}
   
   for(int i=0; i < V; ++i) // updating the graph with above spanning tree nodes.
    for(int j=0; j < V; ++j)
    	if(temp[i][j] == 0)
    		graph[i][j] = 0;
 	cout<<"Final:"<<endl;
  
  for(int i=0; i < V; ++i)//just printing.
    for(int j=i; j < V; ++j)
    	if(graph[i][j] != 0)
    		cout<<graph[i][j]<<endl;
    	
   bool temp1[V][V]={0};	
   for(int i=1; i < numberOfTerminals; ++i) // dont traverse from termianl 0. bcz, its the source.
   {					// removing the leaf nodes which are not terminals.
   	  int j = terminals[i];
   	  while(parent[j] != -1 )
	   {
		    temp1[parent[j]][j] = 1;
		    temp1[j][parent[j]] = 1;
		    j=parent[j];//cout<<"sud"<<endl;
	   }
   }
   for(int i=0; i < V; ++i) // updating the graph with the above results.
    for(int j=0; j < V; ++j)
    	if(temp1[i][j] == 0)
    		graph[i][j] = 0;
   
  //  for(int i = 0; i < V; i++)
  // {
  //   for(int j = 0; j < V; j++)
  //   {
  //     cout<<graph[i][j]<<" ";
  //   }  
  //   cout<<endl;
  // }
   
   gettimeofday(&present, NULL);
   double present_time = present.tv_sec + present.tv_usec/1000000.0;
   cout<<"time"<<present_time-start_time<<endl;
    return 0;
}

