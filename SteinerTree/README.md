# Steiner Tree

Steiner Tree Problem (STP) Steiner tree problem (named after Jacob Steiner) one of the NP-Complete Problems. 
If we apply restriction on the problem than we can solved some of the problems in polynomial time. 
This problem is some what similar to the minimum spanning tree problem. In mst problem, we are required construct a 
tree that cover all the vertices of a graph with minimum cost. while in the Steiner tree problem, we are given some vertices, 
which we called as the terminals(required vertices) and some other points which are the optional points(Steiner Point), 
we are required to connect all the terminal vertices with minimum cost and in this connection, 
we may use some optional points(Steiner Point) to cover all the terminal vertices with minimum cost.


In this problem we come up with a better running time Heuristic Algorithm for construct a Steiner tree from a graph
~~~
INPUT :- 
~~~
A undirected connected graph G = (V, E) with edge distance or cost d between the vertices vi and vj , and a subset of nodes S is 
The subset of V , S is called required vertex.

~~~
OUTPUT :-
~~~
A Steiner Tree Ts that connect all the subset node with minimum cost



