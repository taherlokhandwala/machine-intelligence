def A_star_Traversal(
    # add your parameters
):
    l = []

    return l


def UCS_Traversal(
    # add your parameters
):
    l = []

    return l


frontier = []


def DFS_Traversal(cost, goals, visited):
    global frontier
    if len(frontier):
        start_point = frontier.pop()
        visited.append(start_point)
        if start_point in goals:
            return visited
        l = []
        for i in range(1, len(cost)):
            if cost[start_point][i] > 0 and i not in visited:
                l.append((i, cost[start_point][i]))
        l.sort(key=lambda x: x[1])
        j = len(l)-1
        while j >= 0:
            frontier.append(l[j][0])
            j -= 1
        return DFS_Traversal(cost, goals, visited)
    else:
        return []


'''
Function tri_traversal - performs DFS, UCS and A* traversals and returns the path for each of these traversals 

n - Number of nodes in the graph
m - Number of goals ( Can be more than 1)
1<=m<=n
Cost - Cost matrix for the graph of size (n+1)x(n+1)
IMP : The 0th row and 0th column is not considered as the starting index is from 1 and not 0. 
Refer the sample test case to understand this better

Heuristic - Heuristic list for the graph of size 'n+1' 
IMP : Ignore 0th index as nodes start from index value of 1
Refer the sample test case to understand this better

start_point - single start node
goals - list of size 'm' containing 'm' goals to reach from start_point

Return : A list containing a list of all traversals [[],[],[]]
1<=m<=n
cost[n][n] , heuristic[n][n], start_point, goals[m]

NOTE : you are allowed to write other helper functions that you can call in the given fucntion
'''


def tri_traversal(cost, heuristic, start_point, goals):
    l = []
    frontier.append(start_point)
    t1 = DFS_Traversal(cost, goals, [])
    t2 = UCS_Traversal(
        # send whatever parameters you require
    )
    t3 = A_star_Traversal(
        # send whatever parameters you require
    )

    l.append(t1)
    l.append(t2)
    l.append(t3)
    return l
