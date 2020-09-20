def not_in_frontier(node, frontier):
    for i in frontier:
        if node == i[1]:
            return False
    return True


def not_in_visited(node, visited):
    for i in visited:
        if node == i[0]:
            return False
    return True


def replace_cost_UCS(frontier, node, cost, P_node):
    for i in frontier:
        if i[1] == node:
            if i[0] > cost:
                i = (cost, node, P_node)
            break


def replace_cost_Astar(frontier, node, cost, P_node, heuristic):
    for i in frontier:
        if i[1] == node:
            if i[0] + i[3] > cost + heuristic:
                i = (cost, node, P_node, heuristic)
            break


def DFS_Traversal(cost, goals, visited, frontier):
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
        return DFS_Traversal(cost, goals, visited, frontier)
    else:
        return []


def UCS_Traversal(cost, start_point, goals):  # Checks all goal states
    visited = []
    path = []
    mincost = -1
    frontier = [(0, start_point, 0)]
    while len(frontier):
        frontier.sort(key=lambda x: x[0])
        ele = frontier.pop(0)
        path_cost = ele[0]
        node = ele[1]
        parent = ele[2]
        if(node in goals):
            if mincost == -1 or path_cost < mincost:
                path = []
                mincost = path_cost
                path.append(node)
                while(parent != 0):  # Parent of start_node is 0
                    for i in visited:
                        if parent == i[0]:
                            path.append(i[0])
                            parent = i[1]
                path.reverse()
        visited.append((node, parent))
        for i in range(1, len(cost)):
            if cost[node][i] > 0 and not_in_visited(i, visited):
                if not_in_frontier(i, frontier):
                    frontier.append((path_cost + cost[node][i], i, node))
                else:
                    replace_cost_UCS(frontier, i, path_cost +
                                     cost[node][i], node)
    return path


def A_star_Traversal(cost, start_point, goals, heuristic):
    frontier = []
    visited = []
    path = []
    mincost = -1
    # path_cost,node,parent,heuristic
    frontier.append((0, start_point, 0, heuristic[start_point]))
    while len(frontier):
        frontier = sorted(frontier, key=lambda x: x[1], reverse=True)
        frontier = sorted(frontier, key=lambda x: x[0]+x[3])
        ele = frontier.pop(0)
        path_cost = ele[0]
        node = ele[1]
        parent = ele[2]
        if(node in goals):
            if mincost == -1 or path_cost < mincost:
                path = []
                mincost = path_cost
                path.append(node)
                while(parent != 0):  # Parent of start_node is 0
                    for i in visited:
                        if parent == i[0]:
                            path.append(i[0])
                            parent = i[1]
                path.reverse()
        # node,parent
        visited.append((node, parent))
        for i in range(1, len(cost)):
            if cost[node][i] > 0 and not_in_visited(i, visited):
                if not_in_frontier(i, frontier):
                    frontier.append(
                        (path_cost + cost[node][i], i, node, heuristic[i]))
                else:
                    replace_cost_Astar(
                        frontier, i, path_cost + cost[node][i], node, heuristic[i])
    return path


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
    t1 = DFS_Traversal(cost, goals, [], [start_point])

    t2 = UCS_Traversal(cost, start_point, goals)

    t3 = A_star_Traversal(cost, start_point, goals, heuristic)

    l.append(t1)
    l.append(t2)
    l.append(t3)
    return l
