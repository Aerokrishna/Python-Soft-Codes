import math as m

class kp():
    def __init__(self):
        self.vertices = [(0,0),(0,1),(1,1),(2,1),(2,2),(1,2),(0,2),(2,0),(1,4),(2.5,4),(1.5,3),(2,5),(3,3)] #x and y

        self.edges = [(0,1),(1,0),(1,2),(1,6),(2,1),(2,3),(2,5),(3,2),
                      (3,4),(4,3),(4,5),(5,2),(5,6),(5,4),(6,5),(6,1),(3,7),(7,3),
                      (6,10),(10,6),(4,10),(10,4),(10,8),(8,10),(10,9),(9,10),
                      (8,9),(9,8),(8,11),(11,8),(10,11),(11,10),(9,12),(12,9),
                      (10,12),(12,10),(4,12),(12,4)] #start end
        
        self.open_nodes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        self.closed_nodes = []
        
        self.f_cost = []
        self.curr = self.open_nodes[7]

    def get_fcost(self,home,goal): #home (x,y) and goal (x,y)
        for i in range(len(self.vertices)):
            x = self.vertices[i][0]
            y = self.vertices[i][1]

            g = m.sqrt((x - home[0])**2 + (y - home[1])**2)
            h = m.sqrt((x - goal[0])**2 + (y - goal[1])**2)
            f = 0.2 * g + 0.9 * h

            self.f_cost.append(f)

    def get_next_node(self):
        neighbour_fcost = []
        neighbour_nodes= []
        open_edges = []

        #put current node in closed nodes list
        for i in range(len(self.open_nodes)):
            if self.curr == self.open_nodes[i]:
                self.closed_nodes.append(self.curr)
                self.open_nodes.remove(self.curr)
                break
        
        #remove the path already traversed

        for j in range(len(self.edges)):
            for k in range(len(self.open_nodes)):
                if self.edges[j][1] == self.open_nodes[k] and self.edges[j][0] == self.curr:
                    neighbour_nodes.append(self.edges[j][1])
                    #getting current node's neighbour's fcost and hcost
                    neighbour_fcost.append(self.f_cost[self.edges[j][1]])

        #getting minimum of the fcost and hcost
        min_fcost = neighbour_fcost.index(min(neighbour_fcost))
        
        #get the next node
        next_node = neighbour_nodes[min_fcost]
        self.curr = next_node

        return next_node
    
    def get_shortest_path(self,home_node,goal_node):
        shortest_path = [home_node]
        while self.curr!=goal_node:
            shortest_path.append(self.get_next_node())
            
        return shortest_path

path = kp()
path.get_fcost((path.vertices[7]),path.vertices[11]) #start and end points are given to calculate fcost
print(path.get_shortest_path(7,11)) 
print(path.open_nodes)

