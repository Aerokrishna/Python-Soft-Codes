from collections import defaultdict

roadmap = defaultdict(list)
qtable = defaultdict(dict)

roadmap[0] = [1, 2]
roadmap[1] = [0, 3]

for i in roadmap:
    for j in roadmap[i]:
        qtable[i][j] = 0

print("Roadmap:", roadmap)
print("Q-table:", qtable)