import numpy as np
scores = []
from numpy.random import choice
totals = []
for i in range(10000):
    if i%1000 == 0:
        print(i)
    leader = choice([0, 1], 1, p=[0.6,0.4]) * 0.55
    vice = np.mean(choice([0,1], 3, p=[0.6,0.4],replace=True))*0.20
    people = np.mean(choice([0,1], 10000, p=[0.6,0.4], replace=True))*0.25
    total = leader + vice + people
    totals.append(total)
# one_split = int(55*3*0.4)
# zero_split = 55*3 - one_split
# distrib = [1]*one_split + [0]*zero_split
# leader = choice([0, 1], 1, p=[0.6, 0.4]) * 0.55
# vice = np.mean(choice(distrib, 3, replace=False)) * 0.2

print(np.mean(totals))
a = 0.549
print(a)
greater = 0
for num in totals:
    if num > a:
        greater += 1
print(greater/len(totals)*1.0)