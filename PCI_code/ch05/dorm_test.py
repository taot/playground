import dorm

# Page 108
dorm.printsolution([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

# Page 109
reload(dorm)
s = optimization.randomoptimize(dorm.domain, dorm.dormcost)
dorm.printsolution(s)
dorm.dormcost(s)

reload(dorm)
s = optimization.geneticoptimize(dorm.domain, dorm.dormcost)
dorm.printsolution(s)
dorm.dormcost(s)
