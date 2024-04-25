# Page 91
import optimization

s = [1, 4, 3, 2, 7, 3, 6, 3, 2, 4, 5, 3]

# Page 92
reload(optimization)
domain = optimization.getdomain()
s = optimization.randomoptimize(domain, optimization.schedulecost)
optimization.printschedule(s)
optimization.schedulecost(s)

# Page 94
reload(optimization)
domain = optimization.getdomain()
s = optimization.hillclimb(domain, optimization.schedulecost)
optimization.printschedule(s)
optimization.schedulecost(s)

# Page 96
reload(optimization)
domain = optimization.getdomain()
s = optimization.annealingoptimize(domain, optimization.schedulecost)
optimization.printschedule(s)
optimization.schedulecost(s)

# Page 98
reload(optimization)
domain = optimization.getdomain()
s = optimization.geneticoptimize(domain, optimization.schedulecost, popsize = 100, step = 1, mutprob = 0.8, elite = 0.3, maxiter = 100)
optimization.printschedule(s)
optimization.schedulecost(s)
