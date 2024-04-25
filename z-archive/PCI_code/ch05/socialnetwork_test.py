import socialnetwork
import optimization

# Page 113
reload(socialnetwork)
reload(optimization)
sol = optimization.randomoptimize(socialnetwork.domain, socialnetwork.crosscount)
socialnetwork.crosscount(sol)

sol = optimization.annealingoptimize(socialnetwork.domain, socialnetwork.crosscount)
socialnetwork.crosscount(sol)

sol = optimization.geneticoptimize(socialnetwork.domain, socialnetwork.crosscount)
socialnetwork.crosscount(sol)
