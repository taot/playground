import nn

mynet = nn.searchnet()

wWorld, wRiver, wBank = 101, 102, 103
uWorldBank, uRiver, uEarth = 201, 202, 203

mynet.generatehiddennode([wWorld, wBank], [uWorldBank, uRiver, uEarth])


reload(nn)
mynet = nn.searchnet()
mynet.getresult([wWorld, wBank], [uWorldBank, uRiver, uEarth])

reload(nn)
mynet = nn.searchnet()
mynet.trainquery([wWorld, wBank], [uWorldBank, uRiver, uEarth], uWorldBank)
mynet.getresult([wWorld, wBank], [uWorldBank, uRiver, uEarth])

allurls = [uWorldBank, uRiver, uEarth]
for i in range(30):
    mynet.trainquery([wWorld, wBank], allurls, uWorldBank)
    mynet.trainquery([wRiver, wBank], allurls, uRiver)
    mynet.trainquery([wWorld], allurls, uEarth)

mynet.getresult([wWorld, wBank], allurls)
mynet.getresult([wRiver, wBank], allurls)
mynet.getresult([wBank], allurls)
