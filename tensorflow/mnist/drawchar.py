def drawchar(arr):
    for i in range(0, 28):
        for j in range(0, 28):
            v = arr[28 * i + j]
            if v > 0.75:
                print '#',
            elif v > 0.5:
                print '*',
            elif v > 0.25:
                print '-',
            elif v > 0:
                print '.',
            else:
                print ' ',

        print '\n',
