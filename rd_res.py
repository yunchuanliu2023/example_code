filename = "rate80.txt"
def gg():
    f = open(filename,encoding='utf-8')
    while True:
        line = f.readline()
        if "Accuracy" in line :
            print (line)
        else:
            break
    f.close()
    
def hh():
    with open(filename) as f:
        for line in f.readlines():
            if "Accuracy" in line  :
                print (line)
            # if "rate" in line  :
                # print (line)
            
hh()