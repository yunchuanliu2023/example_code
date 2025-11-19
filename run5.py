import os
import timeit
# parameter 
import subprocess
# 1st: iteration times of Neat
# 2rd: iteration times of how many runs
# 3nd: lag numbers only 


def run():
    s1 = timeit.default_timer()  


    for i in range(1,6):
        j = i+offset
        out = subprocess.call('python ./cwk'+str(i)+'.py '+str(rate)+' '+str(j), shell=True)
    s2 = timeit.default_timer()  
    print ('Runing time is Mins:',round((s2 -s1)/60,2))


def rename():
    #p1 is the folder name tat contain all the file
    name = 'tcresult'
    path = "./"+name+"/"
    path2 = "./" + name
    fileList=os.listdir(path2)

    for f in fileList:
        if "_" in f:
            x = f.split("_", 1)
            x2 = x[1].split(".")
            print(x2)
            srcDir = str(path)+f
            # dstDir = str(path)+str(rate)+"-"+x2[0]+".csv"
            dstDir = str(path)+str(rate)+"-"+str(int(x2[0])+offset)+".csv"
            # print(f,dstDir)
            # print(srcDir,x[1])
            
            try:        
                os.rename(srcDir,dstDir)
            except Exception as e:
                print (e)
                print ('rename dir fail\r\n')
            else:
                print ('rename dir success\r\n')
                
rate = 10
offset = 5

# run()

rename()