import os

if __name__=="__main__":
    
    Fin = open('trainlist.txt','r')
    lines = Fin.readlines()

    chunk = 12000 
    count = 0

    Fout = open('trainlist_'+str(count)+'.txt','w')

    for idx in range(len(lines)):

        if idx > 0 and idx%chunk == 0:
            Fout.close()
            count +=1
            Fout = open('trainlist_'+str(count)+'.txt','w')

        Fout.write(lines[idx])

    Fout.close()
