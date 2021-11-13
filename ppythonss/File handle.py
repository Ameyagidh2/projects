import os
os.chdir(r"A:\Data science")
fhandle=open("ameya.txt","r")
fhandle2=open("amu.txt","w")
contents=fhandle.readlines()
for line in contents:
    #s=line.rstrip()
    #print(s)
    #print(line)
    fhandle2.write(line)


s="ameya is a man of power,power and vigor."
#s1=s.replace("power","hardwork",1)
#s1=s.find("power",19,len(s))
#s1=s.index("power")
#s2="He wants to study ms in computers but has his degree in mechanical engineering"
#s1='-'.join([s,s2])
#print(s1)

#print(s,end="")
#print("akiosjiodjiogn")
#print("Age:{0},date:{1},month:{2},year:{3}".format(21,2,9,1999))
