#!usr/bin/env python

import sys
import subprocess as c

result_path = sys.argv[1]
eval_path = sys.argv[2]
iteration = sys.argv[3]

with open(result_path+"/att_list.txt","r") as f:
    #att_list = []
    att_name = []
    for line in f:
        line = line.strip().split(" ")
        if len(line)==3:
            #att_list.append(int(line[0])) #attID
            att_name.append(line[1]) #attname
print("attribute list:",",".join(att_name))

#make all cases of 2bit attributes
n_att = len(att_name)
att_list = [[int(k) for k in j] for j in [bin(i)[2:].zfill(n_att) for i in range(2**n_att)]]

cmd = "ls "+eval_path+"/iter_"+iteration+"_*.jpg"
ret = c.check_output(cmd, shell=True)
img_path_list = ret.decode("utf-8").strip().split("\n")

with open("index.html","w") as f:
    original_img_path = img_path_list[0]
    for img_path,att in zip(img_path_list[1:],att_list):
        print_att_list = []
        for i,a in enumerate(att):
            if a==1:
                print_att_list.append(att_name[i])
        print('<h1>Original -> '+",".join(print_att_list)+'</h1>',file=f)
        print('<table style="text-align:center;">',file=f)
        print('<tr><td><img src="'+original_img_path+'"/><br>',file=f)
        print('<img src="'+img_path+'"/></td></tr>',file=f)
        print('</table>',file=f)

print("done.")
