#!/usr/bin/env python

import json
import time
annotation_dir = "annotation"

#load face attribute 
print("loading face attribute")
attr_dict = {}
with open(annotation_dir+"/list_attr_celeba.txt","r") as f:
    for i,line in enumerate(f):
        line = line.strip().replace("    "," ").replace("   "," ").replace("  "," ")
        if i==0:
            pass
        elif i==1:
            att_list = line.split(" ")
            print(att_list)
            time.sleep(10)
        else:
            attribute_list = line.split(" ") #imgname,att1,att2,...,att10
            imgname = attribute_list[0]
            attribute_list.pop(0)
            if len(attribute_list) == len(att_list):
                attr_dict[imgname] = {}
                for att,value in zip(att_list,attribute_list):
                    print(i,att,value)
                    attr_dict[imgname][att] = int(value)
            else:
                print("len(attribute_list) != len(att_list)")

#load data type (train:0,val:1,test:2)
print("loading train/val/test label")
type_list = [] 
with open(annotation_dir+"/list_eval_partition.txt","r") as f:
	for line in f:
		line = line.strip()
		type_list.append((line.split(" ")))

#make dict		
print("making dict")
train_dict = {}
val_dict = {}
test_dict = {}

dict_type = {"0":train_dict,"1":val_dict,"2":test_dict}
for imgname,str_num in type_list:
    data_dict = dict_type[str_num] 
    data_dict[imgname] = {"attribute":attr_dict[imgname]}


def save_dict(dict_path,dic):
    with open(dict_path,"w") as f:
        json.dump(dic,f,indent=2)

#save dict
print("save dicts")
save_dict("train.json",train_dict)
save_dict("val.json",val_dict)
save_dict("test.json",test_dict)

print("finish.")
