# chainer-StarGAN
chainer implementation of StarGAN

#Awknowlegement
This repository utilizes codes of following impressive repositories  
- [chainer-gan-lib](https://github.com/pfnet-research/chainer-gan-lib)  
- [chainer-cyclegan](https://github.com/Aixile/chainer-cyclegan)  

#Progress  
-[x] CelebA 
-[ ] RaFD 

#Preparation
Download [CelebA dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).  
```
source/
├── annotation
│   ├── list_attr_celeba.txt	#attribute lsit
│   └── list_eval_partition.txt #train/val/test information
├── img_align_celeba 			#image directory
└── make_dict.py
```

run make_dict.py as follows,
```python
cd source
python make_dict.py
``` 
Then, you will get 3 json files, train.json, val.json and test.json.   

#How to run
Select attributes you use. Attribute checked by putting 1 on tail is valid.  
```
0 5_o_Clock_Shadow
1 Arched_Eyebrows
2 Attractive
3 Bags_Under_Eyes
4 Bald
5 Bangs
6 Big_Lips
7 Big_Nose
8 Black_Hair 1 
9 Blond_Hair 1
10 Blurry
11 Brown_Hair 1
12 Bushy_Eyebrows
13 Chubby
14 Double_Chin
15 Eyeglasses
16 Goatee
17 Gray_Hair
18 Heavy_Makeup
19 High_Cheekbones
20 Male 1
21 Mouth_Slightly_Open
22 Mustache
23 Narrow_Eyes
24 No_Beard
25 Oval_Face
26 Pale_Skin
27 Pointy_Nose
28 Receding_Hairline
29 Rosy_Cheeks
30 Sideburns
31 Smiling
32 Straight_Hair
33 Wavy_Hair
34 Wearing_Earrings
35 Wearing_Hat
36 Wearing_Lipstick
37 Wearing_Necklace
38 Wearing_Necktie
39 Young 1
```
In the above case, 8:Black_Hair, 9:Blond_Hair, 11:Brown_Hair, 20:Male and 39:Young are used in training.  

Then, run train.py
```python
python train.py -g 0 --out result --eval_folder
```

#Known differences from [authors' original code](https://github.com/yunjey/StarGAN)
- This repository uses UpdateBuffer in update.py to make learning more stable
- Max iteration is 200000 by default, and linear learning rate annealing starts after 100000 iteration  

#(Option)
To visualize the generation result of all cases of attributes, you can use make_html.py
```python
python make_html.py <out folder> <eval folder> <iter>
```
