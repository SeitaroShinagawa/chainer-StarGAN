# chainer-StarGAN
chainer implementation of StarGAN  
### Caution: This code currently does not work well. Something wrong. 

### Requirement  
- chainer v4 (you can use v2 by following statements)  
- cv2
- pillow

To use v2, modify updater.py as below,
```pytohn
L.119        #dydx = self.dis.differentiable_backward(xp.ones_like(y_mid.data))   #uncomment
L.120        dydx, = chainer.grad([y_mid],[x_mid_v], enable_double_backprop=True) #comment out
```

## Progress 
Under testing on CelebA dataset
- [x] CelebA 
- [ ] RaFD 

![result](https://github.com/SeitaroShinagawa/chainer-StarGAN/blob/master/img/stargan.jpg)
Probably, this code still has bugs.  
This setting should be same to the [original code](https://github.com/yunjey/StarGAN). I confirmed that a diffrent setting leads better result, e.g. use n_dis=1 lambda_adv=0.25

## Preparation
Download [CelebA dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).  
```
source/
├── annotation
│   ├── list_attr_celeba.txt    #attribute lsit
│   └── list_eval_partition.txt #train/val/test information
├── img_align_celeba            #image directory
└── make_dict.py
```

run make_dict.py as follows,
```
cd source
python make_dict.py
``` 
Then, you will get 3 json files, train.json, val.json and test.json.   

## How to run
Select attributes you use. The attributes you put 1 on tail are valid.  
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
```
python train.py -g 0 --out result --eval_folder
```

## Known differences from [authors' original code](https://github.com/yunjey/StarGAN)
- The learning speed is much slower than original -- reconstruction loss of 30000 iter is the same to that of 3000 iter of original.

## (Option)
To visualize the generation result of all cases of attributes, you can use make_html.py
```
python make_html.py <out folder> <eval folder> <iter>
```

## Acknowledgement
This repository utilizes the codes of following impressive repositories  
- [chainer-gan-lib](https://github.com/pfnet-research/chainer-gan-lib)  
- [chainer-cyclegan](https://github.com/Aixile/chainer-cyclegan)  
- [chainer-cyclegan](https://github.com/naoto0804/chainer-cyclegan)  

