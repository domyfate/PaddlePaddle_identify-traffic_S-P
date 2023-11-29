##nihao

import numpy as np 
import pandas as pd
from keras.datasets import mnist 
from keras.models import load_model

(X_train, y_train_lable),(X_test, y_test_lable) = mnist.load_data()

temp_array = np.zeros((32,32))

#print(temp_array.shape)
num1_data = X_test[0]
print(num1_data.shape)

for i in range(28):
    temp_array[i+2][2:30] = num1_data[i][0:28]

print(temp_array.shape)
#提取5X5神经核
t_mat = np.zeros((5,5))
conv_list = []
for i in range(0,27,2):
    for j in range(0,27,2):
        t_mat = temp_array[i:i+5,j:j+5]
        conv_list.append(t_mat)
        t_mat =[]
       
#print(len(conv_list))
#print(conv_list[56])
# 提取5x5卷积核的权重
model = load_model("model_4_layers.h5")
w1,b1= model.layers[0].get_weights()

#print(w1)
# 转化为矩阵
w_mat =np.zeros((5,5))
w_mat_list = []
for i in  range(5):
    w_mat = w1[i].reshape(5,5)
    w_mat_list.append(w_mat)
#(w_mat_list)
#有权重
#print(b1)


# 填充打包为函数
def data_padding(old_data,new_data,old_data_height,new_data_height,strides):
    for i in range(old_data_height):
        new_data[i+strides][strides:strides+old_data_height] = old_data[i]
    return new_data
        
#print(data_padding(num1_data,temp_array,28,32,2))
####用户
# 数据集   num1_data,
# 权重     w_mat_list-->5个（5X5）

#数据划分
x_1 = np.random.randint(1,3,(28,28))
x_2 = num1_data - x_1
print(x_1)
print(x_2)

w_1_list = []
w_2_list = []
w_1 = np.random.randint(1,3,(5,5))
for i in range(5):
    w_1_list.append(w_1)
    w_2_list.append(w_mat_list[i] - w_1)

print(w_1_list[0])
print(w_2_list[0])

##提取卷积数


x_1_new = np.zeros((32,32))
x_1_padding = data_padding(x_1,x_1_new,28,32,2)

x_2_new = np.zeros((32,32))
x_2_padding = data_padding(x_2,x_2_new,28,32,2)

t_mat = np.zeros((5,5))
t1_mat = np.zeros((5,5))
x_1_list = []
x_2_list = []
for i in range(0,27,2):
    for j in range(0,27,2):
        t_mat = x_1_padding[i:i+5,j:j+5]
        x_1_list.append(t_mat)
        t_mat =[]

        t1_mat = x_2_padding[i:i+5,j:j+5]
        x_2_list.append(t1_mat)
        t1_mat =[]

a = np.random.randint(1,10,(5,5))
b = np.random.randint(1,10,(5,5))
c = a * b

a_1 = np.random.randint(1,8,(5,5))
b_1 = np.random.randint(1,8,(5,5))
c_1 = np.random.randint(1,4,(5,5))


a_2 = a - a_1
b_2 = b - b_1
c_2 = c - c_1
#print(c)
#print(c_1)
#print(c_2)


#服务器1    数据 x_1_list-->(196,5X5)，权重w_1_list--->(5,5X5)

e_1_list = []
f_1_list = []
z_1_list = []
for i in range(len(x_1_list)):
    e_1_list.append(x_1_list[i] - a_1)

for i in range(len(w_1_list)):
    f_1_list.append(w_1_list[i] - b_1)



# 服务器2   数据 x_2_list-->(196,5X5)，权重w_2_list_-->(5,5X5)

e_2_list = []
f_2_list = []
z_2_list = []
for i in range(len(x_2_list)):
    e_2_list.append(x_2_list[i] - a_2)

for i in range(len(w_2_list)):
    f_2_list.append(w_2_list[i] - b_2)

# 交换e_1，f_1和e_2，f_2
e_list = []
f_list = []
for i in range(len(e_1_list)):
    e_list.append(e_1_list[i] + e_2_list[i])

for i in range(len(f_2_list)):
    f_list.append(f_1_list[i] + f_2_list[i])

#计算z1，z2
#服务器A

z1_list1 = []
z1_list2 = []
z1_list3 = []
z1_list4 = []
z1_list5 = []

for i in range(len(e_list)):
    z1_list1.append(e_list[i] * b_1 + f_list[0] * a_1 + c_1 + b1[0] / 2)
    z1_list2.append(e_list[i] * b_1 + f_list[1] * a_1 + c_1 + b1[1] / 2)
    z1_list3.append(e_list[i] * b_1 + f_list[2] * a_1 + c_1 + b1[2] / 2)
    z1_list4.append(e_list[i] * b_1 + f_list[3] * a_1 + c_1 + b1[3] / 2)
    z1_list5.append(e_list[i] * b_1 + f_list[4] * a_1 + c_1 + b1[4] / 2)

print(len(z1_list1))
    

# 服务器B
z2_list1 = []
z2_list2 = []
z2_list3 = []
z2_list4 = []
z2_list5 = []
for i in range(len(e_list)):
    z2_list1.append(e_list[i] * b_2 + f_list[0] * a_2 + c_2 + e_list[i] * f_list[0] + b1[0] / 2)
    z2_list2.append(e_list[i] * b_2 + f_list[1] * a_2 + c_2 + e_list[i] * f_list[0] + b1[1] / 2)
    z2_list3.append(e_list[i] * b_2 + f_list[2] * a_2 + c_2 + e_list[i] * f_list[0] + b1[2] / 2)
    z2_list4.append(e_list[i] * b_2 + f_list[3] * a_2 + c_2 + e_list[i] * f_list[0] + b1[3] / 2)
    z2_list5.append(e_list[i] * b_2 + f_list[4] * a_2 + c_2 + e_list[i] * f_list[0] + b1[4] / 2)


##求和
z_list1 = []
z_list2 = []
z_list3 = []
z_list4 = []
z_list5 = []

for i in range(len(z1_list1)):
    z_list1.append(z1_list1[i] + z2_list1[i])
    z_list2.append(z1_list2[i] + z2_list2[i])
    z_list3.append(z1_list3[i] + z2_list3[i])
    z_list4.append(z1_list4[i] + z2_list4[i])
    z_list5.append(z1_list5[i] + z2_list5[i])
#print(z_list[-8])
#print(w_mat_list[0] * (x_1_list[-8] + x_2_list[-8]) )

#定一个求和的函数
def get_sum_conv(data):
    return  sum(map(sum,data)) 

#data1 = np.random.randint(1,5,(5,5))
#print("data1:",data1)
#print(get_sum_conv(data1))


#将5X5的卷积核求和
z1_sum_list1 = []
z1_sum_list2 = []
z1_sum_list3 = []
z1_sum_list4 = []
z1_sum_list5 = []

z2_sum_list1 = []
z2_sum_list2 = []
z2_sum_list3 = []
z2_sum_list4 = []
z2_sum_list5 = []
for i in range(len(z1_list1)):

    z1_sum_list1.append(get_sum_conv(z1_list1[i]))
    z1_sum_list2.append(get_sum_conv(z1_list2[i]))
    z1_sum_list3.append(get_sum_conv(z1_list3[i]))
    z1_sum_list4.append(get_sum_conv(z1_list4[i]))
    z1_sum_list5.append(get_sum_conv(z1_list5[i]))

    z2_sum_list1.append(get_sum_conv(z2_list1[i]))
    z2_sum_list2.append(get_sum_conv(z2_list2[i]))
    z2_sum_list3.append(get_sum_conv(z2_list3[i]))
    z2_sum_list4.append(get_sum_conv(z2_list4[i]))
    z2_sum_list5.append(get_sum_conv(z2_list5[i]))
#print(z1_list1[0])
#print(z1_sum_list1[0])
print(len(z1_sum_list1))

## 激活函数处理
#基于同态加密实现
import tenseal as ts

context = ts.context(ts.SCHEME_TYPE.CKKS,
                     poly_modulus_degree = 8192,
                     coeff_mod_bit_sizes = [60,40,40,60])
context.generate_galois_keys()
context.global_scale = 2 ** 40

'''
## CSP_1
en_z1_sum_list1 = ts.ckks_vector(context,np.array(z1_sum_list1))
#CSP_2
r_a =  np.random.randint(100,1000,len(z2_sum_list1))
en_z_sum_list1 = en_z1_sum_list1 + np.array(z2_sum_list1)
en_z_list1 = en_z_sum_list1 * r_a
## CSP_1
z_list1 = en_z_list1.decrypt()
print(z_list1)

state1_list1 = []
for i in z_list1:
    flag = 0
    if i > 0:
        flag = 1
    else:
        flag = 0
    state1_list1.append(flag)

#print(len(state1_list1))
#print(state1_list1)

z1_list1_act = np.array(z1_sum_list1) * np.array(state1_list1)
z2_list1_act = np.array(z2_sum_list1) * np.array(state1_list1)
print((z1_list1_act + z2_list1_act).shape)
'''

def hom_act(data_list1,data_list2):
    en_data_list1 = ts.ckks_vector(context,np.array(data_list1))
    r_a =  np.random.randint(100,1000,len(data_list1))
    en_sum_list1 = en_data_list1 + np.array(data_list2)
    en_z_list1 = en_sum_list1 * r_a
    z_list1 = en_z_list1.decrypt()
    state1_list1 = []
    for i in z_list1:
        flag = 0
        if i > 0:
            flag = 1
        else:
            flag = 0
        state1_list1.append(flag)
    
    data_list1_act = np.array(data_list1) * np.array(state1_list1)
    data_list2_act = np.array(data_list2) * np.array(state1_list1)
    return data_list1_act,data_list2_act

z1_sum_list1_act,z2_sum_list1_act = hom_act(z1_sum_list1,z2_sum_list1)
z1_sum_list2_act,z2_sum_list2_act = hom_act(z1_sum_list2,z2_sum_list2)
z1_sum_list3_act,z2_sum_list3_act = hom_act(z1_sum_list3,z2_sum_list3)
z1_sum_list4_act,z2_sum_list4_act = hom_act(z1_sum_list4,z2_sum_list4)
z1_sum_list5_act,z2_sum_list5_act = hom_act(z1_sum_list5,z2_sum_list5)

#print(p_1[0:6]+p_2[0:6])
#print(z1_list1_act[0:6]+z2_list1_act[0:6])
#print(z1_sum_list1_act + z2_sum_list1_act)

con1 = model.layers[2].get_weights()
#print("the len:",len(con1[0][0]))
#print("the len:",len(con1[1]))
#print(con1[1])

# 把5组数据串起来

all1_data = list(z1_sum_list1_act) + list(z1_sum_list2_act) + list(z1_sum_list3_act) + list(z1_sum_list4_act) + list(z1_sum_list5_act)
all2_data = list(z2_sum_list1_act) + list(z2_sum_list2_act) + list(z2_sum_list3_act) + list(z2_sum_list4_act) + list(z2_sum_list5_act)


#print(len(all_data))
'''
conv_list = []
for i in range(len(all_data)):
    temp = []
    for j in range(len(con1[0][0])):
        temp.append(all_data[i] * con1[0][0][j] )
    conv_list.append(temp)
        
print(len(conv_list))

conv_mat = np.zeros((1,100))
for i in conv_list:
    conv_mat += np.array(i)

conv_mat_list = list(conv_mat)
print(len(conv_mat_list[0]))
'''
#划分第一层全连接层
'''
conv1_list = []
conv2_list = []

for i in con1[0]:
    temp_conv_list = np.random.randint(1,10,(1,100))
    conv1_list.append(np.array(i - np.array(temp_conv_list)))
    conv2_list.append(np.array(temp_conv_list))
'''
#print(conv1_list[0][0][0:4])
#print(conv2_list[0][0][0:4])
#print(con1[0][0][0:4])

conv1_list1 = np.random.randint(1,10,(980,100))
conv1_list2 = np.array(con1[0]) - conv1_list1


#__________________________

conv1_a   = np.random.randint(1,5,(980,100))
conv1_a_1 = np.random.randint(1,8,(980,100))
conv1_a_2 = conv1_a - conv1_a_1


conv1_b   = np.random.randint(1,5,(980,100))
conv1_b_1 = np.random.randint(1,8,(980,100))
conv1_b_2 = conv1_b - conv1_b_1 

conv1_c = conv1_a * conv1_b
conv1_c_1 = np.random.randint(1,8,(980,100))
conv1_c_2 = conv1_c - conv1_c_1

#print(conv1_c[0][0:4])
#print(conv1_c_1[0][0:4])
#print(conv1_c_2[0][0:4])



#__________________________


# 服务器A的卷积权重  conv1_list --->(980,100)
# 数据 all1_data
#print(np.array(all1_data).shape)

#  服务器A
conv1_e_1 = np.array(all1_data).reshape(980,1) - conv1_a_1
conv1_f_1 = np.array(conv1_list1) - conv1_b_1

print(conv1_e_1.shape)
print(conv1_f_1.shape)

### 服务器B
conv1_e_2 = np.array(all2_data).reshape(980,1) - conv1_a_2
conv1_f_2 = np.array(conv1_list2) - conv1_b_2

print(conv1_e_2.shape)
print(conv1_f_2.shape)

### 汇总e，f

conv1_e = conv1_e_1 + conv1_e_2
conv1_f = conv1_f_1 + conv1_f_2

# 服务器A

conv1_z_1 = conv1_e * conv1_b_1 + conv1_f * conv1_a_1 + conv1_c_1

conv1_sum_1 = np.zeros((1,100))

for  i in conv1_z_1:
    conv1_sum_1 += i
print(conv1_sum_1.shape)

# 服务器B

conv1_z_2 = conv1_e * conv1_b_2 + conv1_f * conv1_a_2 + conv1_c_2 + conv1_e * conv1_f
conv1_sum_2 = np.zeros((1,100))

for i in conv1_z_2:
    conv1_sum_2 += i
#print(conv1_sum_2)

##直接调用sum（data，axis=0）进列求和
# = np.sum(conv1_z_1,axis=0)
#print(sum_con1[0:10])

print(conv1_sum_2.shape)

conv1_b = np.array(con1[1])/2
#服务器A
conv1_sumb_1 = conv1_sum_1 + conv1_b

#print(conv1_sum_1[0][0:4])
#print(conv1_b[0:4])
#print(conv1_sumb_1[0][0:4])

#服务器B
conv1_sumb_2 = conv1_sum_2 + conv1_b

################################
#######第二个全连接层 开始
####################################

conv2 = model.layers[3].get_weights()
conv2_w = conv2[0]
conv2_b = conv2[1]

# 划分权重
conv2_w_1 = np.random.randint(-3,3,(100,10))
conv2_w_2 = np.array(conv2_w) - conv2_w_1
#print(conv2_w[0][0:4])
#print(conv2_w_1[0][0:4])
#print(conv2_w_2[0][0:4])

# 产生a,b,c
conv2_a   = np.random.randint(1,8,(100,10))
conv2_a_1 = np.random.randint(1,5,(100,10))
conv2_a_2 = conv2_a - conv2_a_1

conv2_b   = np.random.randint(1,6,(100,10))
conv2_b_1 = np.random.randint(1,4,(100,10))
conv2_b_2 = conv2_b - conv2_b_1

conv2_c = np.array(conv2_a) *np.array(conv2_b)
conv2_c_1 = np.random.randint(1,6,(100,10))
conv2_c_2 = conv2_c - conv2_c_1


## 服务器A
conv2_e_1 = np.array(conv1_sumb_1.reshape(100,1))  - conv2_a_1
conv2_f_1 = conv2_w_1 - conv2_b_1


## 服务器B
conv2_e_2 = np.array(conv1_sumb_2.reshape(100,1))  - conv2_a_2
conv2_f_2 = conv2_w_2 - conv2_b_2



# 交换，计算e,f
conv2_e = conv2_e_1 + conv2_e_2
conv2_f = conv2_f_1 + conv2_f_2

#服务器A
conv2_z_1 = conv2_e * conv2_b_1 + conv2_f * conv2_a_1 + conv2_c_1
print(conv2_z_1.shape)
#服务器B
conv2_z_2 = conv2_e * conv2_b_2 + conv2_f * conv2_a_2 + conv2_c_2 + conv2_e * conv2_f

conv2_z = np.array(conv2_z_1) + np.array(conv2_z_2)

#print("the conv2_z:",conv2_z[0])
#print( (np.array((conv1_sumb_1 + conv1_sumb_2).reshape(100,1)) * np.array(conv2_w))[0][0:4] )

conv2_sum_1 = np.zeros((1,10))

for  i in conv2_z_1:
    conv2_sum_1 += i

conv2_sum_2 = np.zeros((1,10))

for  i in conv2_z_2:
    conv2_sum_2 += i


conv2_b_part = np.array(conv2[1])/2

conv2_sumb_1 = np.array(conv2_sum_1) + np.array(conv2_b_part)
conv2_sumb_2 = conv2_sum_2 + conv2_b_part

print(np.amax(np.array(conv2_sumb_1) + np.array(conv2_sumb_2)))
#print(np.array(conv2_sumb_1) + np.array(conv2_sumb_2))
print(np.argmax(np.array(conv2_sumb_1) + np.array(conv2_sumb_2)))
print(y_test_lable[0:10])
################################
#######第二个全连接层 结束
####################################


###demo 汇总

#conv1_z = conv1_z_1 + conv1_z_2 
#print(conv1_z.shape)
#print(conv1_z[-8][0:10])
#print((np.array(all1_data[-8]+all2_data[-8]) * np.array(con1[0][-8]))[0:10]) #可以验证呢

# demo-------------
'''
num1 = [1]
num2 = np.random.randint(1,10,(1,3))
num3 = np.array(1) - num2
print(num2)
print(num3)
'''
#________________
# 服务器B的卷积权重  conv2_list --->(980,100)
# 数据 all2_data











'''
a1 = [1,2,3]
a2 = [2,3,4]
r_a = np.random.randint(1,10,3)
print(r_a)
en_a1 = ts.ckks_vector(context,np.array(a1))
en_a = en_a1 + np.array(a2)
en_az = en_a * r_a
print(en_az.decrypt())
'''









#权重划分


####服务器A
####服务器B














'''
temp_array1 = np.zeros((7,7))
num1_data1 = np.random.randint(1,10,(3,3))
print(num1_data1)
for i in range(num1_data1.shape[1]):
    temp_array1[i+2][2:5] = num1_data1[i][0:3]
print(temp_array1)
print(data_padding(num1_data1,temp_array1,3,7,2))
'''
'''
num1_data1 = np.random.randint(1,10,(7,7))
print(num1_data1)
print("1")
print(num1_data1[2:5,2:5])
'''
