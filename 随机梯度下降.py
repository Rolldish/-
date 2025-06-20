import numpy as np
import matplotlib.pyplot as plt

#读入训练数据
train = np.loadtxt("C:/Users/13712/Desktop/data.csv",delimiter=",",skiprows=1)
train_x =train[:,0]
train_y =train[:,1]

theta = np.random.rand(3)

#预测函数
def f(x):
    return np.dot(x,theta)

#标准化
mu =train_x.mean()
sigma  =train_x.std()
def standardize(x):
    return (x-mu)/sigma


train_z = standardize(train_x)

#创建训练数据的矩阵
def to_matrix(x):
    return np.vstack([np.ones(x.shape[0]),x,x**2]).T


X=to_matrix(train_z)

#均方误差
def MSE(x,y):
    return (1/x.shape[0])*np.sum((y-f(x))**2)

#均方误差的历史记录
errors = []

#误差的差值
diff = 1

#学习率
ETA= 1e-5

#重复学习
errors.append(MSE(X,train_y))
while diff >1e-2:
    #为了调整训练数据的顺序，准备随机的序列, 1.解释permutation函数的使用 2.解释shape函数的作用
    p = np.random.permutation(X.shape[0])
    #随机取出训练数据，使用随机梯度下降法更新参数3.解释zip函数的作用
    for x,y in zip(X[p,:],train_y[p]):
        theta -= ETA*(f(x)-y)*x
    #计算差值
    errors.append(MSE(X,train_y))
    diff = errors[-2] - errors[-1]

x= np.linspace(-3,3,100)
plt.plot(train_z,train_y,"o")
plt.plot(x,f(to_matrix(x)))
plt.show()

















































# #绘图
# # plt.plot(train_x,train_y,"o")
# # plt.show()
#
# #参数初始化
# # theta0 =np.random.rand()
# # theta1 =np.random.rand()
# # theta = np.random.rand(3)
#
# #预测函数
# # def f(x):
# #     return theta0+theta1*x
#
# #目标函数
# def E(x,y):
#     return 0.5*np.sum((y-f(x))**2)
#
# #标准化
# mu =train_x.mean()
# sigma  =train_x.std()
# def standardize(x):
#     return (x-mu)/sigma
#
#
# train_z = standardize(train_x)
#
# #创建训练数据的矩阵
# def to_matrix(x):
#     return np.vstack([np.ones(x.shape[0]),x,x**2]).T
#
# X=to_matrix(train_z)
#
# #预测函数
# def f(x):
#     return np.dot(x,theta)
#
# #目标函数
# def E(x,y):
#     return 0.5*np.sum((y-f(x))**2)
#
# #绘图
# # plt.plot(train_z,train_g,"o")
# # plt.show()
#
# #均方误差
# def MSE(x,y):
#     return (1/x.shape[0])*np.sum((y-f(x))**2)
#
# #用随机值初始化参数
# theta =np.random.rand(3)
#
# #误差的差值
# errors = []
#
# #误差的差值
# diff = 1
#
# #学习率
# ETA= 1e-5
#
# #重复学习
# errors.append(MSE(X,train_y))
#
# while abs(diff)>1e-3:
#     # tmp0 = theta0 - ETA * np.dot( f(X) - train_y,X )
#     theta = theta - ETA * np.dot( f(X) - train_y,X)
#
#     errors.append(MSE(X,train_y))
#     diff = errors[-2] - errors[-1]
#
#
# #绘制误差变化图
# x= np.arange(len(errors))
#
# plt.plot(x,errors)
# plt.show()
#
# #误差的差值
# diff =1