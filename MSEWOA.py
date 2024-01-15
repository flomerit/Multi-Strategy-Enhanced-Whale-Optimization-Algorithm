# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 00:24:55 2021

@author: ZongSing_NB2
"""

import numpy as np
import matplotlib.pyplot as plt

class MSEWOA():
    '''初始化,形参，默认值属性'''
    def __init__(self, fitness, D=30, P=20, G=500, ub=1, lb=0, 
                 a_max=2, a_min=0, a2_max=-1, a2_min=-2, l_max=1, l_min=-1, b_max=500, b_min=0): 
        self.fitness = fitness 
        self.D = D    #维数
        self.P = P    #种群数量
        self.G = G    #最大迭代次数
        self.ub = ub*np.ones([self.P, self.D])    #上界
        self.lb = lb*np.ones([self.P, self.D])    #下界（np数组进行维数匹配）
        self.a_max = a_max    #收敛因子，从2线性递减到0
        self.a_min = a_min
        self.a2_max = a2_max
        self.a2_min = a2_min
        self.l_max = l_max    #螺旋位置更新
        self.l_min = l_min
        self.b_max = b_max
        self.b_min = b_min

        '''不能通过位置、关键字进行参数传递，句号属性方法（初始化）'''
        self.gbest_X = np.zeros([self.D])    #初始化全局最优位置，全0
        self.gbest_F = np.inf    #初始化全局最优解的适应度值为inf无穷大
        self.loss_curve = np.zeros(self.G)    #维数最大迭代次数
        
        
    def opt(self):
        # 内部初始化，uniform用于生成指定范围内均匀分布的随机数。self.整个类的实例中使用，因为它们是对象的特征。其他方法内部定义的参数（无self)通常是用于执行具体操作的局部变量
        self.X = np.random.uniform(low=self.lb, high=self.ub, size=[self.P, self.D])
        
        # 迭代
        for g in range(self.G):
            # OBL
            self.X, F = self.OBL()    #OBJ方法的返回值
            # F = self.fitness(self.X)
            
            # 更新最佳解，F是整个种群的适应度
            if np.min(F) < self.gbest_F:
                idx = F.argmin()    #用于找到数组中最小值所在的位置，并返回该位置的索引。
                self.gbest_X = self.X[idx].copy()    #浅拷贝，这意味着在拷贝中做的修改不会影响原始数组，但如果修改原始数组的数据，拷贝也会受到影响。
                self.gbest_F = F.min()
            
            # 收斂曲線
            self.loss_curve[g] = self.gbest_F    #损失函数的横纵坐标，迭代次数，最佳适应度
            old_X = self.X.copy()
            old_F = F.copy()
            
            # 更新
            a2 = self.a2_max - (self.a2_max-self.a2_min)*(g/self.G)
            
            for i in range(self.P):
                p = np.random.uniform()
                r2 = np.random.uniform()
                r3 = np.random.uniform()
                C = 2*r2
                l = (a2-1)*r3 + 1
                b = np.random.randint(low=self.b_min, high=self.b_max)
                
                if p>0.5:
                    D = np.abs(self.gbest_X - self.X[i, :])
                    self.X[i, :] = D*np.exp(b*l)*np.cos(2*np.pi*l)+self.gbest_X
                else:
                    D = C*self.gbest_X - self.X[i, :]
                    self.X[i, :] = self.gbest_X - D*np.cos(2*np.pi*l)
            
            # 邊界處理
            self.Reflect2()
            self.Reflect()
            
    def OBL(self):
        # 產生反向解
        k = np.random.uniform()
        alpha = self.X.min(axis=0)
        beta = self.X.max(axis=0)
        obl_X = k*(alpha+beta) - self.X
        
        # 對反向解進行邊界處理
        rand_X = np.random.uniform(low=alpha, high=beta, size=[self.P, self.D])
        mask = np.logical_or(obl_X>self.ub, obl_X<self.lb)
        obl_X[mask] = rand_X[mask].copy()
        
        # 取得新解
        concat_X = np.vstack([obl_X, self.X])
        F = self.fitness(concat_X)
        top_idx = F.argsort()[:self.P]
        top_F = F[top_idx].copy()
        top_X = concat_X[top_idx].copy()
        
        return top_X, top_F
    
    def Reflect(self):
        mask1 = self.X>self.ub
        mask2 = self.X<self.lb
        
        X1 = self.ub - (self.X-self.ub)
        X2 = self.lb + (self.lb-self.X)
        
        self.X[mask1] = X1[mask1]
        self.X[mask2] = X2[mask2]
        
        rand_X = np.random.uniform(low=self.lb, high=self.ub, size=[self.P, self.D])
        mask = np.logical_or(self.X>self.ub, self.X<self.lb)
        self.X[mask] = rand_X[mask].copy()
    
    def Reflect2(self):
        r5 = np.random.uniform()
        r6 = np.random.uniform()
        
        mask1 = self.X>self.ub
        mask2 = self.X<self.lb
        max_map = self.ub + r5*self.ub*(self.ub-self.X)/self.X
        min_map = self.lb + r6*np.abs(self.lb*(self.lb-self.X)/self.X)
        
        rand_X = np.random.uniform(low=self.lb, high=self.ub, size=[self.P, self.D])
        mask3 = max_map==np.inf
        mask4 = min_map==np.inf
        max_map[mask3] = rand_X[mask3]
        min_map[mask4] = rand_X[mask4]
        
        self.X[mask1] = max_map[mask1]
        self.X[mask2] = min_map[mask2]
        
        # rand_X = np.random.uniform(low=self.lb, high=self.ub, size=[self.P, self.D])
        # mask = np.logical_or(self.X>self.ub, self.X<self.lb)
        # self.X[mask] = rand_X[mask].copy()

    def plot_curve(self):
        plt.figure()
        plt.title('loss curve ['+str(round(self.gBest_curve[-1], 3))+']')
        plt.plot(self.gBest_curve, label='loss')
        plt.grid()
        plt.legend()
        plt.show()
            
