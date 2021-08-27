#! /usr/bin/env python3

class LinerRegression(object):
  def __init__(self, lr, iteration):
    self.lr = lr
    self.iteration = iteration
    self.w = 0.0
    self.bias = 0.0

  def cost(self, X, Y):
    n = len(X)
    cost_total = 0.0
    for i in range(0, n):
      cost_total += (Y[i] - (self.w * X[i] + self.bias)) ** 2
    return cost_total / n

  def update_weight(self, X, Y):
    n = len(X)
    wt = 0.0
    bt = 0.0
    for i in range(0, n):
      wt += (-2 * X[i] * (Y[i] - self.w * X[i] - self.bias))
      bt += -2 * (Y[i] - self.w * X[i] - self.bias)
    # 方差一定是正的。为啥这里一定是减呢?
    self.w -= wt / n * self.lr
    self.bias -= bt / n * self.lr

  def fit(self, X, Y):
    for i in range(0, self.iteration):
      self.update_weight(X, Y)
      print ("iteration:", i, " cost:", self.cost(X, Y))

  def predict(self, X):
    # 预测也得归一化
    x = [(k + 100) / 200 for k in X]
    n = len(X)
    for i in range(0, n):
      print("input:", X[i], " ==> output:", (self.w * x[i] + self.bias))

x = [1,2,3,10,20,50,100,-2,-10,-100,-5,-20]
y = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]

model = LinerRegression(0.01, 500)

# 别忘了归一化输入
X = [(k + 100) / 200 for k in x]

model.fit(X, y)

test = [12,34,34, -99, -39, -58]
model.predict(test)