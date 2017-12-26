# /usr/bin/python
# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets
from tensorflow.python.framework import ops

ops.reset_default_graph()
sess = tf.Session()


# 对称圆数据集
# 半径为1(cos x, sin x)的圆随机扰动
# base采集点为（100，200）
x_vals = np.cos(np.random.normal(0, 1, 100))
y_vals = [np.random.choice([1, -1], size=1)[0] * np.sqrt(1 - x_val ** 2) for x_val in x_vals]
x_vals = np.array([x_val + 100 for x_val in x_vals])
y_vals = np.array([y_vals + 200 for y_vals in y_vals])


# 归一化
# def normalize_cols(m):
#     col_max = m.max(axis=0)
#     col_min = m.min(axis=0)
#     return (m-col_min) / (col_max - col_min)
#
#
# x_vals = np.nan_to_num(normalize_cols(x_vals))
# y_vals = np.nan_to_num(normalize_cols(y_vals))

plt.plot(x_vals, y_vals, 'o', label='Data Points')
plt.show()

# iris 数据集超参数
# learning_rate = 0.05
# batch_size = 25

# 对称圆数据集超参数
learning_rate = 0.001
batch_size = 25

x_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)
# Perfect initialize
# A = tf.Variable(tf.constant([[2.]]))
# b = tf.Variable(tf.constant([[0.]]))

# Simple initialize
A = tf.Variable(tf.random_normal(shape=[1, 1]))
b = tf.Variable(tf.random_normal(shape=[1, 1]))

model_output = tf.add(tf.matmul(x_data, A), b)

# L2 正则损失
loss = tf.reduce_mean(tf.square(y_target - model_output))
# L1 正则损失
# loss = tf.reduce_mean(tf.abs(y_target - model_output))

init = tf.global_variables_initializer()
sess.run(init)
my_opt = tf.train.GradientDescentOptimizer(learning_rate)
train_step = my_opt.minimize(loss)

loss_vec = []
for i in range(1000):
    rand_index = np.random.choice(len(x_vals), size=batch_size)
    rand_x = np.transpose([x_vals[rand_index]])
    rand_y = np.transpose([y_vals[rand_index]])
    sess.run(train_step, feed_dict={
        x_data: rand_x,
        y_target: rand_y
    })
    temp_loss = sess.run(loss, feed_dict={
        x_data: rand_x,
        y_target: rand_y
    })
    loss_vec.append(temp_loss)
    if (i + 1) % 10 == 0 or i < 10:
        [temp_slope] = sess.run(A)
        [temp_y_intercept] = sess.run(b)
        print('Loss:' + str(temp_loss) + 'slope' + str(temp_slope) + 'intercept' + str(temp_y_intercept))

[slope] = sess.run(A)
[y_intercept] = sess.run(b)
best_fit = []
x_s = np.linspace(0, 100, 100)
for i in x_s:
    print(i, slope * i + y_intercept)
    best_fit.append(slope * i + y_intercept)

plt.plot(x_vals, y_vals, 'o', label='Data Points')
plt.plot(x_s, best_fit, 'r-', label='Best Fit Line')
plt.legend(loc='upper left')
plt.title('Sepal Length vs Pedal width')
plt.xlabel('Pedal Width')
plt.ylabel('Sepal Length')
plt.show()
plt.plot(loss_vec, 'k-')
plt.title('L1 Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('L1 Loss')
plt.show()
