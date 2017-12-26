# /usr/bin/python
# coding: utf-8
# 基于矩阵分解的回归算法实现

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

sess = tf.Session()
x_vals = np.linspace(0, 10, 100000)
x_vals = x_vals.reshape([100, 1000])
# y_vals = np.array([np.average(arr) for arr in x_vals]) + np.random.normal(0, 1, 1000)

x_vals_column = np.transpose(np.matrix(x_vals))
ones_column = np.transpose(np.matrix(np.repeat(1, 1000)))
A = np.column_stack((ones_column, x_vals_column))

y_vals = np.array([np.average(arr) for arr in x_vals_column]) + np.random.normal(0, 1, 1000)
b = np.transpose(np.matrix(y_vals))

A_tensor = tf.constant(A)
b_tensor = tf.constant(b)

tA_A = tf.matmul(tf.transpose(A_tensor), A_tensor)
tA_A_inv = tf.matrix_inverse(tA_A)
product = tf.matmul(tA_A_inv, tf.transpose(A_tensor))
solution = tf.matmul(product, b_tensor)

solution_eval = sess.run(solution)

slope = solution_eval[1][0]
y_intercept = solution_eval[0][0]

best_fit = []
for i in x_vals:
    best_fit.append(slope * i + y_intercept)
plt.plot(x_vals, y_vals, 'o', label='Data')
plt.plot(x_vals, best_fit, 'r-', label='Best fit line', linewidth=3)
plt.legend(loc='upper left')
plt.show()

