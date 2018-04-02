import tensorflow as tf
import numpy as np
#
# def argmax2(x):
#     # todo assert(first dimension - batch, second argmax)
#     #pad = tf.expand_dims(tf.zeros_like(x[:,1]),1)
#     #x_pad1 = tf.concat([pad,x,pad],axis=1)
#     #argm1 = tf.argmax(x,axis=1)
#     #x_ = tf.scatter_update(argm1,updates=tf.zeros_like(x[:,0]),indices=argm1)
#     #x_pad2 = tf.concat([x_pad1[0:argm1],pad,x_pad1[argm1+1:]],axis=-1)
#     #argm2 = tf.argmax(x_pad2)
#     #return tf.shape(x),tf.shape(argm1),tf.shape(argm2),tf.shape(x_pad1),tf.shape(x_pad2)
#     #return argm1,argm2
#     return tf.nn.top_k(x,k=2).indices
# a = tf.constant([[1,4,6,7999,78,7,7],
#                  [1,4,6,7,78,700,7]])
#
# b = argmax2(a)
#
# sess = tf.Session()
#
# print sess.run(tf.argmax(a,axis=1))

class TrainingMonitor:
    def __init__(self):
        self._hist_records = {}
    def ave(self,name,value,num=20):
        if not name in self._hist_records:
            self._hist_records[name] = []
        self._hist_records[name].append(value)
        return np.average(self._hist_records[name][-num:])
    def prints(self):
        for key in self._hist_records:
            print key, self._hist_records[key][-1],"ave:", np.average(self._hist_records[key][-20:])

tm = TrainingMonitor()
print tm.ave("train",10)
print tm.ave("train",10)
tm.prints()