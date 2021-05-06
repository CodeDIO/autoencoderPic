import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
# 自编码器部分
'''
定义输入层 （128，128） =16384
第一层隐含层500个
第二层100个
第三层500
输出层784 
这是因为自编码就是希望神经网络自己学习图片特征，然后再用学习到的特征去组成原始图片，所以最后
输出层是（128，128） =16384
'''
path="./outmodel/"#文件保存路径，如果不存在就会被重建
if not os.path.exists(path):#如果路径不存在
    os.makedirs(path)
input_n=16384       # 输入的数据是16384个，即（128*128）维度的
hidden1_n=500
hidden2_n=400
leastd = int(np.sqrt(hidden2_n))
hidden3_n=500
output_n=16384      # 输入的数据也是16384个，即（128*128）维度的
 
learn_rate=0.01
batch_size=100
train_epoch=2000
 
x=tf.placeholder(tf.float32,[None,input_n])
y=tf.placeholder(tf.float32,[None,input_n])
a2 = tf.placeholder(tf.float32,[None,hidden2_n]) 
weights1=tf.Variable(tf.truncated_normal([input_n,hidden1_n],stddev=0.1))
bias1=tf.Variable(tf.constant(0.1,shape=[hidden1_n]))
 
weights2=tf.Variable(tf.truncated_normal([hidden1_n,hidden2_n],stddev=0.1))
bias2=tf.Variable(tf.constant(0.1,shape=[hidden2_n]))
 
weights3=tf.Variable(tf.truncated_normal([hidden2_n,hidden3_n],stddev=0.1))
bias3=tf.Variable(tf.constant(0.1,shape=[hidden3_n]))
 
weights4=tf.Variable(tf.truncated_normal([hidden3_n,output_n],stddev=0.1))
bias4=tf.Variable(tf.constant(0.1,shape=[output_n]))
 
def get_result(x,weights1,bias1,weights2,bias2,weights3,bias3,weights4,bias4):
    a1=tf.nn.sigmoid(tf.matmul(x,weights1)+bias1)
    "a1.size:(?, 500)"
    a2=tf.nn.sigmoid(tf.matmul(a1,weights2)+bias2)
    "a2.size:(?, 100)"
    a3=tf.nn.sigmoid(tf.matmul(a2,weights3)+bias3)
    "a3.size:(?, 500)"
    y_=tf.nn.sigmoid(tf.matmul(a3,weights4)+bias4)
    "y_.size:(?, 16384)"
    return y_,a2

def decoder(a2,weights3,bias3,weights4,bias4):
    a3=tf.nn.sigmoid(tf.matmul(a2,weights3)+bias3)
    "a3.size:(?, 500)"
    y_=tf.nn.sigmoid(tf.matmul(a3,weights4)+bias4)
    "y_.size:(?, 16384)"
    return y_

y_,hidOut=get_result(x,weights1,bias1,weights2,bias2,weights3,bias3,weights4,bias4)

img_decode = decoder(a2,weights3,bias3,weights4,bias4)
 
loss=tf.reduce_mean(tf.pow(y_-y,2))
 
train_op=tf.train.RMSPropOptimizer(learn_rate).minimize(loss)

from PIL import Image
num = 128
img = Image.open('pictures/花.png')
img = img.resize((num,num))
pixel = np.array(img)
pixel = pixel.reshape((num*num , 3))/255

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for i in range(train_epoch):
        xs = pixel.T
        if i%1000 == 0:
            print('epoch:',i)
            print('loss:',sess.run(loss,feed_dict={x:xs,y:xs}))
        sess.run(train_op,feed_dict={x:xs,y:xs})
    saver = tf.train.Saver()  # 准备好保存的模型    
    saver.save(sess, "outmodel/out_model")
    
    
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    saver = tf.train.Saver()
    saver.restore(sess, "outmodel/out_model")
    xt = pixel.T

    yt=xt 
    encode_decode,hidden=sess.run([y_,hidOut],feed_dict={x:xt,y:yt})
    saver.save(sess, "outmodel/out_model")
    print(encode_decode.shape)
    
    
#    f,a =plt.subplots(3,3,figsize=(10,8))
#    for i in range(3):
#        a[0][i].imshow(np.reshape(xt[i],(num,num)))
#        a[1][i].imshow(np.reshape(hidden[i],(leastd,leastd)))
#        a[2][i].imshow(np.reshape(encode_decode[i],(num,num)))
#    f.show()
# ============================================================================
# Kmeans部分
def findCloset(X, centroids):
    m = X.shape[0]           # 获取数据条数
    k = centroids.shape[0]   # 获取类别数
    idx = np.zeros(m)        # 数据X每个样本对应的类别
    
    # 遍历所有数据，找到距离聚类中心最近的
    for i in range(m):
        min_dist = 1000000
        # 确定当经数据，离哪个中心点更近
        for j in range(k):
            # 计算当前点与k个类别中心的举例
            dist = np.sum((X[i,:] - centroids[j,:]) ** 2)
            if dist < min_dist:
                min_dist = dist
                idx[i] = j
    return idx


def setNewCenter(X, idx, k):
    # 得到矩阵大小，初始化矩阵
    m, n = X.shape
    centroids = np.zeros((k, n))  # 即数据又k簇，每簇的中心点由数据的特征数决定
    # 计算聚类中心
    for i in range(k):
        # 获取数据中类别为i的数据索引
        indices =  np.where(idx == i)
        # 计算新的中心点
        if (len(indices[0])==0):
            break
        centroids[i,:] = (np.sum(X[indices,:], axis=1) / len(indices[0])).ravel()
    return centroids

def runKmeans(X, initial_centroids, max_iters):
    m, n = X.shape
    k = initial_centroids.shape[0]
    idx = np.zeros(m)
    centroids = initial_centroids
    # STEP2：实施聚类算法，调用之前的两个函数
    # your code here  (appro ~ 2 lines)    
    for i in range(max_iters):
        idx = findCloset(X, centroids)
        centroids = setNewCenter(X, idx, k)
    
    return idx, centroids

def setCenters(X, k):
    m, n = X.shape  # 获取数据数量及特征数
    # 根据数据信息以及簇个数构建各簇中心点
    centroids = np.zeros((k, n))
    # 在数据中随机找三个点作为簇中心点
    idx = np.random.randint(0, m, k)
    
    for i in range(k):
        centroids[i,:] = X[idx[i],:]
    
    return centroids

A = pixel / 255.
A = hidden.T.reshape(leastd,leastd,3)
# 重置矩阵大小，将行数和列数合并，通道为单独的一维
X = np.reshape(A, (A.shape[0] * A.shape[1], A.shape[2]))

# 随机初始化聚类中心
initial_centroids = setCenters(X, 30)  # 随机抽取16个样本点作为聚类中心
# 运行kmeans聚类算法，迭代10次
idx, centroids = runKmeans(X, initial_centroids, 10)

# 获取样本点属于类别
idx = findCloset(X, centroids)

# 把每一个像素值与聚类结果进行匹配
X_recovered = centroids[idx.astype(int),:]
X_recovered.shape
"""
(16384, 3)
"""
# 将数据转成之前的数据格式
X_recovered = np.reshape(X_recovered, (A.shape[0], A.shape[1], A.shape[2]))
X = X.reshape(leastd,leastd, 3)
#plt.subplot(121)  
#plt.imshow(X)  
#plt.subplot(122)  
#plt.imshow(X_recovered) 
"""
(128, 128, 3)
"""

#with tf.Session() as sess:
#    tf.global_variables_initializer().run()
#    saver = tf.train.Saver()
#    saver.restore(sess, "outmodel/out_model")
#    X = X.reshape(hidden2_n,3).T 
#    decode_img=sess.run([img_decode],feed_dict={a2:X})
#    
#    print(decode_img[0].shape)
#    xt = pixel.reshape(16384,3).T
#data1 = decode_img[0].T.reshape(128,128,3)
#plt.imshow(data1)

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    saver = tf.train.Saver()
    saver.restore(sess, "outmodel/out_model")
    X_recovered = X_recovered.reshape(hidden2_n,3).T 
    decode_img=sess.run([img_decode],feed_dict={a2:X_recovered})
    
    print(decode_img[0].shape)
    xt = pixel.reshape(16384,3).T
data1 = decode_img[0].T.reshape(128,128,3)
data1 = np.array(data1)
#temp = data1[:,:,1]
#data1[:,:,0] = data1[:,:,1]
#data1[:,:,0] = temp')
plt.imshow(data1)