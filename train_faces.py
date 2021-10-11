import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
import cv2
import numpy as np
import os
import random
import sys
from sklearn.model_selection import train_test_split

familiar_faces_path = './familiar_faces'
unfamiliar_faces_path = './unfamiliar_faces'
size = 64

imgs = []
labs = []

def getPaddingSize(img):
    h, w, _ = img.shape
    top, bottom, left, right = (0,0,0,0)
    longest = max(h, w)

    if w < longest:
        tmp = longest - w
        left = tmp // 2
        right = tmp - left
    elif h < longest:
        tmp = longest - h
        top = tmp // 2
        bottom = tmp - top
    else:
        pass
    return top, bottom, left, right

def readImage(path , h=size, w=size):
    for filename in os.listdir(path):
        if filename.endswith('.jpg'):
            filename = path + '/' + filename

            img = cv2.imread(filename)

            top,bottom,left,right = getPaddingSize(img)
            # enlarge and resize the image
            img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0,0,0])
            img = cv2.resize(img, (h, w))

            imgs.append(img)
            labs.append(path)

readImage(familiar_faces_path)
readImage(unfamiliar_faces_path)
# convert image data into array
imgs = np.array(imgs)
labs = np.array([[0,1] if lab == familiar_faces_path else [1,0] for lab in labs])
# train test split
train_x,test_x,train_y,test_y = train_test_split(imgs, labs, test_size=0.05, random_state=random.randint(0,100))
# parameters: total number of images, hight, with
train_x = train_x.reshape(train_x.shape[0], size, size, 3)
test_x = test_x.reshape(test_x.shape[0], size, size, 3)
# convert the train test data to be less than 1
train_x = train_x.astype('float32')/255.0
test_x = test_x.astype('float32')/255.0

print('train size:%s, test size:%s' % (len(train_x), len(test_x)))
# batch size is 100
batch_size = 100
num_batch = len(train_x) // batch_size

x = tf.placeholder(tf.float32, [None, size, size, 3])
y_ = tf.placeholder(tf.float32, [None, 2])

keep_prob_5 = tf.placeholder(tf.float32)
keep_prob_75 = tf.placeholder(tf.float32)

def weightVar(shape):
    init = tf.random_normal(shape, stddev=0.01)
    return tf.Variable(init)

def biasVar(shape):
    init = tf.random_normal(shape)
    return tf.Variable(init)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def maxPool(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def dropout(x, keep):
    return tf.nn.dropout(x, keep)

def cnnLayer():
    #first layer
    W1 = weightVar([3,3,3,32]) 
    b1 = biasVar([32])
    conv1 = tf.nn.relu(conv2d(x, W1) + b1)
    pool1 = maxPool(conv1)
    drop1 = dropout(pool1, keep_prob_5)

    # second layer
    W2 = weightVar([3,3,32,64])
    b2 = biasVar([64])
    conv2 = tf.nn.relu(conv2d(drop1, W2) + b2)
    pool2 = maxPool(conv2)
    drop2 = dropout(pool2, keep_prob_5)

    # third layer
    W3 = weightVar([3,3,64,64])
    b3 = biasVar([64])
    conv3 = tf.nn.relu(conv2d(drop2, W3) + b3)
    pool3 = maxPool(conv3)
    drop3 = dropout(pool3, keep_prob_5)

    Wf = weightVar([8*8*64, 512])
    bf = biasVar([512])
    drop3_flat = tf.reshape(drop3, [-1, 8*8*64])
    dense = tf.nn.relu(tf.matmul(drop3_flat, Wf) + bf)
    dropf = dropout(dense, keep_prob_75)

    Wout = weightVar([512,2])
    bout = biasVar([2])
    #out = tf.matmul(dropf, Wout) + bout
    out = tf.add(tf.matmul(dropf, Wout), bout)
    return out

def cnnTrain():
    out = cnnLayer()

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=y_))

    train_step = tf.train.AdamOptimizer(0.01).minimize(cross_entropy)
    # coompare if the label matchs and find out the average
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(out, 1), tf.argmax(y_, 1)), tf.float32))
    # save the loss and accuracy for tensorboard
    tf.summary.scalar('loss', cross_entropy)
    tf.summary.scalar('accuracy', accuracy)
    merged_summary_op = tf.summary.merge_all()
    # initiate data saver
    saver = tf.train.Saver()

    with tf.Session() as session:

        session.run(tf.global_variables_initializer())

        summary_FileWriter = tf.summary.FileWriter('./tmp', graph=tf.get_default_graph())

        for n in range(10):
            #128 batch size
            for i in range(num_batch):
                batch_x = train_x[i*batch_size : (i+1)*batch_size]
                batch_y = train_y[i*batch_size : (i+1)*batch_size]
                # start training
                _,loss,summary = session.run([train_step, cross_entropy, merged_summary_op],
                                           feed_dict={x:batch_x,y_:batch_y, keep_prob_5:0.5,keep_prob_75:0.75})
                summary_FileWriter.add_summary(summary, n*num_batch+i)
                # loss
                print(n*num_batch+i, loss)

                if (n*num_batch+i) % 100 == 0:
                    # accuracy
                    acc = accuracy.eval({x:test_x, y_:test_y, keep_prob_5:1.0, keep_prob_75:1.0})
                    print(n*num_batch+i, acc)
                    # when accuracy > 0.98 save and exit
                    if acc > 0.98 and n > 2:
                        saver.save(session, './train_faces.model', global_step=n*num_batch+i)
                        sys.exit(0)
        print('accuracy is less than 0.98, more training needed, exited!')

cnnTrain()
