import tensorflow as tf
import numpy as np

def get_place_holder (q_count, type, shape):
    ans = ()
    for _ in range(q_count):
        ans += (tf.placeholder(type, shape=shape),)
    return ans


p0 = get_place_holder(1, tf.float32,
                              [None])  # q_count*[tf.placeholder(tf.float32, [None])] # [batch_size]
q0 = get_place_holder(1, tf.float32, [None])

p = tf.nn.softmax(p0[0])  # [question_count, answer_count]
q = tf.nn.softmax(q0[0])

# p = tf.convert_to_tensor([10.0,30.0,-8.0,4.0])
# q = tf.convert_to_tensor([10.0,10.0,4.0,3.0])
# q = tf.nn.softmax(q)  # [question_count, answer_count]
# p = tf.nn.softmax(p)

jp = np.ones(4, np.float32)
jp [1] = 0

loss = tf.multiply(jp, p)



#loss = tf.reduce_sum(
#                        tf.multiply(p, tf.log(p)) - tf.multiply(p, tf.log(q))
#                               )


with tf.Session() as sess:
    initializer = tf.global_variables_initializer()
    sess.run(initializer)

    # result = []
    # result.append(tf.while_loop(condition, body, [x]))
    # result = tf.concat(0, result)
    #result = tf.ceil(g1)

    _truth = []
    _input_vector = []
    for i in range(1):
        _truth.append(np.array([10.0,30.0,-8.0,4.0]))
        _input_vector.append(np.array([0,1,2,1]))

    # print (_truth)
    feed_dict = {
        p0: tuple(_truth),
        q0: tuple(_input_vector),
    }
    loss_value = sess.run([loss], feed_dict=feed_dict)

    print (loss_value)
    #print(p.eval())