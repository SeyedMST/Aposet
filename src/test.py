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


# with tf.Session() as sess:
#     initializer = tf.global_variables_initializer()
#     sess.run(initializer)
#
#     # result = []
#     # result.append(tf.while_loop(condition, body, [x]))
#     # result = tf.concat(0, result)
#     #result = tf.ceil(g1)
#
#     _truth = []
#     _input_vector = []
#     for i in range(1):
#         _truth.append(np.array([10.0,30.0,-8.0,4.0]))
#         _input_vector.append(np.array([0,1,2,1]))
#
#     # print (_truth)
#     feed_dict = {
#         p0: tuple(_truth),
#         q0: tuple(_input_vector),
#     }
#     loss_value = sess.run([loss], feed_dict=feed_dict)
#
#     print (loss_value)
#     #print(p.eval())


def dcg_at_k(r, k, method):
    r = np.asfarray(r)[:k]
    print (r)
    print (r.size)
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    print ("hi")
    return 0.

r = [2, 3, 0, 1, 1, 1, 1]
x = np.zeros(10)
print (x)
print(dcg_at_k(sorted(r, reverse=True), 1, 1))
print (dcg_at_k(r, 1, 1))