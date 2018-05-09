import tensorflow as tf

from ..toolkit.tool import cast_to_float32


def compute_adem_score(context, model_response,
                       mr_dim, ct_dim):
    [context, model_response] = cast_to_float32([context, model_response])
    with tf.variable_scope('score'):
        M = tf.Variable(name='M', initial_value=tf.random_normal(
            shape=[mr_dim, ct_dim]), dtype=tf.float32)
        # N = tf.Variable(name='N', initial_value=tf.random_normal(
        #     shape=[mr_dim, rr_dim]), dtype=tf.float32)

        alpha = tf.Variable(
            name='alpha', initial_value=tf.random_uniform([1], maxval=5.0))

        beta = tf.Variable(
            name='beta', initial_value=tf.random_uniform([1], maxval=5.0))

        # tmp_score = (tf.reduce_sum((context * tf.matmul(model_response, M)), axis=1) +
        #          tf.reduce_sum((reference_response * tf.matmul(model_response, N)), axis=1) -
        #          alpha) / beta
        # score = tf.Variable(tmp_score,shape= [64], name="model_score1", dtype=tf.float32)
        tmp_score = tf.divide((tf.reduce_sum((context * tf.matmul(model_response, M)), axis=1) - alpha), beta, name= "model_score1")
        # score.assign(tmp_score)

    return tmp_score, M


def tf_static_adem_score(context, model_response, reference_response):
    rr_size, rr_dim = reference_response.get_shape().as_list()
    mr_size, mr_dim = model_response.get_shape().as_list()
    ct_size, ct_dim = context.get_shape().as_list()
    with tf.control_dependencies(
        [tf.assert_equal(rr_size, mr_size, message='responses size not equal'),
         tf.assert_equal(ct_size, mr_size, message='context response size not equal')]):
        score, M = compute_adem_score(
            context, model_response, mr_dim, ct_dim)
    return score, M


def tf_dynamic_adem_score(context, model_response, shape_info):
    if not isinstance(shape_info, dict):
        raise TypeError('shape info should be dict.')

    # for info in ['rr_dim', 'mr_dim', 'ct_dim']:
    for info in ['mr_dim', 'ct_dim']:
        if info not in shape_info:
            raise KeyError('{} is not in shape_info dict'.format(info))

    score, M = compute_adem_score(context
                                     , model_response
                                     , mr_dim=shape_info['mr_dim']
                                     , ct_dim=shape_info['ct_dim']
                                     )
    return score, M
