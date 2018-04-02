import tensorflow as tf
import os,time
flags = tf.app.flags


############################
#    hyper parameters      #
############################

# For separate margin loss
flags.DEFINE_float('mc_m_plus', 0.9, 'the parameter of m plus')
flags.DEFINE_float('mc_m_minus', 0.1, 'the parameter of m minus')
flags.DEFINE_float('mc_lambda_val', 0.5, 'down weight of the loss for absent digit classes')

# for training
flags.DEFINE_integer('mc_batch_size', 100, 'batch size')
flags.DEFINE_integer('mc_num_epochs', 50, 'epoch')
flags.DEFINE_integer('mc_iter_routing', 3, 'number of iterations in routing algorithm')
flags.DEFINE_boolean('mc_mask_with_y', True, 'use the true label to mask out target capsule or not')

flags.DEFINE_float('mc_stddev', 0.01, 'stddev for W initializer')
flags.DEFINE_float('mc_regularization_scale', 0.392, 'regularization coefficient for reconstruction loss, default to 0.0005*784=0.392')


############################
#   environment setting    #
############################
#home = os.path.dirname(os.path.abspath(__file__))
flags.DEFINE_string('mc_dataset_full', '/media/data4/affnist/mmnist', 'the path for the full MMNIST dataset')
flags.DEFINE_string('mc_dataset', '/home/urops/andrewg/capsule-b/submmnist6', 'the path for the subMMNIST dataset')
flags.DEFINE_boolean('mc_is_training', True, 'train or predict phase')
flags.DEFINE_integer('mc_num_threads', 8, 'number of threads of enqueueing exampls')
flags.DEFINE_string('mc_logdir', 'logdir', 'logs directory')
flags.DEFINE_integer('mc_train_sum_freq', 100, 'the frequency of saving train summary(step)') # 50
flags.DEFINE_integer('mc_test_sum_freq', 200, 'the frequency of saving test summary(step)') # 500
flags.DEFINE_integer('mc_save_freq', 2, 'the frequency of saving model(epoch)')
flags.DEFINE_string('mc_results', 'results', 'path for saving results')

############################
#   distributed setting    #
############################
#flags.DEFINE_integer('num_gpu', 2, 'number of gpus for distributed training')
#flags.DEFINE_integer('batch_size_per_gpu', 128, 'batch size on 1 gpu')
#flags.DEFINE_integer('thread_per_gpu', 4, 'Number of preprocessing threads per tower.')

cfg = tf.app.flags.FLAGS
# tf.logging.set_verbosity(tf.logging.INFO)
