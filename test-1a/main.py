import sys

import tensorflow as tf

sys.path.append("../")
from config import cfg

from capslib import utils as u

from capslib.capsnet import CapsNet


def main(_):
    # Get the batches of training and testing data
    training_X_batch, training_Y_batch, testing_batch = u.create_training_and_testing_batches()

    # Create a capsule network
    capsnet = CapsNet()

    # Get training errors and reconstructions
    (train_total_error,
     train_margin_error,
     train_reconstruction_error,
     train_reconstructed_combined_image,
     train_reconstructed_first_image,
     train_reconstructed_second_image,
     train_accuracy,
     train_memo_image_reconstructions,
     train_memo_margin_loss,
     train_memo_accuracy) = capsnet.compute_output(training_X_batch, training_Y_batch, keep_prob=0.5)

    # Create operations to minimize training loss
    train_op = tf.train.AdamOptimizer().minimize(train_total_error)
    train_memo_op = tf.train.AdamOptimizer().minimize(train_memo_margin_loss)

    # Get test errors and reconstructions
    (test_0px_total_error,
     test_0px_margin_error,
     test_0px_reconstruction_error,
     _,  # For testing, we don't care about the reconstructed images
     _,  # Reconstructed image 1
     _,  # Reconstructed image 2
     test_0px_accuracy,
     test_0px_memo_image_reconstructions,
     test_0px_memo_margin_loss,
     test_0px_memo_accuracy) = capsnet.compute_output(testing_batch[0], testing_batch[1], keep_prob=1)

    (test_2px_total_error,
     test_2px_margin_error,
     test_2px_reconstruction_error,
     _,  # For testing, we don't care about the reconstructed images
     _,  # Reconstructed image 1
     _,  # Reconstructed image 2
     test_2px_accuracy,
     test_2px_memo_image_reconstructions,
     test_2px_memo_margin_loss,
     test_2px_memo_accuracy) = capsnet.compute_output(testing_batch[2], testing_batch[3], keep_prob=1)

    (test_4px_total_error,
     test_4px_margin_error,
     test_4px_reconstruction_error,
     _,  # For testing, we don't care about the reconstructed images
     _,  # Reconstructed image 1
     _,  # Reconstructed image 2
     test_4px_accuracy,
     test_4px_memo_image_reconstructions,
     test_4px_memo_margin_loss,
     test_4px_memo_accuracy) = capsnet.compute_output(testing_batch[4], testing_batch[5], keep_prob=1)

    (test_6px_total_error,
     test_6px_margin_error,
     test_6px_reconstruction_error,
     _,  # For testing, we don't care about the reconstructed images
     _,  # Reconstructed image 1
     _,  # Reconstructed image 2
     test_6px_accuracy,
     test_6px_memo_image_reconstructions,
     test_6px_memo_margin_loss,
     test_6px_memo_accuracy) = capsnet.compute_output(testing_batch[6], testing_batch[7], keep_prob=1)

    (test_8px_total_error,
     test_8px_margin_error,
     test_8px_reconstruction_error,
     _,  # For testing, we don't care about the reconstructed images
     _,  # Reconstructed image 1
     _,  # Reconstructed image 2
     test_8px_accuracy,
     test_8px_memo_image_reconstructions,
     test_8px_memo_margin_loss,
     test_8px_memo_accuracy) = capsnet.compute_output(testing_batch[8], testing_batch[9], keep_prob=1)

    with tf.Session() as sess:
        # Initialize the graph, and receive the queue coordinator and the training monitor
        coord, training_monitor = u.init(sess)

        # Pretrain the network on the first part--classifying and splitting
        for i in range(1, 1000):
            sys.stdout.write("Pretraining: " + str(i))
            # sys.stdout.write("Pretraining: (%d/1000)   \r" % (i))
            sys.stdout.flush()
            sess.run([train_op])

        # Now, run the actual training
        for batch_num in range(1, int(600000 / cfg.batch_size) * cfg.num_epochs):

            # Run the training operations, and get the corresponding errors
            (curr_train_total_error,
             curr_train_margin_error,
             curr_train_reconstruction_error,
             curr_train_accuracy,
             curr_memo_margin_loss,
             curr_memo_accuracy,
             _,
             _) = sess.run([train_total_error,
                            train_margin_error,
                            train_reconstruction_error,
                            train_accuracy,
                            train_memo_margin_loss,
                            train_memo_accuracy,
                            train_op,
                            train_memo_op])

            # Add all the losses to the training monitor
            training_monitor.addsix(curr_train_total_error,
                                    curr_train_margin_error,
                                    curr_train_reconstruction_error,
                                    curr_train_accuracy,
                                    curr_memo_margin_loss,
                                    curr_memo_accuracy)

            print(("Step: " + str(batch_num)).ljust(10)[:10]),
            print(("Total loss: " + str(curr_train_total_error)).ljust(25)[:25]),
            print(("Margin loss: " + str(curr_train_margin_error)).ljust(25)[:25]),
            print(("Reconstruct loss: " + str(curr_train_reconstruction_error)).ljust(25)[:25]),
            print(("Train Accuracy: " + str(curr_train_accuracy)).ljust(25)[:25]),
            print(("Memo margin: " + str(curr_memo_margin_loss)).ljust(25)[:25]),
            print(("Memo accuracy: " + str(curr_memo_accuracy)).ljust(25)[:25]),
            print("\n")

            # Every 100 iterations, display the current testing results.
            if batch_num % 100 == 9:
                # Get the errors on the test data
                (curr_total_error,
                 curr_margin_error,
                 curr_reconstruction_error,
                 curr_accuracy,
                 curr_memo_margin_error,
                 curr_memo_accuracy,
                 curr_accuracy_2px,
                 curr_accuracy_4px,
                 curr_accuracy_6px,
                 curr_accuracy_8px) = sess.run([test_0px_total_error,
                                                test_0px_margin_error,
                                                test_0px_reconstruction_error,
                                                test_0px_accuracy,
                                                test_0px_memo_margin_loss,
                                                test_0px_memo_accuracy,
                                                test_2px_accuracy,
                                                test_4px_accuracy,
                                                test_6px_accuracy,
                                                test_8px_accuracy])

                # Add the losses to the training monitor
                training_monitor.addsixtest(curr_total_error,
                                            curr_margin_error,
                                            curr_reconstruction_error,
                                            curr_accuracy,
                                            curr_memo_margin_error,
                                            curr_memo_accuracy,
                                            curr_accuracy_2px,
                                            curr_accuracy_4px,
                                            curr_accuracy_6px,
                                            curr_accuracy_8px)

                # Display the current training performance
                training_monitor.prints()


if __name__ == "__main__":
    tf.app.run()
