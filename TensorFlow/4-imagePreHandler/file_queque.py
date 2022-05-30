import tensorflow as tf

filenames = tf.train.match_filenames_once('.\data\*.csv')
filename_queue = tf.train.string_input_producer(filenames, shuffle=False, num_epochs=3)

reader = tf.TextLineReader()
_, value = reader.read(filename_queue)

example, label = tf.decode_csv(value, record_defaults=[['null'], ['null']])
## example_batch, label_batch = tf.train.batch([exmaple, label], batch_size=5)

## record_list = [ tf.decode_csv(value, record_defaults=[['null'], ['null']]) for _ in range(2)]
## example_batch, label_batch = tf.train.batch_join(record_list, batch_size=5)


init_op = tf.local_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    coord = tf.train.Coordinator() #线程管理协调器
    threads = tf.train.start_queue_runners(coord=coord) #启动QueueRunner
    """
    try:
        while not coord.should_stop():
            print(sess.run([example, label]))
    except tf.errors.OutOfRangeError:
        print('Epochs complete!')
    finally:
        coord.request_stop()
    """
    for _ in range(5):
        print(sess.run([example, label]))

    coord.request_stop()
    coord.join(threads)
