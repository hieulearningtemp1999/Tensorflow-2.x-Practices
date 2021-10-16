import tensorflow as tf
from loss import loss
from tqdm import tqdm

@tf.function
def test_step(model, data, target):
    pred = model(data)
    test_loss = loss(pred, target)
    pred_label = tf.argmax(pred, axis=1)
    target_label = tf.argmax(target, axis=1)
    check_equal = (tf.cast(pred_label, tf.int64)) == (tf.cast(target_label, tf.int64))
    correct = tf.reduce_sum(tf.cast(check_equal, tf.float32))
    return test_loss, correct


def run(test_dataset, model):
    print("Testing...")
    test_losses = []
    corrects = 0
    loader = tqdm(test_dataset)
    for data, target in loader:
        test_loss, correct = test_step(model, data, target)
        test_losses.append(test_loss)
        corrects += correct
        loader.set_postfix(test_loss_batch=test_loss)

    return sum(test_losses) / len(test_losses), corrects