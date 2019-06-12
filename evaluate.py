import numpy as np
import tensorflow as tf
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import scikitplot as skplt

# ===- load data and preprocessing =====
x = np.load('data\\input.npy')
y = np.load('data\\output.npy').astype('int32')

num_class = 2
y = tf.keras.utils.to_categorical(y,num_classes=num_class)

# ======= load trained model ======
model =  tf.keras.models.load_model('model\\lstm_downsampled.h5')

# =========== Evaluation ==========
score = model.evaluate(x, y, verbose=0)
print('Error: %.4f' % score[0])
print('Accuracy: %.4f' % score[1])

# predict crisp classes for test set
yhat = model.predict(x, verbose=0)
yhat_classes = np.asarray([np.argmax(yhat_test, axis=None, out=None) for yhat_test in yhat], dtype=np.int32)
y_classes = np.asarray([np.argmax(y_test, axis=None, out=None) for y_test in y], dtype=np.int32)

# calculate precision
precision = precision_score(y_classes, yhat_classes)
print('Precision: ', precision)
# Calculate recall
recall = recall_score(y_classes, yhat_classes)
print('Recall: ', recall)
# Calculate F1 score
f1 = f1_score(y_classes, yhat_classes)
print('F1 score: ', f1)

# confusion matrix
skplt.metrics.plot_confusion_matrix(y_classes, yhat_classes, normalize=False)
plt.show()