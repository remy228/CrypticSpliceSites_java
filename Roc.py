
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
dataPoints = pd.read_csv("Neigh_in_Auth_PWM.txt", header=None, error_bad_lines=False)
name = "Neighbors_in_Authentic_PWM"

# column starts at 0. the 0th column is the class and the 1st is the log probabilities
y_test = np.array(dataPoints)[:, 0]
probas = np.array(dataPoints)[:, 1]

# Compute ROC curve and area the curve
tpr, fpr, thresholds = roc_curve(y_test, probas)
roc_auc = auc(tpr, fpr)
print("Area under the ROC curve : %f" % roc_auc)

# Plot ROC curve
lw = 2
plt.title('ROC Curve-' + name)
plt.plot(tpr, fpr, 'b', color='purple',
         label='AUC = %0.2f' % roc_auc, linewidth=3)

plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--',  color='red', lw = lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig('Neighbors_in_Authentic_PWM.png')
plt.show()
