#fix cell that wasnt working
print("\nSimple Decision Stump Accuracy: "+str(accuracy_score(true, dec_stump_preds)))
print("\nSimple Decision Stump F1 Score: "+str(f1_score(true, dec_stump_preds, pos_label='C')))
#roc-auc and dictionary output
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def evaluate_classification(true, pred):
   accuracy = accuracy_score(true, pred)
   f1 = f1_score(true, pred, pos_label='C')
   roc_auc = roc_auc_score(true=='C', pred=='C')
   fpr, tpr, thresholds = roc_curve(true=='C', pred=='C')
   plt.plot(fpr, tpr)
   plt.plot([0, 1], [0, 1])
   plt.xlabel('False Positive Rate')
   plt.ylabel('True Positive Rate')
   plt.title('ROC Curve')
   plt.show()
   return {'accuracy': accuracy, 'f1': f1, 'roc score': roc_auc}

e_metrics = evaluate_classification(true, dec_stump_preds)
print(e_metrics)
