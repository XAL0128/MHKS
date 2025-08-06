import numpy as np
from sklearn.metrics import accuracy_score, f1_score


class Metrics():
    def __init__(self):
        self.metrics_dict = {
            'MOSI': self.__eval_mosi,
            'MOSEI': self.__eval_mosei,
            'SIMS': self.__eval_sims,
        }

    def __multiclass_acc(self, y_pred, y_true):
        """
        Compute the multiclass accuracy w.r.t. groundtruth

        :param preds: Float array representing the predictions, dimension (N,)
        :param truths: Float/int array representing the groundtruth classes, dimension (N,)
        :return: Classification accuracy
        """
        return np.sum(np.round(y_pred) == np.round(y_true)) / float(len(y_true))

    def __eval_mosi(self, y_pred, y_true):
        y_pred_a7 = np.clip(y_pred, a_min=-3., a_max=3.)
        y_true_a7 = np.clip(y_true, a_min=-3., a_max=3.)
        mult_a7 = self.__multiclass_acc(y_pred_a7, y_true_a7)

        # negative / positive
        non_zeros = np.array([i for i, e in enumerate(y_true) if e != 0])
        non_zeros_binary_truth = (y_true[non_zeros] > 0)
        non_zeros_binary_preds = (y_pred[non_zeros] > 0)
        # Non0_mae = np.mean(np.absolute(y_pred[non_zeros] - y_true[non_zeros]))
        # Non0_corr = np.corrcoef(y_pred[non_zeros], y_true[non_zeros])[0][1]
        non_zeros_acc2 = accuracy_score(non_zeros_binary_preds, non_zeros_binary_truth)
        non_zeros_f1_score = f1_score(non_zeros_binary_truth, non_zeros_binary_preds, average='weighted')

        # negative / non-negative
        Has0_mae = np.mean(np.absolute(y_pred - y_true))
        Has0_corr = np.corrcoef(y_pred, y_true)[0][1]
        binary_truth = (y_true >= 0)
        binary_preds = (y_pred >= 0)
        acc2 = accuracy_score(binary_preds, binary_truth)
        f_score = f1_score(binary_truth, binary_preds, average='weighted')

        eval_results = {
            "Mult_acc_7": round(mult_a7, 4),
            "Has0_acc_2": round(acc2, 4),
            "Non0_acc_2": round(non_zeros_acc2, 4),
            "Has0_F1_score": round(f_score, 4),
            "Non0_F1_score": round(non_zeros_f1_score, 4),
            "MAE": round(Has0_mae, 4),
            "Corr": round(Has0_corr, 4)
        }
        return eval_results

    def __eval_mosei(self, y_pred, y_true):
        return self.__eval_mosi(y_pred, y_true)

    def __eval_sims(self, y_pred, y_true):
        # test_preds = y_pred.view(-1).cpu().detach().numpy()
        # test_truth = y_true.view(-1).cpu().detach().numpy()
        test_preds = y_pred
        test_truth = y_true
        test_preds = np.clip(test_preds, a_min=-1., a_max=1.)
        test_truth = np.clip(test_truth, a_min=-1., a_max=1.)

        # two classes{[-1.0, 0.0], (0.0, 1.0]}
        ms_2 = [-1.01, 0.0, 1.01]
        test_preds_a2 = test_preds.copy()
        test_truth_a2 = test_truth.copy()
        for i in range(2):
            test_preds_a2[np.logical_and(test_preds > ms_2[i], test_preds <= ms_2[i + 1])] = i
        for i in range(2):
            test_truth_a2[np.logical_and(test_truth > ms_2[i], test_truth <= ms_2[i + 1])] = i

        mae = np.mean(np.absolute(test_preds - test_truth)).astype(
            np.float64)  # Average L1 distance between preds and truths
        corr = np.corrcoef(test_preds, test_truth)[0][1]
        mult_a2 = self.__multiclass_acc(test_preds_a2, test_truth_a2)
        f_score = f1_score(test_truth_a2, test_preds_a2, average='weighted')

        eval_results = {
            "Mult_acc_2": round(mult_a2, 4),
            "F1_score": round(f_score, 4),
            "MAE": round(mae, 4),
            "Corr": round(corr, 4),  # Correlation Coefficient
        }
        return eval_results

    def getMetics(self, datasetName):
        return self.metrics_dict[datasetName.upper()]
