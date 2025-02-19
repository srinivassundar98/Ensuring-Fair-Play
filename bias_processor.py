import pandas as pd
import numpy as np
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from aif360.explainers import MetricTextExplainer
from aif360.algorithms.preprocessing import Reweighing, DisparateImpactRemover, LFR
from sklearn.preprocessing import StandardScaler

class BiasMitigator:
    def __init__(self):
        # Initialize necessary components
        pass

    def preprocess(self, df, method, sens_attr, unprivileged_groups, privileged_groups, label,threshold=None):
        # Convert df to an AIF360 dataset
        dataset = BinaryLabelDataset(favorable_label=1, unfavorable_label=0, df=df,
                                    label_names=[label], protected_attribute_names=[sens_attr])
        
        if method == 'reweighing':
            rw = Reweighing(unprivileged_groups=unprivileged_groups,
                            privileged_groups=privileged_groups)
            mitigated_dataset = rw.fit_transform(dataset)
            # Get the additional instance weights
            weights = mitigated_dataset.instance_weights
        
        elif method == 'disparate_impact_remover':
            di = DisparateImpactRemover()
            mitigated_dataset = di.fit_transform(dataset)
            weights = None  # No weights for DI
        
        elif method == 'lfr':
            lfr = LFR(unprivileged_groups=unprivileged_groups,
                    privileged_groups=privileged_groups)
            mitigated_dataset = lfr.fit_transform(dataset,threshold = threshold)
            
            weights = mitigated_dataset.instance_weights
        
        # Extract features, labels, and possibly weights
        features = mitigated_dataset.features
        labels = mitigated_dataset.labels.ravel()  # Flatten the label array
        feature_names = mitigated_dataset.feature_names

        # Create a DataFrame from the features
        df_mitigated = pd.DataFrame(features, columns=feature_names)
        df_mitigated['label'] = labels
        if weights is not None:
            df_mitigated['weights'] = weights
        
        return df_mitigated
    def compute_bias_metrics(self, df, unprivileged_groups, privileged_groups, sens_attr, label):
        # Compute bias metrics for the dataset
        dataset = BinaryLabelDataset(favorable_label=1, unfavorable_label=0, df=df,
                                     label_names=[label], protected_attribute_names=[sens_attr])
        metric = BinaryLabelDatasetMetric(dataset, 
                                          unprivileged_groups=unprivileged_groups,
                                          privileged_groups=privileged_groups)
        explainer = MetricTextExplainer(metric)

        results = {
            'Mean Difference': metric.mean_difference(),
            'Consistency': metric.consistency(),
            'Statistical Parity Difference': metric.statistical_parity_difference(),
            'Disparate Impact': metric.disparate_impact(),
            'Mean Difference Explanation': explainer.mean_difference(),
            'Consistency Explanation': explainer.consistency(),
            'Statistical Parity Difference Explanation': explainer.statistical_parity_difference(),
            'Disparate Impact Explanation': explainer.disparate_impact()
        }
        return pd.DataFrame([results], index=['Values'])




