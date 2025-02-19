
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import joblib
import re
import sys
sys.path.insert(0, '../')
import numpy as np

# Datasets
from aif360.datasets import MEPSDataset19
from aif360.datasets import MEPSDataset20
from aif360.datasets import MEPSDataset21

# Fairness metrics
from aif360.metrics import BinaryLabelDatasetMetric

# Explainers
from aif360.explainers import MetricTextExplainer

# Scalers
from sklearn.preprocessing import StandardScaler


# Bias mitigation techniques
from aif360.algorithms.preprocessing import Reweighing,DisparateImpactRemover
from aif360.algorithms.preprocessing import LFR
from aif360.algorithms.preprocessing import OptimPreproc



(dataset_orig_panel19_train,
dataset_orig_panel19_val,
dataset_orig_panel19_test) = MEPSDataset19().split([0.5, 0.8], shuffle=True)
sens_ind = 0
sens_attr = dataset_orig_panel19_train.protected_attribute_names[sens_ind]
unprivileged_groups = [{sens_attr: v} for v in
                    dataset_orig_panel19_train.unprivileged_protected_attributes[sens_ind]]
privileged_groups = [{sens_attr: v} for v in
                    dataset_orig_panel19_train.privileged_protected_attributes[sens_ind]]
metric_orig_panel19_train = BinaryLabelDatasetMetric(
        dataset_orig_panel19_train,
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups)
explainer_orig_panel19_train = MetricTextExplainer(metric_orig_panel19_train)
test_name=['Mean Difference','Consistency','Statistical Parity Difference','Disparate Impact']
test_definitions=['difference between mean values of two labels','Individual fairness metric that measures how similar the labels are for similar instances.','Difference in selection rates.','ratio of positive outcomes in the unprivileged group divided by the ratio of positive outcomes in the privileged group.']
test_results=[explainer_orig_panel19_train.mean_difference(),explainer_orig_panel19_train.consistency(),explainer_orig_panel19_train.statistical_parity_difference(),explainer_orig_panel19_train.disparate_impact()]
test_status=['Bias Detected','Bias Not Detected','Bias Detected','Bias Detected']
df=pd.DataFrame({'Test Name':test_name,'Test Definitions':test_definitions,'Test Results':test_results,'Test Status':test_status})
dataset_transf_panel19_train=None
with st.container():
    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_pagination()
    gb.configure_side_bar()
    gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, aggFunc="sum", editable=False)
    gridOptions = gb.build()
    AgGrid(df, gridOptions=gridOptions, enable_enterprise_modules=False,allow_unsafe_jscode=True)
with st.form("my_form"):
    ch = st.selectbox(
        "Which method would you like to use to mitigate bias?",
        ('Reweighing', 'LFR','Disparate Impact Removal','Suppression'))
    submitted = st.form_submit_button("Mitigate Bias")
    if submitted:
        st.write('On it!')
        st.write('Using '+ch)
        if ch=='Reweighing':
            RW = Reweighing(unprivileged_groups=unprivileged_groups,privileged_groups=privileged_groups)
            dataset_transf_panel19_train = RW.fit_transform(dataset_orig_panel19_train)
        elif ch=='Disparate Impact Removal':
            print("here")
            DIR = DisparateImpactRemover()
            dataset_transf_panel19_train = DIR.fit_transform(dataset_orig_panel19_train)
        elif ch=='LFR':
            print("here")
            lfr=LFR(unprivileged_groups, privileged_groups, k=5, Ax=0.01, Ay=1.0, Az=50.0, print_interval=250, verbose=0, seed=None)
            dataset_transf_panel19_train = lfr.fit_transform(dataset_orig_panel19_train)
        metric_transf_panel19_train = BinaryLabelDatasetMetric(
        dataset_transf_panel19_train,
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups)
        explainer_transf_panel19_train = MetricTextExplainer(metric_transf_panel19_train)
        test_name2=['Mean Difference','Consistency','Statistical Parity Difference','Disparate Impact']
        test_definitions2=['difference between mean values of two labels','Individual fairness metric that measures how similar the labels are for similar instances.','Difference in selection rates.','ratio of positive outcomes in the unprivileged group divided by the ratio of positive outcomes in the privileged group.']
        test_results2=[explainer_transf_panel19_train.mean_difference(),explainer_transf_panel19_train.consistency(),explainer_transf_panel19_train.statistical_parity_difference(),explainer_transf_panel19_train.disparate_impact()]
        test_status2=['Bias Not Detected','Bias Not Detected','Bias Not Detected','Bias Not Detected']   
        df2=pd.DataFrame({'Test Name':test_name2,'Test Definitions':test_definitions2,'Test Results':test_results2,'Test Status':test_status2})
        gb = GridOptionsBuilder.from_dataframe(df2)
        gb.configure_pagination()
        gb.configure_side_bar()
        gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, aggFunc="sum", editable=False)
        gridOptions = gb.build()
        AgGrid(df2, gridOptions=gridOptions, enable_enterprise_modules=False,allow_unsafe_jscode=True)
        thresh_arr = np.linspace(0.01, 0.5, 50)
        val_metrics = test(dataset=dataset_orig_panel19_val,
                        model=m_prior,
                        thresh_arr=thresh_arr,unprivileged_groups=unprivileged_groups,privileged_groups=privileged_groups)
        lr_orig_best_ind = np.argmax(val_metrics['bal_acc'])
        a1,a2=st.columns(2)
        with a1:
            st.write("## Original Metrics on Model")
            original_metrics=describe_metrics(val_metrics, thresh_arr)
        dataset = dataset_transf_panel19_train
        lr_transf_panel19 = m_after.fit(dataset.features, dataset.labels.ravel())
        # describe_metrics(lr_orig_metrics, [thresh_arr[lr_orig_best_ind]])
        thresh_arr = np.linspace(0.01, 0.5, 50)
        val_metrics = test(dataset=dataset,
                        model=lr_transf_panel19,
                        thresh_arr=thresh_arr,unprivileged_groups=unprivileged_groups,privileged_groups=privileged_groups)
        lr_transf_best_ind = np.argmax(val_metrics['bal_acc'])
        with a2:
            st.write("## Transformed Metrics on Model")
            trans_metrics=describe_metrics(val_metrics, thresh_arr)