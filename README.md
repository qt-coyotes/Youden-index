# Youden-index
Purpose of notebook **Youden_index_generate_report_demo.ipynb** is to demonstrate the process of getting optimal cutoff (based on Youden index) from training ROC, and generating performance metrics based on the selected cutoff for test predictions. This can be used for CV fold or holdout testing in the similar way.
After using the Youden Index to select the optimal cutoff, the precision and recall of the model may become higher than before.
# Usage
Write your training label, training probability, testing label, testing probability to csv file, and run Youden_index_generate_report_demo.ipynb.
