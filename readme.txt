readme.txt 2022/6/12

You should:
1. Download the datasets and code.
2. Installation the environment. All requirements are detailed in requirements.txt. Anaconda is strongly recommended.
$conda create -n GraphLoc python=3.6
$source activate GraphLoc
$pip install -r requirements.txt
3. Follow the following steps to generate the results of our paper.

Step1: Extracting features of IHC images by CNN module. (same with ImPLoc.)
Step2: Building graph. (prepare_distanceMatrix_Adj.py)
Step3: Training Graph neural network. (train_HPA.sh)
Step4: Inference subcellular location(s) with dynamic threshold. (inference_HPA.py)
Step4: Identifying of location biomarkers. (biomarkerAnalysis.m)
Step5: Analyzing protein networks. (Protein_analysis_KEGG.py)



