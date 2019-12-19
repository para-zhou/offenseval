# Offenseval
SemEval 2019 - Task 6 - Identifying and Categorizing Offensive Language in Social Media 
Sub-task A - Offensive language identification;  [Offensive: OFF, Not Offensive: NOT]
## TODO 
- 完成实验
	* Pre Processing (Para)
	* Classification 
		- Bayes
		- NN
	* Visulization (可能耗时较久)

- 完成报告 (不做实验是否可以完成[Y/N])
	* [Y]Related work (可能耗时较久)
	* [N]Data Stastics (1/2)
	* Methods
		- [Y]算法选择及原因说明
		- [N]具体实现
	* [N]Results
	* Conclusion

## Plan
[Y]： 已完成
- 整理数据和代码 for preprocessing
	* [Y]subtask B/C 相关都放在 useless文件夹
	* [Y]Zeyad源代码放在code\_Zeyad文件夹
- 预处理
	* [Y]参考Zeyad代码完成数据读入
	* 完成预处理
	* 同时完成report data section)
- 模型
	* 将实验过程汇总在一个ipynb
	* 完成report algorithm section II
	* 完成training
- Test
	* 完成可视化
	* 完成report Results section
- 完成report其他部分


## Notes from Zeyad
## Implementation

### Preprocessing
Tokenization, Stopwords Removal, Lemmatizaion, Stemming

### Vectorization
TFIDF, Count, Word2Vec, GloVe, fastText

### Classification
KNN, Naïve Bayes, SVM, Decision Trees, Random Forest, Logistic Regression, MLP, Adaboost, Bagging
#### Deeplearning
LSTM, 1-D CNN

## Running
- Install requiremetns using `pip3 install -r requirements.txt`
 
- `python3 tune.py` to do a complete search on all combinations of prepocessing, vectorization and non-deep classification techniques while tuning the classifiers hyper-params.

- `python3 train.py` to train one of the deeplearning models.

## Sample tuning.py output
![alt text](https://github.com/ZeyadZanaty/offenseval/blob/master/docs/tuning-reults/tuning-b-f1/LogisticRegression.png?raw=true "Logistic Regression")
![alt text](https://github.com/ZeyadZanaty/offenseval/blob/master/docs/tuning-reults/tuning-b-f1/SVC.png?raw=true "SVC")
![alt text](https://github.com/ZeyadZanaty/offenseval/blob/master/docs/tuning-reults/tuning-a/RandomForest.png?raw=true "RF")
