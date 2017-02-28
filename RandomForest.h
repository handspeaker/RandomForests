/************************************************
*Random Forest Program
*Function:		trian&test Random Forest Model
*Author:		handspeaker@163.com
*CreateTime:	2014.7.10
*Version:		V0.1
*************************************************/
#ifndef RANDOM_FOREST_H
#define RANDOM_FOREST_H

#include<math.h>
#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include"Tree.h"
#include"Sample.h"

class RandomForest
{
public:
	/*************************************************************
	*treeNum:	the number of trees in this forest
	*maxDepth:	the max Depth of one single tree
	*minLeafSample:terminate criterion,the min samples in a leaf              
	*minInfoGain:terminate criterion,the min information
	*            gain in a node if it can be splitted
	**************************************************************/
	RandomForest(int treeNum,int maxDepth,int minLeafSample,float minInfoGain);
	RandomForest(const char*modelPath);
	~RandomForest();
	/*************************************************************
	*trainset:	the trainset,every row is a sample,every column is 
	*a feature,the total size is SampleNum*featureNum
	*labels:the labels or regression values of the trainset,
	*the total size is SampleNum
	*SampleNum:the total number of trainset
	*featureNum:the number of features
	*classNum:the class number,regressiong is 1
	*isRegression:if the problem is regression(true) or classification(false)
	*trainFeatureNumPerNode:the feature number used in every node while training
	*************************************************/
	void train(float**trainset,float*labels,int SampleNum,int featureNum,
			   int classNum,bool isRegression,int trainFeatureNumPerNode);
	void train(float**trainset,float*labels,int SampleNum,int featureNum,
			   int classNum,bool isRegression);
	/************************************************
	*sample: a single sample
	*response: the predict result
	*************************************************/
	void predict(float*sample,float&response);
	/************************************************
	*testset: the test set
	*SampleNum:the sample number in the testset
	*responses: the predict results
	*************************************************/
	void predict(float**testset,int SampleNum,float*responses);
	/************************************************
	*path: the path to save the model
	*************************************************/
	void saveModel(const char*path);
	/************************************************
	*path: the path to read the model
	*************************************************/
	void readModel(const char*path);
private:
	int _trainSampleNum;  //the total training sample number
	int _testSampleNum;  //the total testing sample number
	int _featureNum;  //the feature dimension 
	int _trainFeatureNumPerNode;  //the feature number used in a node while training
	int _treeNum;  //the number of trees
	int _maxDepth;  //the max depth which a tree can reach
	int _classNum;  //the number of classes(if regresssion,set it to 1)
	bool _isRegression;  //if it is a regression problem
	int _minLeafSample;  //terminate condition£ºthe min samples in a node
	float _minInfoGain;  //terminate condition£ºthe min information gain in a node
	Tree**_forest;//to store every tree(classification tree or regression tree)
	Sample*_trainSample;  //hold the whole trainset and some other infomation
};

#endif//RANDOM_FOREST_H
