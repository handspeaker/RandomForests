/************************************************
 *Random Forest Program
 *Function:     implementation of CART--Classification and Regression Tree without pruning.
                for classification, utilize Gini Index as the criterion
                Gini(t)=1-sum(p(j|t),j=[1,J]),assuming there are J classes
                for regression, utilize sum of square residue as the criterion
                I(t)=sum(Ni(left)-E(Ni(left))^2+Ni(right)-E(Ni(right))^2)
 *Author:		handspeaker@163.com
 *CreateTime:	2014.7.10
 *Version:		V0.1
 *************************************************/
#ifndef CARTREE_H
#define CARTREE_H

#include<stdio.h>
#include<math.h>
#include"Sample.h"
#include"Node.h"

class Tree
{
public:
	/*************************************************************
	*MaxDepth:the max Depth of one single tree
	*trainFeatureNumPerNode:the feature number used in every node while training
	*minLeafSample:terminate criterion,the min samples in a leaf   
	*minInfoGain:terminate criterion,the min information gain in
	*a node if it can be splitted
	*isRegression:if the problem is regression(true) or classification(false)
	**************************************************************/
	Tree(int MaxDepth,int trainFeatureNumPerNode,int minLeafSample,float minInfoGain,bool isRegression);
	virtual ~Tree();
	virtual void train(Sample*Sample)=0;
	Result predict(float*data);
	inline Node**getTreeArray(){return _cartreeArray;};
	virtual void createNode(int id,int featureIndex,float threshold)=0;
protected:
	bool _isRegression;	//the type of this tree
	int _MaxDepth;
	int _nodeNum;  //the number of node,=2^_MaxDepth-1
	int _minLeafSample;
	int _trainFeatureNumPerNode;
	float _minInfoGain;
//	Sample*_samples;//all samples used while training the tree
	Node** _cartreeArray;  //utilize a node array to store the tree,
						   //every node is a split or leaf node
};
//Classification Tree
class ClasTree:public Tree
{
public:
	ClasTree(int MaxDepth,int trainFeatureNumPerNode,int minLeafSample,float minInfoGain,bool isRegression);
	~ClasTree();
	void train(Sample*Sample);
	void createNode(int id,int featureIndex,float threshold);
	void createLeaf(int id,float clas,float prob);
};
//Regression Tree
class RegrTree:public Tree
{
public:
	RegrTree(int MaxDepth,int trainFeatureNumPerNode,int minLeafSample,float minInfoGain,bool isRegression);
	~RegrTree();
	void train(Sample*Sample);
	void createNode(int id,int featureIndex,float threshold);
	void createLeaf(int id,float value);
};
#endif//CARTREE_H
