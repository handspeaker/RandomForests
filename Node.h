/************************************************
 *Random Forest Program
 *Function:		implementation of two kinds of node
                for classification and regression
 *Author:		handspeaker@163.com
 *CreateTime:	2014.7.10
 *Version:		V0.1
 *************************************************/
#ifndef NODE_H
#define NODE_H

#include"Sample.h"
#include<stdio.h>

struct Result
{
    float label;  //label or value
    float prob;  //prob or 1
};

struct Pair
{
	float feature;
	int id;
};

int compare_pair( const void* a, const void* b );

class Node
{
public:
	Node();
	virtual ~Node();
	//sort the selected samples in the ascending order based on featureId
	void sortIndex(int featureId);
	Sample*_samples;//the samples hold by this node
	//set this node as leaf node
	inline void setLeaf(bool flag){_isLeaf=flag;};
	inline bool isLeaf(){return _isLeaf;};
	//calculate the information gain
	virtual void calculateInfoGain(Node**_cartreeArray,int id,float minInfoGain)=0;
	virtual void calculateParams()=0;
	//create a leaf node
	virtual void createLeaf()=0;
	//predict the data
	virtual int predict(float*data,int id)=0;
	virtual void getResult(Result&r)=0;

	inline int getFeatureIndex(){return _featureIndex;};
	inline void setFeatureIndex(int featureIndex){_featureIndex=featureIndex;};
	inline float getThreshold(){return _threshold;};
	inline void setThreshold(float threshold){_threshold=threshold;};
protected:
	bool _isLeaf;
	int _featureIndex;
	float _threshold;
};

class ClasNode:public Node
{
public:
	ClasNode();
	~ClasNode();
	void calculateInfoGain(Node**_cartreeArray,int id,float minInfoGain);
	void calculateParams();
	void createLeaf();
	int predict(float*data,int id);
	void getResult(Result&r);

	inline float getClass(){return _class;};
	inline float getProb(){return _prob;};

	inline void setClass(float clas){_class=clas;};
	inline void setProb(float prob){_prob=prob;};
	//parameters for training
	float _gini;
	float*_probs;
private:
	float _class;  //the class
	float _prob;	//the probablity
};

class RegrNode:public Node
{
public:
	RegrNode();
	~RegrNode();
	void calculateInfoGain(Node**_cartreeArray,int id,float minInfoGain);
	void calculateParams();
	void createLeaf(); 
	int predict(float*data,int id);
	void getResult(Result&r);

	inline float getValue(){return _value;};
	inline void setValue(float value){_value=value;};
	//parameters for training
	float _mean;  
	float _variance; 
private:
	float _value; //the regression value
};

#endif//NODE_H
