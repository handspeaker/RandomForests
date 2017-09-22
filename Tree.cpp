#include"Tree.h"

Tree::Tree(int MaxDepth,int trainFeatureNumPerNode,int minLeafSample,float minInfoGain,bool isRegression)
{
	_MaxDepth=MaxDepth;
	_trainFeatureNumPerNode=trainFeatureNumPerNode;
	_minLeafSample=minLeafSample;
	_minInfoGain=minInfoGain;
	_nodeNum=static_cast<int>(pow(2.0,_MaxDepth)-1);
	_cartreeArray=new Node*[_nodeNum];
	_isRegression=isRegression;
	for(int i=0;i<_nodeNum;++i)
	{_cartreeArray[i]=NULL;}
}

Tree::~Tree()
{
	if(_cartreeArray!=NULL)
	{
		for(int i=0;i<_nodeNum;++i)
		{
			if(_cartreeArray[i]!=NULL)
			{
				delete _cartreeArray[i];
				_cartreeArray[i]=NULL;
			}
		}
		delete[] _cartreeArray;
		_cartreeArray=NULL;
	}
}

Result Tree::predict(float*data)
{
	int position=0;
	Node*head=_cartreeArray[position];
	while(!head->isLeaf())
	{
		position=head->predict(data,position);
		head=_cartreeArray[position];
	}
	Result r;
	head->getResult(r);
	return r;
}
/************************************************/
//Classification Tree
ClasTree::ClasTree(int MaxDepth,int trainFeatureNumPerNode,int minLeafSample,float minInfoGain,bool isRegression)
	:Tree(MaxDepth,trainFeatureNumPerNode,minLeafSample,minInfoGain,isRegression)
{}
ClasTree::~ClasTree()
{}
void ClasTree::train(Sample*sample)
{
	//initialize root node
	//random generate feature index
	int*_featureIndex=new int[_trainFeatureNumPerNode];
    Sample*nodeSample=new Sample(sample,0,sample->getSelectedSampleNum()-1);
	_cartreeArray[0]=new ClasNode();
	_cartreeArray[0]->_samples=nodeSample;
	//calculate the probablity and gini
	_cartreeArray[0]->calculateParams();
	for(int i=0;i<_nodeNum;++i)
	{
		int parentId=(i-1)/2;
		//if current node's parent node is NULL,continue 
		if(_cartreeArray[parentId]==NULL)
		{continue;}
		//if the current node's parent node is a leaf,continue
		if(i>0&&_cartreeArray[parentId]->isLeaf())
		{continue;}
		//if it reach the max depth
		//set current node as a leaf and continue
		if(i*2+1>=_nodeNum)  //if left child node is out of range
		{
			_cartreeArray[i]->createLeaf();
			continue;
		}
		//if current samples in this node is less than the threshold
		//set current node as a leaf and continue
		if(_cartreeArray[i]->_samples->getSelectedSampleNum()<=_minLeafSample)
		{
			_cartreeArray[i]->createLeaf();
			continue;
		}
        _cartreeArray[i]->_samples->randomSelectFeature
        (_featureIndex,sample->getFeatureNum(),_trainFeatureNumPerNode);
		//else calculate the information gain
		_cartreeArray[i]->calculateInfoGain(_cartreeArray,i,_minInfoGain);
		_cartreeArray[i]->_samples->releaseSampleIndex();
	}
	delete[] _featureIndex;
	_featureIndex=NULL;
    delete nodeSample;
}

void ClasTree::createNode(int id,int featureIndex,float threshold)
{
	_cartreeArray[id]=new ClasNode();
	_cartreeArray[id]->setLeaf(false);
	_cartreeArray[id]->setFeatureIndex(featureIndex);
	_cartreeArray[id]->setThreshold(threshold);
}

void ClasTree::createLeaf(int id,float clas,float prob)
{
	_cartreeArray[id]=new ClasNode();
	_cartreeArray[id]->setLeaf(true);
	((ClasNode*)_cartreeArray[id])->setClass(clas);
	((ClasNode*)_cartreeArray[id])->setProb(prob);
}
/************************************************/
//Regression Tree
RegrTree::RegrTree(int MaxDepth,int trainFeatureNumPerNode,int minLeafSample,float minInfoGain,bool isRegression)
	:Tree(MaxDepth,trainFeatureNumPerNode,minLeafSample,minInfoGain,isRegression)
{}
RegrTree::~RegrTree()
{}
void RegrTree::train(Sample*sample)
{
	//initialize root node
	//random generate feature index
	int*_featureIndex=new int[_trainFeatureNumPerNode];
	Sample*nodeSample=new Sample(sample,0,sample->getSelectedSampleNum()-1);
	_cartreeArray[0]=new RegrNode();
	_cartreeArray[0]->_samples=nodeSample;
	//calculate the mean and variance
	_cartreeArray[0]->calculateParams();
	for(int i=0;i<_nodeNum;++i)
	{
		int parentId=(i-1)/2;
		//if current node's parent node is NULL,continue 
		if(_cartreeArray[parentId]==NULL)
		{continue;}
		//if the current node's parent node is a leaf,continue
		if(i>0&&_cartreeArray[parentId]->isLeaf())
		{continue;}
		//if it reach the max depth
		//set current node as a leaf and continue
		if(i*2+1>=_nodeNum)  //if left child node is out of range
		{
			_cartreeArray[i]->createLeaf();
			continue;
		}
		//if current samples in this node is less than the threshold
		//set current node as a leaf and continue
		if(_cartreeArray[i]->_samples->getSelectedSampleNum()<=_minLeafSample)
		{
			_cartreeArray[i]->createLeaf();
			continue;
		}
		_cartreeArray[i]->_samples->randomSelectFeature
		(_featureIndex,sample->getFeatureNum(),_trainFeatureNumPerNode);
		//else calculate the information gain
		_cartreeArray[i]->calculateInfoGain(_cartreeArray,i,_minInfoGain);
		_cartreeArray[i]->_samples->releaseSampleIndex();
	}
	delete[] _featureIndex;
	_featureIndex=NULL;
    delete nodeSample;
}

void RegrTree::createNode(int id,int featureIndex,float threshold)
{
	_cartreeArray[id]=new RegrNode();
	_cartreeArray[id]->setLeaf(false);
	_cartreeArray[id]->setFeatureIndex(featureIndex);
	_cartreeArray[id]->setThreshold(threshold);
}

void RegrTree::createLeaf(int id,float value)
{
	_cartreeArray[id]=new RegrNode();
	_cartreeArray[id]->setLeaf(true);
	((RegrNode*)_cartreeArray[id])->setValue(value);
}
