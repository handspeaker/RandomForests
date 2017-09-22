#include"RandomForest.h"

RandomForest::RandomForest(int treeNum,int maxDepth,int minLeafSample,float minInfoGain)
{
	_treeNum=treeNum;
	_maxDepth=maxDepth;
	_minLeafSample=minLeafSample;
	_minInfoGain=minInfoGain;
	_trainSample=NULL;
	printf("total tree number:%d\n",_treeNum);
	printf("max depth of a single tree:%d\n",_maxDepth);
	printf("the minimum samples in a leaf:%d\n",_minLeafSample);
	printf("the minimum information gain:%f\n",_minInfoGain);
    
	_forest=new Tree*[_treeNum];
	for(int i=0;i<_treeNum;++i)
	{_forest[i]=NULL;}
}

RandomForest::RandomForest(const char*modelPath)
{
	readModel(modelPath);
}

RandomForest::~RandomForest()
{
	//printf("destroy RandomForest...\n");
	if(_forest!=NULL)
	{
		for(int i=0;i<_treeNum;++i)
		{
			if(_forest[i]!=NULL)
			{
				delete _forest[i];
				_forest[i]=NULL;
			}
		}
		delete[] _forest;
		_forest=NULL;
	}
	if(_trainSample!=NULL)
	{
		delete _trainSample;
		_trainSample=NULL;
	}
}

void RandomForest::train(float**trainset,float*labels,int SampleNum,int featureNum,
			   int classNum,bool isRegression)
{
	int trainFeatureNumPerNode=static_cast<int>(sqrt(static_cast<float>(featureNum)));
	train(trainset,labels,SampleNum,featureNum,classNum,isRegression,trainFeatureNumPerNode);
}

void RandomForest::train(float**trainset,float*labels,int SampleNum,int featureNum,
			   int classNum,bool isRegression,int trainFeatureNumPerNode)
{
	if(_treeNum<1)
	{
		printf("total tree number must bigger than 0!\n");
		printf("training failed\n");
		return;
	}
	if(_maxDepth<1)
	{
		printf("the max depth must bigger than 0!\n");
		printf("training failed\n");
		return;
	}
	if(_minLeafSample<2)
	{
		printf("the minimum samples in a leaf must bigger than 1!\n");
		printf("training failed\n");
		return;
	}
	_trainSampleNum=SampleNum;
	_featureNum=featureNum;
	_classNum=classNum;
	_trainFeatureNumPerNode=trainFeatureNumPerNode;
	_isRegression=isRegression;
	//initialize every tree
	if(_isRegression)
	{
		_classNum=1;
		for(int i=0;i<_treeNum;++i)
		{
			_forest[i]=new RegrTree(_maxDepth,_trainFeatureNumPerNode,
				_minLeafSample,_minInfoGain,_isRegression);
		}
	}
	else
	{
		for(int i=0;i<_treeNum;++i)
		{
			_forest[i]=new ClasTree(_maxDepth,_trainFeatureNumPerNode,
				_minLeafSample,_minInfoGain,_isRegression);
		}
	}
	//this object hold the whole trainset&labels
	_trainSample=new Sample(trainset,labels,_classNum,_trainSampleNum,_featureNum);
	srand(static_cast<unsigned int>(time(NULL)));
	int*_sampleIndex=new int[_trainSampleNum];
	//start to train every tree in the forest
	for(int i=0;i<_treeNum;++i)
	{
		printf("train the %d th tree...\n",i);
		//random sampling from trainset
		Sample*sample=new Sample(_trainSample);
		sample->randomSelectSample(_sampleIndex,_trainSampleNum,_trainSampleNum);
		_forest[i]->train(sample);
        delete sample;
	}
	delete[] _sampleIndex;
	_sampleIndex=NULL;
}

void RandomForest::predict(float*data,float&response)
{
	//get the predict from every tree
	//if regression,_classNum=1
	float*result=new float[_classNum];
	int i=0;
	for(i=0;i<_classNum;++i)
	{result[i]=0;}
	for(i=0;i<_treeNum;++i)//_treeNum
	{
		Result r;
		r.label=0;
		r.prob=0;//Result 
		r=_forest[i]->predict(data);
		result[static_cast<int>(r.label)]+=r.prob;
	}
	if(_isRegression)
	{response=result[0]/_treeNum;}
	else
	{
		float maxProbLabel=0;
		float maxProb=result[0];
		for(i=1;i<_classNum;++i)
		{
			if(result[i]>maxProb)
			{
				maxProbLabel=i;
				maxProb=result[i];
			}
		}
		response=maxProbLabel;
	}
	delete[] result;
}

void RandomForest::predict(float**testset,int SampleNum,float*responses)
{
	//get the predict from every tree
	for(int i=0;i<SampleNum;++i)
	{
		predict(testset[i],responses[i]);
	}
}

void RandomForest::saveModel(const char*path)
{
	FILE* saveFile=fopen(path,"wb");
	fwrite(&_treeNum,sizeof(int),1,saveFile);
	fwrite(&_maxDepth,sizeof(int),1,saveFile);
	fwrite(&_classNum,sizeof(int),1,saveFile);
	fwrite(&_isRegression,sizeof(bool),1,saveFile);
	int nodeNum=static_cast<int>(pow(2.0,_maxDepth)-1);
	int isLeaf=0;
	for(int i=0;i<_treeNum;++i)
	{
		Node**arr=_forest[i]->getTreeArray();
		isLeaf=0;
		for(int j=0;j<nodeNum;++j)
		{
			if(arr[j]!=NULL)
			{
				if(arr[j]->isLeaf())
				{
					isLeaf=1;
					fwrite(&isLeaf,sizeof(int),1,saveFile);
					if(_isRegression)
					{
						float value=((RegrNode*)arr[j])->getValue();
						fwrite(&value,sizeof(float),1,saveFile);
					}
					else
					{
						float clas=((ClasNode*)arr[j])->getClass();
						float prob=((ClasNode*)arr[j])->getProb();
						fwrite(&clas,sizeof(float),1,saveFile);
						fwrite(&prob,sizeof(float),1,saveFile);
					}
				}
				else
				{
					isLeaf=0;
					fwrite(&isLeaf,sizeof(int),1,saveFile);
					int featureIndex=arr[j]->getFeatureIndex();
					float threshold=arr[j]->getThreshold();
					fwrite(&featureIndex,sizeof(int),1,saveFile);
					fwrite(&threshold,sizeof(float),1,saveFile);
				}
			}
		}
		////write an numb node to denote the tree end
		//isLeaf=-1;
		//fwrite(&isLeaf,sizeof(int),1,saveFile);
	}
	fclose(saveFile);
}

void RandomForest::readModel(const char*path)
{
	_minLeafSample=0;
	_minInfoGain=0;
	_trainFeatureNumPerNode=0;
	FILE* modelFile=fopen(path,"rb");
	fread(&_treeNum,sizeof(int),1,modelFile);
	fread(&_maxDepth,sizeof(int),1,modelFile);
	fread(&_classNum,sizeof(int),1,modelFile);
	fread(&_isRegression,sizeof(bool),1,modelFile);
	int nodeNum=static_cast<int>(pow(2.0,_maxDepth)-1);
	_trainSample=NULL;
	printf("total tree number:%d\n",_treeNum);
	printf("max depth of a single tree:%d\n",_maxDepth);
	printf("_classNum:%d\n",_classNum);
	printf("_isRegression:%d\n",_isRegression);
	_forest=new Tree*[_treeNum];
	//initialize every tree
	if(_isRegression)
	{
		for(int i=0;i<_treeNum;++i)
		{
			_forest[i]=new RegrTree(_maxDepth,_trainFeatureNumPerNode,
			_minLeafSample,_minInfoGain,_isRegression);
		}
	}
	else
	{
		for(int i=0;i<_treeNum;++i)
		{
			_forest[i]=new ClasTree(_maxDepth,_trainFeatureNumPerNode,
				_minLeafSample,_minInfoGain,_isRegression);
		}
	}
	int*nodeTable=new int[nodeNum];
	int isLeaf=-1;
	int featureIndex=0;
	float threshold=0;
	float value=0;
	float clas=0;
	float prob=0;
	for(int i=0;i<_treeNum;++i)
	{
		memset(nodeTable,0,sizeof(int)*nodeNum);
		nodeTable[0]=1;
		for(int j=0;j<nodeNum;j++)
		{
			//if current node is marked as null,continue
			if(nodeTable[j]==0)
			{continue;}
			fread(&isLeaf,sizeof(int),1,modelFile);
			if(isLeaf==0)  //split node
			{
				nodeTable[j*2+1]=1;
				nodeTable[j*2+2]=1;
				fread(&featureIndex,sizeof(int),1,modelFile);
				fread(&threshold,sizeof(float),1,modelFile);
				_forest[i]->createNode(j,featureIndex,threshold);
			}
			else if(isLeaf==1)  //leaf
			{
				if(_isRegression)
				{
					fread(&value,sizeof(float),1,modelFile);
					((RegrTree*)_forest[i])->createLeaf(j,value);
				}
				else
				{
					fread(&clas,sizeof(float),1,modelFile);
					fread(&prob,sizeof(float),1,modelFile);
					((ClasTree*)_forest[i])->createLeaf(j,clas,prob);
				}
			}
		}
		//fread(&isLeaf,sizeof(int),1,modelFile);
	}
	fclose(modelFile);
	delete[] nodeTable;
}
