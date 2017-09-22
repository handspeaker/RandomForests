#include"Node.h"
/***************************************************************/
//Node
Node::Node()
{
	_isLeaf=false;
	_featureIndex=-1;
	_threshold=0;
	_samples=NULL;
}

Node::~Node()
{
}

void Node::sortIndex(int featureId)
{
	float**data=_samples->_dataset;
	int*sampleId=_samples->getSampleIndex();
	Pair*pairs=new Pair[_samples->getSelectedSampleNum()];
	for(int i=0;i<_samples->getSelectedSampleNum();++i)
	{
		pairs[i].id=sampleId[i];
		pairs[i].feature=data[sampleId[i]][featureId];
	}
	qsort(pairs,_samples->getSelectedSampleNum(),sizeof(Pair),compare_pair);
	for(int i=0;i<_samples->getSelectedSampleNum();++i)
	{sampleId[i]=pairs[i].id;}
	delete[] pairs;
}

int compare_pair( const void* a, const void* b ) 
{
   Pair* arg1 = (Pair*) a;
   Pair* arg2 = (Pair*) b;
   if( arg1->feature < arg2->feature ) return -1;
   else if( arg1->feature == arg2->feature ) return 0;
   else return 1;
}     
/***************************************************************/
//ClasNode
ClasNode::ClasNode()
	:Node()
{
	_class=-1;
	_prob=0;
}

ClasNode::~ClasNode()
{
	if(_probs!=NULL)
	{
		delete[] _probs;
		_probs=NULL;
	}
}

void ClasNode::calculateParams()
{
	int i=0;
	int*sampleId=_samples->getSampleIndex();
	int sampleNum=_samples->getSelectedSampleNum();
	int classNum=_samples->getClassNum();
	float gini=0;
	_probs=new float[classNum];
	for(i=0;i<classNum;++i)
	{_probs[i]=0;}
	for(i=0;i<sampleNum;++i)
	{_probs[static_cast<int>(_samples->_labels[sampleId[i]])]++;}
	for(i=0;i<classNum;++i)
	{
		float p=_probs[i]/sampleNum;
		gini+=(p*p);
	}
	_gini=1-gini;
}

void ClasNode::calculateInfoGain(Node**nodeArray,int id,float minInfoGain)
{
	//some used variables
	int i=0,j=0,k=0;
	int*sampleId=_samples->getSampleIndex();
	int*featureId=_samples->getFeatureIndex();
	float**data=_samples->_dataset;
	float*labels=_samples->_labels;
	int featureNum=_samples->getSelectedFeatureNum();
	int sampleNum=_samples->getSelectedSampleNum();
	int classNum=_samples->getClassNum();
	//the final params need to store
	float maxInfoGain=0;
	int maxFeatureId=0;
	float maxThreshold=0;
	float maxGiniLeft=0;
	float maxGiniRight=0;
	int maxSamplesOnLeft=0;
	float*maxProbsLeft=new float[classNum];
	float*maxProbsRight=new float[classNum];
	for(i=0;i<classNum;++i)
	{
		maxProbsLeft[i]=0;
		maxProbsRight[i]=0;
	}
	//the params need to store in first loop
	float fMaxinfoGain=0;
	int fMaxFeatureId=0;
	float fMaxThreshold=0;
	float fMaxGiniLeft=0;
	float fMaxGiniRight=0;
	int fMaxSamplesOnLeft=0;
	float*fMaxProbsLeft=new float[classNum];
	float*fMaxProbsRight=new float[classNum];
	//the temp params in inner loop
	float giniLeft=0,giniRight=0,infoGain=0;
	float*probsLeft=new float[classNum];
	float*probsRight=new float[classNum];
	for(i=0;i<featureNum;++i)  //for every dimension
	{
		//sort the samples according to the current feature
		//this means only exchange the position of the index
		//in sampleIndex.the trainset and labels never change
		fMaxinfoGain=0;
		fMaxFeatureId=featureId[i];
		fMaxGiniLeft=0;
		fMaxGiniRight=0;
		fMaxThreshold=0;
		fMaxSamplesOnLeft=0;
		for(j=0;j<classNum;++j)
		{
			fMaxProbsLeft[j]=0;
			fMaxProbsRight[j]=0;
		}
        //sort samples by current feature
		sortIndex(featureId[i]);
		//initialize the probsLeft&probsRight
		for(k=0;k<classNum;++k)
		{
			probsLeft[k]=0;
			probsRight[k]=0;
		}
		memcpy(probsRight,_probs,sizeof(float)*classNum);
		for(j=0;j<sampleNum-1;++j)
		{
			giniLeft=0;
			giniRight=0;
			infoGain=0;
			probsLeft[static_cast<int>(labels[sampleId[j]])]++;
			probsRight[static_cast<int>(labels[sampleId[j]])]--;
            //do not do calculation if the nearby samples' feature are too similar(<0.000001)
			if((data[sampleId[j+1]][featureId[i]]-data[sampleId[j]][featureId[i]])<0.000001)
			{continue;}
			for(k=0;k<classNum;++k)
			{
				float p=probsLeft[k]/(j+1);
				giniLeft+=(p*p);
			}
			giniLeft=1-giniLeft;
			for(k=0;k<classNum;++k)
			{
				float p=probsRight[k]/(sampleNum-j-1);
				giniRight+=(p*p);
			}
			giniRight=1-giniRight;
			float leftRatio=(j+1.0)/sampleNum;
			float rightRatio=(sampleNum-j-1.0)/sampleNum;
			infoGain=_gini-leftRatio*giniLeft-rightRatio*giniRight;
			if(infoGain>fMaxinfoGain)
			{
				fMaxinfoGain=infoGain;
				fMaxGiniLeft=giniLeft;
				fMaxGiniRight=giniRight;
				fMaxThreshold=(data[sampleId[j]][featureId[i]]+data[sampleId[j+1]][featureId[i]])/2;
				fMaxSamplesOnLeft=j;
				memcpy(fMaxProbsLeft,probsLeft,sizeof(float)*classNum);
				memcpy(fMaxProbsRight,probsRight,sizeof(float)*classNum);
			}
		}
		if(fMaxinfoGain>maxInfoGain)
		{
			maxInfoGain=fMaxinfoGain;
			maxGiniLeft=fMaxGiniLeft;
			maxGiniRight=fMaxGiniRight;
			maxFeatureId=fMaxFeatureId;
			maxThreshold=fMaxThreshold;
			maxSamplesOnLeft=fMaxSamplesOnLeft;
			memcpy(maxProbsLeft,fMaxProbsLeft,sizeof(float)*classNum);
			memcpy(maxProbsRight,fMaxProbsRight,sizeof(float)*classNum);
		}
	}
	sortIndex(maxFeatureId);
	if(maxInfoGain<minInfoGain)
	{createLeaf();}
	else
	{
		_featureIndex=maxFeatureId;
		_threshold=maxThreshold;
		nodeArray[id*2+1]=new ClasNode();
		nodeArray[id*2+2]=new ClasNode();
		((ClasNode*)nodeArray[id*2+1])->_gini=maxGiniLeft;
		((ClasNode*)nodeArray[id*2+1])->_probs=maxProbsLeft;
		((ClasNode*)nodeArray[id*2+2])->_gini=maxGiniRight;
		((ClasNode*)nodeArray[id*2+2])->_probs=maxProbsRight;
		//assign samples to left and right
		Sample*leftSamples=new Sample(_samples,0,maxSamplesOnLeft);
		Sample*rightSamples=new Sample(_samples,maxSamplesOnLeft+1,sampleNum-1);
		nodeArray[id*2+1]->_samples=leftSamples;
		nodeArray[id*2+2]->_samples=rightSamples;
	}
	delete[] _probs;
	_probs=NULL;
	delete[] fMaxProbsLeft;
	delete[] fMaxProbsRight;
	delete[] probsLeft;
	delete[] probsRight;
}

void ClasNode::createLeaf()
{
	_class=0;
	_prob=_probs[0];
	for(int i=1;i<_samples->getClassNum();++i)
	{
		if(_probs[i]>_prob)
		{
			_class=i;
			_prob=_probs[i];
		}
	}
	_prob/=_samples->getSelectedSampleNum();
	_isLeaf=true;
}

int ClasNode::predict(float*data,int id)
{
	if(data[_featureIndex]<_threshold)
	{return id*2+1;}
	else
	{return id*2+2;}
}

void ClasNode::getResult(Result&r)
{
	r.label=_class;
	r.prob=_prob;
}
/***************************************************************/
//RegrNode
RegrNode::RegrNode()
	:Node()
{
	_value=0;
}

RegrNode::~RegrNode()
{}

void RegrNode::calculateParams()
{
	int i=0;
	int*labelId=_samples->getSampleIndex();
	int sampleNum=_samples->getSelectedSampleNum();
	double mean=0,variance=0;
	for(i=0;i<sampleNum;++i)
	{mean+=_samples->_labels[labelId[i]];}
	mean/=sampleNum;
	for(i=0;i<sampleNum;++i)
	{
		float diff=_samples->_labels[labelId[i]]-mean;
		variance+=diff*diff;
	}
	_mean=mean;
	_variance=variance/sampleNum;
}

void RegrNode::calculateInfoGain(Node**nodeArray,int id,float minInfoGain)
{
	//some used variables
	int i=0,j=0,k=0;
	int*sampleId=_samples->getSampleIndex();
	int*featureId=_samples->getFeatureIndex();
	float**data=_samples->_dataset;
	float*labels=_samples->_labels;
	int featureNum=_samples->getSelectedFeatureNum();
	int sampleNum=_samples->getSelectedSampleNum();
	//the final params need to store
	float maxInfoGain=0;
	int maxFeatureId=0;
	float maxThreshold=0;
	float maxVarLeft=0;
	float maxVarRight=0;
	int maxSamplesOnLeft=0;
	float maxMeanLeft=0;
	float maxMeanRight=0;
	//the params need to store in first loop
	float fMaxinfoGain=0;
	int fMaxFeatureId=0;
	float fMaxThreshold=0;
	float fMaxVarLeft=0;
	float fMaxVarRight=0;
	int fMaxSamplesOnLeft=0;
	float fMaxMeanLeft=0;
	float fMaxMeanRight=0;
	//the temp params in inner loop
	float infoGain=0;
	float varLeft=0,varRight=0;
	float meanLeft=0,meanRight=0;
	for(i=0;i<featureNum;++i)  //for every dimension
	{
		//sort the samples according to the current feature
		//this means only exchange the position of the index
		//in sampleIndex.the trainset and labels never change
		fMaxinfoGain=0;
		fMaxFeatureId=featureId[i];
		fMaxVarLeft=0;
		fMaxVarRight=0;
		fMaxMeanLeft=0;
		fMaxMeanRight=0;
		fMaxThreshold=0;
		fMaxSamplesOnLeft=0;
		//sort the samples by the current selected feature
		sortIndex(featureId[i]);
		//initialize the probsLeft&probsRight
		meanLeft=0;
		meanRight=_mean;
		for(j=0;j<sampleNum-1;++j)
		{
			varLeft=0;
			varRight=0;
			infoGain=0;
			//recalculate the current mean for left and right
			meanLeft=(meanLeft*j+labels[sampleId[j]])/(j+1);
			meanRight=(meanRight*(sampleNum-j)-labels[sampleId[j]])/(sampleNum-j-1);
			//the difference is too tiny,ignore
			if((data[sampleId[j+1]][featureId[i]]-data[sampleId[j]][featureId[i]])<0.000001)
			{continue;}
			for(k=0;k<=j;++k)
			{
				float diff=labels[sampleId[k]]-meanLeft;
				varLeft+=diff*diff;
			}
			varLeft/=(j+1);
			for(k=j+1;k<sampleNum;++k)
			{
				float diff=labels[sampleId[k]]-meanRight;
				varRight+=diff*diff;
			}
			varRight/=(sampleNum-j-1);
			//calculate the infoGain to decide to update
			float leftRatio=(j+1.0)/sampleNum;
			float rightRatio=(sampleNum-j-1.0)/sampleNum;
			infoGain=_variance-leftRatio*varLeft-rightRatio*varRight;
			if(infoGain>fMaxinfoGain)
			{
				fMaxinfoGain=infoGain;
				fMaxVarLeft=varLeft;
				fMaxVarRight=varRight;
				fMaxThreshold=(data[sampleId[j]][featureId[i]]+data[sampleId[j+1]][featureId[i]])/2;
				fMaxSamplesOnLeft=j;
				fMaxMeanLeft=meanLeft;
				fMaxMeanRight=meanRight;
			}
		}
		if(fMaxinfoGain>maxInfoGain)
		{
			maxInfoGain=fMaxinfoGain;
			maxVarLeft=fMaxVarLeft;
			maxVarRight=fMaxVarRight;
			maxFeatureId=fMaxFeatureId;
			maxThreshold=fMaxThreshold;
			maxSamplesOnLeft=fMaxSamplesOnLeft;
			maxMeanLeft=fMaxMeanLeft;
			maxMeanRight=fMaxMeanRight;
		}
	}

	if(maxInfoGain<minInfoGain)
	{createLeaf();}
	else
	{
		//sort the samples so that all the samples 
		//less than the threshold will be on the left
		//and others will be on the right
		sortIndex(maxFeatureId);
		_featureIndex=maxFeatureId;
		_threshold=maxThreshold;
		nodeArray[id*2+1]=new RegrNode();
		nodeArray[id*2+2]=new RegrNode();
		((RegrNode*)nodeArray[id*2+1])->_variance=maxVarLeft;
		((RegrNode*)nodeArray[id*2+1])->_mean=maxMeanLeft;
		((RegrNode*)nodeArray[id*2+2])->_variance=maxVarRight;
		((RegrNode*)nodeArray[id*2+2])->_mean=maxMeanRight;
		//assign samples to left and right
		Sample*leftSamples=new Sample(_samples,0,maxSamplesOnLeft);
		Sample*rightSamples=new Sample(_samples,maxSamplesOnLeft+1,sampleNum-1);
		nodeArray[id*2+1]->_samples=leftSamples;
		nodeArray[id*2+2]->_samples=rightSamples;
	}
}

void RegrNode::createLeaf()
{
	_value=_mean;
	_isLeaf=true;
}

int RegrNode::predict(float*data,int id)
{
	if(data[_featureIndex]<_threshold)
	{return id*2+1;}
	else
	{return id*2+2;}
}

void RegrNode::getResult(Result&r)
{
	r.label=0;
	r.prob=_value;
}
