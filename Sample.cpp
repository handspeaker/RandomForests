#include"Sample.h"

Sample::Sample(float**dataset,float*labels,int classNum,int sampleNum,int featureNum)
{
	_dataset=dataset;
	_labels=labels;
	_sampleNum=sampleNum;
	_featureNum=featureNum;
	_classNum=classNum;
	_selectedSampleNum=sampleNum;
	_selectedFeatureNum=featureNum;
	_sampleIndex=NULL;
	_featureIndex=NULL;
}

Sample::Sample(Sample* samples)
{
	_dataset=samples->_dataset;
	_labels=samples->_labels;
	_classNum=samples->getClassNum();
	_featureNum=samples->getFeatureNum();
	_sampleNum=samples->getSampleNum();
	_selectedSampleNum=samples->getSelectedSampleNum();
	_selectedFeatureNum=samples->getSelectedFeatureNum();
	_sampleIndex=samples->getSampleIndex();
	_featureIndex=samples->getFeatureIndex();
}

Sample::Sample(Sample* samples,int start,int end)
{
	_dataset=samples->_dataset;
	_labels=samples->_labels;
	_classNum=samples->getClassNum();
	_sampleNum=samples->getSampleNum();
	_selectedSampleNum=end-start+1;
	_featureNum=samples->getFeatureNum();
	_selectedFeatureNum=samples->getSelectedFeatureNum();
	_sampleIndex=new int[_selectedSampleNum];
	memcpy(_sampleIndex,samples->getSampleIndex()+start,sizeof(float)*_selectedSampleNum);
}

Sample::~Sample()
{
	_sampleIndex=NULL;
	_featureIndex=NULL;
}

void Sample::randomSelectSample(int*sampleIndex,int SampleNum,int selectedSampleNum)
{
	_sampleNum=SampleNum;
	_selectedSampleNum=selectedSampleNum;
	if(_sampleIndex!=NULL)
	{delete[] _sampleIndex;}
	_sampleIndex=sampleIndex;
	int i=0,index=0;
	//sampling trainset with replacement
	for(i=0;i<selectedSampleNum;++i)
	{
		index=rand()%SampleNum;
		_sampleIndex[i]=index;
	}
}
void Sample::randomSelectFeature(int*featureIndex,int featureNum,int selectedFeatureNum)
{
	_featureNum=featureNum;
	_selectedFeatureNum=selectedFeatureNum;
	_featureIndex=featureIndex;
	int i=0,j=0,k=0,index=0;
	//sampling feature without replacement
	for(i=0,j=featureNum-selectedFeatureNum;j<featureNum;++j,++i)
	{
        if(j == 0)
            index = 0;
        else
            index = rand()%j;
		bool flag=false;
		for(k=0;k<i;++k)
		{
			if(_featureIndex[k]==index)
			{
				flag=true;
				break;
			}
		}
		if(flag)
		{
			_featureIndex[i]=j;	
		}
		else
		{
			_featureIndex[i]=index;	
		}
	}
}
