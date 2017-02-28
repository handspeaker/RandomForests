/************************************************
 *Random Forest Program
 *Function:		this class is used to store the selected index
                of the samples and features.
 *Author:		handspeaker@163.com
 *CreateTime:	2014.7.10
 *Version:		V0.1
 *************************************************/
#ifndef SAMPLE_H
#define SAMPLE_H

#include<stdio.h>
#include<stdlib.h>
#include<string.h>

class Sample
{
public:
	static const int SAMPLESELECTION=1;
	static const int FEATURESELECTION=2;

	//create a empty samples
	Sample(float**dataset,float*labels,int classNum,int sampleNum,int featureNum);
	//copy infomation from samples
	Sample(Sample* samples);
	//copy a part[start,end] from samples
	Sample(Sample* samples,int start,int end);
	~Sample();
	//random select samples with replacement
	void randomSelectSample(int*sampleIndex,int SampleNum,int selectedSampleNum);
	//random select features without replacement
	void randomSelectFeature(int*featureIndex,int featureNum,int selectedFeatureNum);
	
	inline int getClassNum(){return _classNum;};

	inline int getSampleNum(){return _sampleNum;};
	inline int getFeatureNum(){return _featureNum;};

	inline int getSelectedSampleNum(){return _selectedSampleNum;};
	inline int getSelectedFeatureNum(){return _selectedFeatureNum;};

	inline int*getSampleIndex(){return _sampleIndex;};
	inline int*getFeatureIndex(){return _featureIndex;};

	inline void releaseSampleIndex()
	{
		if(_sampleIndex!=NULL)
		{
			delete[] _sampleIndex;
			_sampleIndex=NULL;
		}
	};


	float**_dataset; //pointer to the input dataset
	float*_labels;  //pointer to the input labels
	
private:
	int*_sampleIndex;  //all sample index
	int*_featureIndex; //all feature index
	int _classNum;  //class number
	int _featureNum;  //all feature dimension
	int _sampleNum;  //all sample number
	int _selectedSampleNum; //selected sample number
	int _selectedFeatureNum;//selected feature number
};
#endif//SAMPLE_H
