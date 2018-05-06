#include"RandomForest.h"
#include"MnistPreProcess.h"
#include"csv_reader.h"
#include <vector>
#define TRAIN_NUM 60000
#define TEST_NUM 10000
#define FEATURE 784
#define NUMBER_OF_CLASSES 10

int main(int argc, const char * argv[])
{
    //1. prepare data
	float**trainset;
	float** testset;
	float*trainlabels;
	float*testlabels;
	trainset=new float*[TRAIN_NUM];
	testset=new float*[TEST_NUM];
	trainlabels=new float[TRAIN_NUM];
	testlabels=new float[TEST_NUM];
	for(int i=0;i<TRAIN_NUM;++i)
	{trainset[i]=new float[FEATURE];}
	for(int i=0;i<TEST_NUM;++i)
	{testset[i]=new float[FEATURE];}

	std::vector<std::vector<std::string> >  trainset_v = new CSVReader(argv[1]).getData();
	std::vector<std::vector<std::string> > trainlabels_v = new CSVReader(argv[2]).getData();
	std::vector<std::vector<std::string> >  testset_v = new CSVReader(argv[3]).getData();
	std::vector<std::vector<std::string> > testlabels_v = new CSVReader(argv[4]).getData();

    //2. create RandomForest class and set some parameters
	RandomForest randomForest(100,10,10,0);

	//3. start to train RandomForest
//	randomForest.train(trainset,trainlabels,TRAIN_NUM,FEATURE,10,true,56);//regression
    randomForest.train(trainset,trainlabels,TRAIN_NUM,FEATURE,10,false);//classification

    //restore model from file and save model to file
//	randomForest.saveModel("E:\\RandomForest2.Model");
//	randomForest.readModel("E:\\RandomForest.Model");
//	RandomForest randomForest("E:\\RandomForest2.Model");

    //predict single sample
//  float resopnse;
//	randomForest.predict(testset[0],resopnse);

    //predict a list of samples
    float*resopnses=new float[TEST_NUM];
	randomForest.predict(testset,TEST_NUM,resopnses);
	float errorRate=0;
	for(int i=0;i<TEST_NUM;++i)
	{
        if(resopnses[i]!=testlabels[i])
        {
            errorRate+=1.0f;
        }
        //for regression
//		float diff=abs(resopnses[i]-testlabels[i]);
//		errorRate+=diff;
	}
	errorRate/=TEST_NUM;
	printf("the total error rate is:%f\n",errorRate);

	delete[] resopnses;
	for(int i=0;i<TRAIN_NUM;++i)
	{delete[] trainset[i];}
	for(int i=0;i<TEST_NUM;++i)
	{delete[] testset[i];}
	delete[] trainlabels;
	delete[] testlabels;
	delete[] trainset;
	delete[] testset;
	return 0;
};
