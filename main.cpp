#include"RandomForest.h"
#include"MnistPreProcess.h"
#include"csv_reader.h"
#include <boost/timer.hpp>
#include <vector>
#include <set>

void toMatrix(std::vector<std::vector<float> > v, float ** m) {
	for(int i = 0; i < v.size(); ++i) {
		m[i] = new float[v[i].size()];
		for(int j = 0; j < v[i].size(); ++j)
		    m[i][j] = v[i][j];
	}
}

int main(int argc, const char * argv[])
{
    std::vector<float> trainlabels_v = CSVReader(argv[2]).getData()[0];
    std::vector<float> testlabels_v = CSVReader(argv[4]).getData()[0];
	std::vector<std::vector<float> >  trainset_v = CSVReader(argv[1]).getData();
    std::vector<std::vector<float> >  testset_v = CSVReader(argv[3]).getData();


    float** trainset = new float *[trainset_v.size()];
    toMatrix(trainset_v, trainset);
    float** testset = new float *[testset_v.size()];
    toMatrix(testset_v, testset);
    float * trainlabels = trainlabels_v.data();
    float * testlabels = testlabels_v.data();

    std::cout << trainset_v.size() << " " << trainlabels_v.size() << " " << testset_v.size() << " " << testlabels_v.size()
              << " " << testset_v[0].size()  << " " << trainset_v[0].size();
    //2. create RandomForest class and set some parameters
    RandomForest randomForest(std::stoi((std::string) argv[5]), std::stoi((std::string) argv[6]), 5, 0.001);

    std::set<float> labels;
    for(int i = 0; i < trainlabels_v.size(); ++i) {
        labels.insert(trainlabels_v[i]);
    }
    for(int i=0; i< testlabels_v.size(); ++i) {
        labels.insert(testlabels_v[i]);
    }
    std::cout << labels.size() << std::endl;
    //3. start to train RandomForest
//	randomForest.train(trainset,trainlabels,TRAIN_NUM,FEATURE,10,true,56);//regression
    boost::timer t;
    randomForest.train(trainset,trainlabels,trainset_v.size(),trainset_v[0].size(), labels.size(), false);//classification
    std::cout << "Training time: " << static_cast<float>(t.elapsed()) << "\n";
    printf("Node count: %.f\n", randomForest.nodeCount());

    //restore model from file and save model to file
    // randomForest.saveModel(argv[5]);
    //	randomForest.readModel("E:\\RandomForest.Model");
    //	RandomForest randomForest("E:\\RandomForest2.Model");

    //predict single sample
//  float resopnse;
//	randomForest.predict(testset[0],resopnse);

    float *resopnses1 = new float[trainlabels_v.size()];
    randomForest.predict(trainset, trainlabels_v.size(), resopnses1);
    float errorRate = 0;
    for (int i = 0; i < trainlabels_v.size(); ++i)
    {
        std::cout << trainlabels_v[i] << " " << resopnses1[i] << std::endl;
        if (resopnses1[i] != trainlabels_v[i])
        {
            errorRate += 1.0f;
        }
        //for regression
        //		float diff=abs(resopnses[i]-testlabels[i]);
        //		errorRate+=diff;
    }
    errorRate /= trainlabels_v.size();
    printf("the training error rate is:%f\n", errorRate);

    //predict a list of samples
    float*resopnses=new float[testlabels_v.size()];
    randomForest.predict(testset,testlabels_v.size(),resopnses);
    errorRate=0;
    for(int i=0;i<testlabels_v.size();++i)
    {
        if(resopnses[i]!=testlabels[i])
        {
            errorRate+=1.0f;
        }
        //for regression
//		float diff=abs(resopnses[i]-testlabels[i]);
//		errorRate+=diff;
    }
    errorRate/=testlabels_v.size();
    printf("the total error rate is:%f\n", errorRate);
//
//    for(int i = 0; i < testlabels_v.size(); ++i) {
//        std::cout << resopnses[i] << testlabels[i] << std::endl;
//    }

	return 0;
};
