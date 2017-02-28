#include"MnistPreProcess.h"

void readData(float** dataset,float*labels,const char* dataPath,const char*labelPath)
{
	FILE* dataFile=fopen(dataPath,"rb");
	FILE* labelFile=fopen(labelPath,"rb");
	int mbs=0,number=0,col=0,row=0;
	fread(&mbs,4,1,dataFile);
	fread(&number,4,1,dataFile);
	fread(&row,4,1,dataFile);
	fread(&col,4,1,dataFile);
	revertInt(mbs);
	revertInt(number);
	revertInt(row);
	revertInt(col);
	fread(&mbs,4,1,labelFile);
	fread(&number,4,1,labelFile);
	revertInt(mbs);
	revertInt(number);
	unsigned char temp;
	for(int i=0;i<number;++i)
	{
		for(int j=0;j<row*col;++j)
		{
			fread(&temp,1,1,dataFile);
			dataset[i][j]=static_cast<float>(temp);
		}
		fread(&temp,1,1,labelFile);
		labels[i]=static_cast<float>(temp);
	}
	fclose(dataFile);
	fclose(labelFile);
};
