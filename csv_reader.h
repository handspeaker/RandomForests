#include <iostream>
#include <fstream>
#include <vector>
#include <iterator>
#include <string>
#include <algorithm>
#include <boost/algorithm/string.hpp>

/*
 * A class to read data from a csv file.
 */
class CSVReader
{
	std::string fileName;
	std::string delimeter;

public:
	CSVReader(std::string filename, std::string delm = ",") :
			fileName(filename), delimeter(delm)
	{ }

	// Function to fetch data from a CSV File
	std::vector<std::vector<std::string> > getData();
};

/*
* Parses through csv file line by line and returns the data
* in vector of vector of strings.
*/
std::vector<std::vector<std::string> > CSVReader::getData()
{
	std::ifstream file(fileName);

	float ** dataList = new float*;

	std::string line = "";
	int i = 0
	// Iterate through each line and split the content using delimeter
	while (getline(file, line))
	{
		dataList[]
		std::vector<std::string> vec;
		float ** data;
		boost::algorithm::split(vec, line, boost::is_any_of(delimeter));
		for(int i = 0; i < vec.size(); ++i) {
			data
		}
		dataList.push_back(vec);
	}
	// Close the File
	file.close();

	return dataList;
}
