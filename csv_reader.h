#include <iostream>
#include <fstream>
#include <vector>
#include <iterator>
#include <string>

#include <boost/tokenizer.hpp>
#include <boost/tokenizer.hpp>

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
	std::vector<std::vector<float> > getData();

	std::vector<float> parseRow (const std::vector<std::string> & _row);
};

std::vector<float> CSVReader::parseRow (const std::vector<std::string> & _row)
{
	std::vector<float> dataPoint ;
	dataPoint.reserve(_row.size());

	// Parse the string elements of the vector
	int counter = 0;
	for (std::vector<std::string>::const_iterator it = _row.begin(); it != _row.end(); ++it)
	{
		dataPoint.push_back(atof(it->c_str()));
	}

	return dataPoint;
}

std::vector<std::vector<float> > CSVReader::getData()
{
	std::vector<std::vector<float> > set;

	std::string data(fileName);

	std::ifstream in(data.c_str());
	if (!in.is_open())
	{
		std::cout << "Could not open data set file.";
		throw "Could not open data set file.";
	}

	typedef boost::tokenizer< boost::escaped_list_separator<char> > Tokenizer;

	std::vector< std::string > row;
	std::string line;


	while (std::getline(in,line))
	{

		Tokenizer tok(line);
		row.assign(tok.begin(),tok.end());

		// Do not consider blank line
		if (row.size() == 0) continue;

		set.push_back(parseRow(row));
	}
	return set;
}
