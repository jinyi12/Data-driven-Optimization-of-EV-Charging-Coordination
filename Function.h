#include <ilcplex/ilocplex.h>
#include <ilconcert/ilocsvreader.h>
#include <fstream>
#include <iostream>
#include <string>

#include <stdlib.h>

using namespace std;


// Function # 1 : Count Number of Rows from Excel Files
/*=========================================================================================================================*/
inline int Count_Rows(char* DataLocation)
{
	IloEnv Count;

	char * Data_Location = (char *)DataLocation;

	IloCsvReader Generator_Data(Count, Data_Location);
	IloCsvReader::LineIterator  L_Count(Generator_Data); // Assign Line Number as L1 
	++L_Count; // Increment by 1 to ignore headers

	int Number_Rows = 0;

	while (L_Count.ok())
	{
		++L_Count;
		Number_Rows += 1;
	}
	return Number_Rows;

	Count.end();
}





// Function # 2 : Print 8 Data per line.
/*=========================================================================================================================*/
inline void print(IloNumArray List, IloInt Amount)
{
	IloInt Size;
	Size = List.getSize();

	for (int i = 0; i < Size; i++)
	{
		if (i > 0 && i % Amount == 0)
		{
			cout << endl;
		}

		cout << List[i] << "	 ";
	}
}





// Function # 3 : Compare 2 int and return min value.
/*=========================================================================================================================*/
inline IloInt min(IloInt x, IloInt y)
{
	return y ^ ((x ^ y) & -(x < y));
}




