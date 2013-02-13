#include <string>
#include <vector>
#include <iostream>
#include <stdio.h>
#include "EmuPlusPlus.h" // load the emuplusplus header

//using namespace std;
using std::cout;
using std::cin;
using std::vector;
using std::string;

int main (int argc, char** argv){

	if(argc < 2){
		fprintf(stderr, "run with path to statefile as first argument\n");
		return(EXIT_FAILURE);
	}
		
	string filename(argv[1]);

	cout << "# loading emulator from: " << filename << '\n';

	emulator my_emu(filename); // construct the emulator
	
	vector<double> the_point;
	vector<double> the_mean;
	vector<double> the_covariance;
	double dtemp;
	int expected_params = my_emu.getNumberOfParameters();
	int number_of_outputs = my_emu.getNumberOfOutputs();
	int r;
	//int expected_r = 1;
	
	// now read locations in sample space from stdio and loop until eof
	while (cin.good()) {
		//for(int i = 0; (i <  expected_params) && ( r == expected_r);  i++){
		for(int i = 0; (i <  expected_params);  i++){
			cin >> dtemp;
			the_point.push_back(dtemp);
		}
		if (! cin.good())
			break;

		cout << "# the_point: ";
		for(int i = 0; i < expected_params; i++)
			cout << the_point[i] << " "; 
		cout << '\n';

		my_emu.QueryEmulator(the_point, the_mean, the_covariance);

		cout << "# mean: ";
		for(int i = 0; i < the_mean.size(); i++)
			cout << the_mean[i] << " "; 
		cout << '\n';

		cout << "# the_covariance: \n";
		for(int i = 0; i < the_covariance.size(); i++) {
			cout << the_covariance[i]; 
			if ((i + 1) % number_of_outputs == 0)
				cout << '\n'; 
			else
				cout << ' '; 
		}
		cout << '\n';
		
		// now clear up for the next point
		the_point.clear();
		the_mean.clear();
		the_covariance.clear();
	}			
	
	return EXIT_SUCCESS;
}
