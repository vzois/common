#include "ArgParser.h"

int main(int argc, char **argv){

	ArgParser ap;
	ap.parseArgs(argc,argv);

	if(ap.exists(HELP) || ap.count() == 0){
		ap.menu();
		return 0;
	}



	return 0;
}
