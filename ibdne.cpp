#include <string>
#include "tools.h"
#include <map>
#include <vector>
#include <numeric>

using namespace std;

int main(int argc, char **argv){
    if (argc <= 1){
        print_help();
    }

    string ibdfile, endMarker;
    string prefix = "out";
    double alpha = 0.01;
    double initN = 1e4;
    int max_iter = 200;
    int G = 200;
    double minIBD = 2.0;
    int numInds;
    parse_command_line(argc, argv, ibdfile, endMarker, numInds, prefix, alpha, initN, max_iter, G, minIBD);
    map<int, int> bpmap1, bpmap2;
    vector<double> chromlens;
    read_endMarker(endMarker, bpmap1, bpmap2, chromlens);
    fprintf(stdout, "number of chromosomes: %d\n", chromlens.size());
    fprintf(stdout, "total genomic length: %lf\n", 
            accumulate(chromlens.begin(), chromlens.end(), decltype(chromlens)::value_type(0)));
    VectorXd bin1, bin2; 
    VectorXd bin1_midpoint, bin2_midpoint;
    readIBDseg(ibdfile, bpmap1, bpmap2, &bin1, &bin2, &bin1_midpoint, &bin2_midpoint);
    VectorXd finalN = run_ibdne(bin1, bin2, bin1_midpoint, bin2_midpoint, chromlens, 
                G, initN, max_iter, alpha, minIBD, numInds);
    write_result(finalN, prefix);
}