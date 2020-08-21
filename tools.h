#ifndef TOOLS_H
#define TOOLS_H

#include <iostream>
#include <string>
#include "zlib.h"
#include <map>
#include "Eigen/Dense"
#include <vector>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::RowVectorXd;

void print_help();
void parse_command_line(int argc, char **argv, string &ibdfile, string &endMarker, int &numInds,
        string &prefix, double &alpha, double &initN, int &max_iter, int &G, double &minIBD);
void read_endMarker(const string &endMarker, map<int, int> &map1, map<int, int> &map2, vector<double> &chromlens);
void readIBDseg(const string &ibdfile, const map<int, int> &bpmap1, const map<int, int> &bpmap2, 
  VectorXd *bin1, VectorXd *bin2, VectorXd *bin1_midpoint, VectorXd *bin2_midpoint);

VectorXd run_ibdne(VectorXd &bin1, VectorXd &bin2, VectorXd &bin1_midpoint, VectorXd &bin2_midpoint, 
    const vector<double> &chromlens, int G, double initN, int max_iter, double alpha, double minIBD, int numInds);

void write_result(const VectorXd &N, string prefix);

// wrappers for file I/O
template<typename IO_TYPE>
class FileOrGZ {
  public:
    bool open(const char *filename, const char *mode);
    int getline();
    int printf(const char *format, ...);
    int close();

    static const int INIT_SIZE = 1024 * 50;

    // IO_TYPE is either FILE* or gzFile;
    IO_TYPE fp;

    // for I/O:
    char *buf;
    size_t buf_size;
    size_t buf_len;

  private:
    void alloc_buf();
};

template<typename IO_TYPE>
void FileOrGZ<IO_TYPE>::alloc_buf() {
  buf = (char *) malloc(INIT_SIZE);
  if (buf == NULL) {
    fprintf(stderr, "ERROR: out of memory\n");
    exit(1);
  }
  buf_size = INIT_SIZE;
  buf_len = 0;
}

template<> bool FileOrGZ<FILE *>::open(const char *filename, const char *mode);
template<> bool FileOrGZ<gzFile>::open(const char *filename, const char *mode);
template<> int FileOrGZ<FILE *>::getline();
template<> int FileOrGZ<gzFile>::getline();
template<> int FileOrGZ<FILE *>::printf(const char *format, ...);
template<> int FileOrGZ<gzFile>::printf(const char *format, ...);
template<> int FileOrGZ<FILE *>::close();
template<> int FileOrGZ<gzFile>::close();

#endif