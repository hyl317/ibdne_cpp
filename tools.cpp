#include <string.h>
#include "tools.h"
#include "assert.h"
#include "stdarg.h"
#include <vector>
#include <math.h>
#include "em.h"

void print_help(){
  exit(0);
}

void parse_command_line(int argc, char **argv, string &ibdfile, string &endMarker, int &numInds,
        string &prefix, double &alpha, double &initN, int &max_iter, int &G, double &minIBD){
    int i = 1;
    for(; i < argc; i++){
        char *token = argv[i];
        if (strcmp(token, "-h") == 0 || strcmp(token, "--help") == 0){
            print_help();
        }else if (strcmp(token, "-i") == 0){
            ibdfile = argv[i+1];
            i++;
        }else if (strcmp(token, "-e") == 0){
            endMarker = argv[i+1];
            i++;
        }else if (strcmp(token, "--alpha") == 0){
            alpha = stod(argv[i+1]);
            i++;
        }else if (strcmp(token, "-N") == 0){
            initN = stod(argv[i+1]);
            i++;
        }else if (strcmp(token, "-o") == 0){
            prefix = argv[i+1];
            i++;
        }else if (strcmp(token, "--max-iter") == 0){
            max_iter = atoi(argv[i+1]);
            i++;
        }else if (strcmp(token, "-G") == 0){
            G = atoi(argv[i+1]);
            i++;
        }else if (strcmp(token, "--minIBD") == 0){
            minIBD = stod(argv[i+1]);
            i++;
        }else if (strcmp(token, "-n") == 0){
            numInds = atoi(argv[i+1]);
            i++;
        }else{
            fprintf(stderr, "unrecognized token %s\n", token);
            print_help();
        }
    }
}

double read_endMarker(const string &endMarker, map<int, int> &bpmap1, map<int, int> &bpmap2, 
        vector<double> &chromlens){
  FileOrGZ<FILE *> in;
  bool ret = in.open(endMarker.c_str(), "r");
  if (!ret){
    fprintf(stderr, "cannot open %s\n", endMarker.c_str());
    exit(1);
  }

  map<int, double> dmap1, dmap2; // map chr to their endMarker's genetic pos
  // bpmap maps chr to their endMarker's physical pos
  while(in.getline() >= 0){
    int chr, bp;
    double genPos;
    sscanf(in.buf, "%d %*s %lf %d", &chr, &genPos, &bp);
    if (bpmap1.find(chr) == bpmap1.end()){
      bpmap1.insert(make_pair(chr, bp));
      dmap1.insert(make_pair(chr, genPos));
    }else{
      bpmap2.insert(make_pair(chr, bp));
      dmap2.insert(make_pair(chr, genPos));
    }
  }
  in.close();

  for(auto it = dmap1.begin(); it != dmap1.end(); ++it){
    auto it2 = dmap2.find(it->first);
    if (it2 == dmap2.end()){
      fprintf(stderr, "missing one endmarker for %d\n", it->first);
      exit(1);
    }
    double genPos1 = it->second;
    double genPos2 = it2->second;
    chromlens.push_back(abs(genPos1 - genPos2));
    if (genPos1 > genPos2){
      int tmp = bpmap2[it->first];
      bpmap2[it->first] = bpmap1[it->first];
      bpmap1[it->first] = tmp;
    }
  }
  // by now, bpmap1 maps chr to the first marker's physical pos
  // bpmap2 maps chr to the last marker's physical pos
}

void binning(const vector<double> seglens, VectorXd *bin, VectorXd *bin_midpoint){
  double BIN_SIZE = 0.05;
  double min = *min_element(seglens.begin(), seglens.end());
  double max = *max_element(seglens.begin(), seglens.end());
  int numbins = ceil((max - min)/BIN_SIZE);
  vector<int> tmp_bin;
  vector<double> tmp_bin_midpoint;
  for(int i = 0; i < numbins; i++){
    double bin_low = min + BIN_SIZE*i;
    double bin_high = min + BIN_SIZE*(i+1);
    tmp_bin_midpoint.push_back((bin_low + bin_high)/2);
    auto condition = [=](double seglen){return seglen >= bin_low && seglen < bin_high;};
    tmp_bin.push_back(count_if(seglens.begin(), seglens.end(), condition));
  }

  // remove bins containing zero elements
  int num_nonzeros = count_if(tmp_bin.begin(), tmp_bin.end(), [](int count){return count > 0;});
  *bin = VectorXd(num_nonzeros);
  *bin_midpoint = VectorXd(num_nonzeros);
  int index = 0;
  for(int j = 0; j < numbins; j++){
    if (tmp_bin[j] > 0){
      (*bin)(index) = tmp_bin[j];
      (*bin_midpoint)(index) = tmp_bin_midpoint[j];
      index++;
    }
  }
  assert(bin->sum() == seglens.size());
  assert(index == num_nonzeros);
}

void readIBDseg(const string &ibdfile, const map<int, int> &map1, const map<int, int> &map2,
          VectorXd *bin1, VectorXd *bin2, VectorXd *bin1_midpoint, VectorXd *bin2_midpoint){
  FileOrGZ<gzFile> in;
  bool ret = in.open(ibdfile.c_str(), "r");
  if (!ret){
    fprintf(stderr, "cannot open %s\n", ibdfile.c_str());
    exit(1);
  }

  vector<double> seglen1;
  vector<double> seglen2;
  while(in.getline() >= 0){
    int chr, bp_start, bp_end;
    double seglen;
    sscanf(in.buf, "%*s %*d %*s %*d %d %d %d %lf", &chr, &bp_start, &bp_end, &seglen);
    auto it1 = map1.find(chr);
    auto it2 = map2.find(chr);
    if (it1 == map1.end() || it2 == map2.end()){
      fprintf(stderr, "missing endmarker for chromosome %d\n", chr);
      exit(1);
    }

    if(bp_start == it1->second || bp_end == it2->second){
      seglen2.push_back(seglen);
    }else{seglen1.push_back(seglen);}
  }
  in.close();

  fprintf(stdout, "total number of segments: %d\n", seglen1.size() + seglen2.size());
  fprintf(stdout, "number of segments extending to chromosome end: %d\n", seglen2.size());

  // binning
  binning(seglen1, bin1, bin1_midpoint);
  binning(seglen2, bin2, bin2_midpoint);
}

VectorXd run_ibdne(VectorXd &bin1, VectorXd &bin2, VectorXd &bin1_midpoint, VectorXd &bin2_midpoint, 
    const vector<double> &chromlens, int G, double initN, int max_iter, double alpha, double minIBD, int numInds)
{
  using namespace Eigen;
  VectorXd N = VectorXd::Constant(G, initN) + 100*VectorXd::Random(G);
  cout << "initN: " << N.transpose() << endl;

  //pre-calculate log of term3 in the updateN step
  //this quantity is a constant in all iterations
  //so is more efficient to calculate here, save as a local variable, and pass it onto future iteration
  int numChroms = chromlens.size();
  int n_p = 2*numInds*(2*numInds - 2)/2;
  MatrixXd chr_len(chromlens.size(), 1); // effectively a column vector
  for(int chr = 0; chr < chromlens.size(); chr++){chr_len(chr, 0) = chromlens[chr];}
  MatrixXd gen(1, G); //effective a row vector
  for(int g = 1; g < G+1; g++){gen(0, g-1) = g;}
  MatrixXd tmp1 = chr_len*gen;
  //cout << tmp1.topLeftCorner(10, 10) << endl;
  tmp1 = minIBD*tmp1/50.0;
  VectorXd chr_len_v = VectorXd(chromlens.size());
  for(int i = 0; i < chromlens.size(); i++){chr_len_v(i) = chromlens[i];}
  tmp1.colwise() += chr_len_v;
  RowVectorXd gen_v = VectorXd::LinSpaced(G, 1, G);
  tmp1.rowwise() += minIBD*minIBD*gen_v/50.0;
  RowVectorXd log_term3 = tmp1.colwise().sum().array().log();

  MatrixXd T1(bin1.rows(), G+1);
  MatrixXd T2(bin2.rows(), G+1);
  updatePosterior(T1, T2, N, bin1_midpoint, bin2_midpoint);
  double fun = updateN(G, T1, T2, bin1, bin2, bin1_midpoint, bin2_midpoint, 
        n_p, log_term3, N, minIBD, alpha, chr_len_v);
  int iter = 1;
  double diff = numeric_limits<double>::infinity();
  while (diff > 1e-3 && iter < max_iter){
    cout << "iteration " << iter << " done: " << N.transpose() << endl;
    updatePosterior(T1, T2, N, bin1_midpoint, bin2_midpoint);
    double fun1 = updateN(G, T1, T2, bin1, bin2, bin1_midpoint, bin2_midpoint, n_p, 
        log_term3, N, minIBD, alpha, chr_len_v);
    diff = abs(fun1 - fun);
    fun = fun1;
    iter++;
  }
  return N;
}


// specialization for FILE I/O wrapper
// open <filename> using standard FILE *
template<>
bool FileOrGZ<FILE *>::open(const char *filename, const char *mode) {
  // First allocate a buffer for I/O:
  alloc_buf();

  fp = fopen(filename, mode);
  if (!fp)
    return false;
  else
    return true;
}

// open <filename> as a gzipped file
template<>
bool FileOrGZ<gzFile>::open(const char *filename, const char *mode) {
  // First allocate a buffer for I/O:
  alloc_buf();

  fp = gzopen(filename, mode);
  if (!fp)
    return false;
  else
    return true;
}

template<>
int FileOrGZ<FILE *>::getline() {
  return ::getline(&buf, &buf_size, fp);
}

template<>
int FileOrGZ<gzFile>::getline() {
  int n_read = 0;
  int c;

  while ((c = gzgetc(fp)) != EOF) {
    // About to have read one more, so n_read + 1 needs to be less than *n.
    // Note that we use >= not > since we need one more space for '\0'
    if (n_read + 1 >= (int) buf_size) {
      const size_t GROW = 1024;
      char *tmp_buf = (char *) realloc(buf, buf_size + GROW);
      if (tmp_buf == NULL) {
	fprintf(stderr, "ERROR: out of memory!\n");
	exit(1);
      }
      buf_size += GROW;
      buf = tmp_buf;
    }
    buf[n_read] = (char) c;
    n_read++;
    if (c == '\n')
      break;
  }

  if (c == EOF && n_read == 0)
    return -1;

  buf[n_read] = '\0';

  return n_read;
}

template<>
int FileOrGZ<FILE *>::printf(const char *format, ...) {
  va_list args;
  int ret;
  va_start(args, format);

  ret = vfprintf(fp, format, args);

  va_end(args);
  return ret;
}

template<>
int FileOrGZ<gzFile>::printf(const char *format, ...) {
  va_list args;
  int ret;
  va_start(args, format);

  // NOTE: one can get automatic parallelization (in a second thread) for
  // gzipped output by opening a pipe to gzip (or bgzip). For example:
  //FILE *pipe = popen("gzip > output.vcf.gz", "w");
  // Can then fprintf(pipe, ...) as if it were a normal file.

  // gzvprintf() is slower than the code below that buffers the output.
  // Saw 13.4% speedup for processing a truncated VCF with ~50k lines and
  // 8955 samples.
  //  ret = gzvprintf(fp, format, args);
  ret = vsnprintf(buf + buf_len, buf_size - buf_len, format, args);
  if (ret < 0) {
    printf("ERROR: could not print\n");
    perror("printf");
    exit(10);
  }

  if (buf_len + ret > buf_size - 1) {
    // didn't fit the text in buf
    // first print what was in buf before the vsnprintf() call:
    gzwrite(fp, buf, buf_len);
    buf_len = 0;
    // now ensure that redoing vsnprintf() will fit in buf:
    if ((size_t) ret > buf_size - 1) {
      do { // find the buffer size that fits the last vsnprintf() call
	buf_size += INIT_SIZE;
      } while ((size_t) ret > buf_size - 1);
      free(buf);
      buf = (char *) malloc(buf_size);
      if (buf == NULL) {
	printf("ERROR: out of memory");
	exit(5);
      }
    }
    // redo:
    ret = vsnprintf(buf + buf_len, buf_size - buf_len, format, args);
  }

  buf_len += ret;
  if (buf_len >= buf_size - 1024) { // within a tolerance of MAX_BUF?
    // flush:
    gzwrite(fp, buf, buf_len);
    buf_len = 0;
  }

  va_end(args);
  return ret;
}

template<>
int FileOrGZ<FILE *>::close() {
  assert(buf_len == 0);
  // should free buf, but I know the program is about to end, so won't
  return fclose(fp);
}

template<>
int FileOrGZ<gzFile>::close() {
  if (buf_len > 0)
    gzwrite(fp, buf, buf_len);
  // should free buf, but I know the program is about to end, so won't
  return gzclose(fp);
}