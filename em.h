#ifndef EM_H
#define EM_H

#include "Eigen/Dense"

using namespace Eigen;
using namespace std;

void updatePosterior(MatrixXd &T1, MatrixXd &T2, const VectorXd &N, 
        const VectorXd &bin1_midpoint, const VectorXd &bin2_midpoint);

void cumsum_eigen_rowvector(const VectorXd &source, RowVectorXd &dest);
void cumsum_eigen_colvector(const VectorXd &source, VectorXd &dest);
void apply_logsumexp(const MatrixXd &T, VectorXd &v, int axis);
double updateN(int maxGen, const MatrixXd &T1, const MatrixXd &T2, 
        const VectorXd &bin1, const VectorXd &bin2, 
        const VectorXd &bin1_midpoint, const VectorXd &bin2_midpoint, 
        int n_p, const RowVectorXd &log_term3, VectorXd &N, double minIBD, double alpha, const VectorXd &chr_len_cM);
double log_expectedIBD_beyond_maxGen_given_Ne(const VectorXd &N, const VectorXd &chr_len_cM, int n_p, double minIBD);
inline double logaddexp(double d1, double d2){
    if (d1 > d2){return d1 + log1p(exp(d2-d1));}
    else{return d2 + log1p(exp(d1-d2));}
}

double second_diff_sum(const VectorXd &N);
void shift(const VectorXd &source, VectorXd &dest, int shift, double replace=0.0);

class lossFunc
{
private:
    VectorXd log_total_expected_ibd_len_each_gen;
    VectorXd chr_len_cM;
    RowVectorXd log_term3;
    int n_p;
    double minIBD;
    double alpha;
public:
    lossFunc(VectorXd log_total_expected_ibd_len_each_gen_, VectorXd chr_len_cM_,
        RowVectorXd log_term3_, int n_p_, double minIBD_, double alpha_) : 
        log_total_expected_ibd_len_each_gen(log_total_expected_ibd_len_each_gen_),
        chr_len_cM(chr_len_cM_), log_term3(log_term3_), n_p(n_p_), minIBD(minIBD_), alpha(alpha_) {}
    double evaluate(const VectorXd &x);
    void grad_numeric(const VectorXd &x, VectorXd &grad);
    double operator()(const VectorXd &x, VectorXd &grad);
};


#endif