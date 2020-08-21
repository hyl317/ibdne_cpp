#include <iostream>
#include "em.h"
#include <stdlib.h>
#include <math.h>
#include "Eigen/Dense"
#include "Eigen/Core"


void updatePosterior(MatrixXd &T1, MatrixXd &T2, const VectorXd &N, 
        const VectorXd &bin1_midpoint, const VectorXd &bin2_midpoint)
{
    int G = N.rows();
    VectorXd aux(G+1);
    aux(0) = 0;
    aux.tail(G) = log((1 - 1.0/(2.0*N.array())));
    RowVectorXd sum_log_prob_not_coalesce(G+1);
    cumsum_eigen_rowvector(aux, sum_log_prob_not_coalesce);
    
    VectorXd alpha1 = bin1_midpoint/50.0;
    double beta1 = 1.0 - 1.0/(2.0*N(G-1));
    VectorXd tmp1 = 1.0 - beta1*exp(-alpha1.array());
    VectorXd last_col_1 = sum_log_prob_not_coalesce(G) + log(1-beta1) - alpha1.array()*(1.0 + G) - log(2500.0) + 
                (1.0*G*G/tmp1.array() + (2.0*G-1.0)/tmp1.array().square() + 2.0/tmp1.array().cube()).log();
    
    VectorXd alpha2 = bin2_midpoint/50.0;
    double beta2 = 1.0 - 1.0/(2.0*N(G-1));
    VectorXd tmp2 = 1.0 - beta2*exp(-alpha2.array());
    VectorXd last_col_2 = sum_log_prob_not_coalesce(G) + log(1-beta2) - alpha2.array()*(1.0 + G) - log(50.0) + 
            (1.0*G/tmp2.array() + 1.0/tmp2.array().square()).log();

    //calculate the rest of the column
    RowVectorXd gen = VectorXd::LinSpaced(G, 1, G);
    RowVectorXd log_g_over_50 = (gen.array()/50.0).log();
    RowVectorXd N_row = N.transpose();
    RowVectorXd log_2_times_N_g = (2.0*N_row).array().log();
    MatrixXd len_times_g_over_50_1 = bin1_midpoint*gen/50.0;
    MatrixXd len_times_g_over_50_2 = bin2_midpoint*gen/50.0;
    MatrixXd block1 = -len_times_g_over_50_1;
    block1.rowwise() += 2*log_g_over_50;
    block1.rowwise() -= log_2_times_N_g;
    block1.rowwise() += sum_log_prob_not_coalesce.head(G);
    MatrixXd block2 = -len_times_g_over_50_2;
    block2.rowwise() += log_g_over_50;
    block2.rowwise() -= log_2_times_N_g;
    block2.rowwise() += sum_log_prob_not_coalesce.head(G);

    T1.leftCols(G) = block1;
    T2.leftCols(G) = block2;
    T1.rightCols(1) = last_col_1;
    T2.rightCols(1) = last_col_2;

    //apply logsumexp for each row to obtain a column vector
    VectorXd normalizing_const1(T1.rows());
    VectorXd normalizing_const2(T2.rows());
    apply_logsumexp(T1, normalizing_const1, 1);
    apply_logsumexp(T2, normalizing_const2, 1);
    T1.colwise() -= normalizing_const1;
    T2.colwise() -= normalizing_const2;
}

// could make these two functions as a template function
void cumsum_eigen_rowvector(const VectorXd &source, RowVectorXd &dest){
  double cum = 0;
  int n = source.rows();
  for(int i = 0; i < n; i++){
    cum += source(i);
    dest(i) = cum;
  }
}

void cumsum_eigen_colvector(const VectorXd &source, VectorXd &dest){
    double cum = 0;
    int n = source.rows();
    for(int i = 0; i < n; i++){
        cum += source(i);
        dest(i) = cum;
    }
}


void apply_logsumexp(const MatrixXd &T, VectorXd &v, int axis){
    if (axis == 1){
        assert(T.rows() == v.rows());
        for(int i = 0; i < T.rows(); i++){
            double cum = T(i, 0);
            for(int j = 1; j < T.cols(); j++){
                double tmp = T(i,j);
                if (tmp > cum){cum = tmp + log1p(exp(cum-tmp));}
                else{cum = cum + log1p(exp(tmp-cum));}
            }
            v(i) = cum;
        }
    }else if(axis == 0){
        assert(T.cols() == v.rows());
        for(int j = 0; j < T.cols(); j++){
            double cum = T(0,j);
            for(int i = 1; i < T.rows(); i++){
                double tmp = T(i,j);
                if (tmp > cum){cum = tmp + log1p(exp(cum-tmp));}
                else{cum = cum + log1p(exp(tmp-cum));}
            }
            v(j) = cum;
        }
    }else{
        fprintf(stderr, "invalid axis in apply_logsumexp");
        exit(1);
    }
}

double updateN(int maxGen, const MatrixXd &T1, const MatrixXd &T2, 
        const VectorXd &bin1, const VectorXd &bin2, 
        const VectorXd &bin1_midpoint, const VectorXd &bin2_midpoint, 
        int n_p, const RowVectorXd &log_term3, VectorXd &N, double minIBD, double alpha, const VectorXd &chr_len_cM)
{
    using namespace cppoptlib;
    VectorXd log_total_len_each_bin1 = bin1.array().log() + bin1_midpoint.array().log();
    VectorXd log_total_len_each_bin2 = bin2.array().log() + bin2_midpoint.array().log();
    MatrixXd T1_copy(T1);
    MatrixXd T2_copy(T2);
    T1_copy.colwise() += log_total_len_each_bin1;
    T2_copy.colwise() += log_total_len_each_bin2;
    VectorXd log_expected_ibd_len_each_gen1(T1.cols());
    VectorXd log_expected_ibd_len_each_gen2(T2.cols());
    apply_logsumexp(T1_copy, log_expected_ibd_len_each_gen1, 0);
    apply_logsumexp(T2_copy, log_expected_ibd_len_each_gen2, 0);
    VectorXd log_total_expected_ibd_len_each_gen(T1.cols());
    for(int i = 0; i < T1.cols(); i++){
        double opr1 = log_expected_ibd_len_each_gen1(i);
        double opr2 = log_expected_ibd_len_each_gen2(i);
        double ret;
        if (opr1 > opr2){ret = opr1 + log1p(exp(opr2-opr1));}
        else{ret = opr2 + log1p(exp(opr1-opr2));}
        log_total_expected_ibd_len_each_gen(i) = ret;
    }


    // Create solver and function object
    BfgsSolver<lossFunc> solver;
    //VectorXd lb = VectorXd::Constant(N.rows(), 1e2);
    //VectorXd ub = VectorXd::Constant(N.rows(), 1e7);
    lossFunc fun(log_total_expected_ibd_len_each_gen, chr_len_cM, log_term3, n_p, minIBD, alpha);
    //for testing only
    //VectorXd grad(N.rows());
    //double loss = fun(N, grad);
    //cout << "loss: " << loss << endl;
    //fun(N, grad);
    //cout << grad.transpose() << endl;
    //exit(0);
    //VectorXd grad_numeric(N.rows());
    //fun.grad_numeric(N, grad_numeric);
    //cout << "numeric gradient: " << grad_numeric.transpose() << endl;
    //exit(0);
    //end testing
    solver.minimize(fun, N);
    return fun.value(N);
}

double log_expectedIBD_beyond_maxGen_given_Ne(const VectorXd &N, const VectorXd &chr_len_cM,
    int n_p, double minIBD)
{
    int G = N.rows();
    double totg = chr_len_cM.sum();
    int num_chr = chr_len_cM.rows();
    double N_past = N(G-1);
    double alpha = log(1.0 - 1.0/(2.0*N_past)) - minIBD/50.0;
    double log_part_A = log(totg) + (G+1.0)*log((2.0*N_past)/(2.0*N_past-1)) + 
            alpha*(G+1) - log(1.0-exp(alpha));
    double D = (1.0*minIBD/50.0)*totg - 1.0*minIBD*minIBD*num_chr/50.0;
    double log_part_B = log(D) + (G+1.0)*log((2.0*N_past)/(2.0*N_past-1.0)) + alpha*(G+1.0) +
            log(1.0+G*(1.0-exp(alpha))) - 2.0*log(1.0-exp(alpha));
    double sum_not_coalesce = (1 - 1.0/(2.0*N.array())).log().sum();
    return log(n_p) + sum_not_coalesce - log(2.0*N_past) + logaddexp(log_part_A, log_part_B);

}

double lossFunc::value(const VectorXd &x){
    // first calculate value of loss evaluated at x
    int G = x.rows();
    VectorXd aux(G+1);
    aux(0) = 0;
    aux.tail(G) = log((1 - 1.0/(2.0*x.array())));
    VectorXd sum_log_prob_not_coalesce(G+1);
    cumsum_eigen_colvector(aux, sum_log_prob_not_coalesce);

    VectorXd log_expectation(G+1);
    VectorXd gen = VectorXd::LinSpaced(G, 1, G);
    log_expectation.head(G) = log(n_p) + sum_log_prob_not_coalesce.head(G).array() - (2.0*x).array().log()
        - minIBD*gen.array()/50.0 + log_term3.transpose().array();
    double log_IBD_beyond_maxGen = log_expectedIBD_beyond_maxGen_given_Ne(x, chr_len_cM, n_p, minIBD);
    log_expectation(G) = log_IBD_beyond_maxGen;
    
    double penalty = alpha*second_diff_sum(x);
    double chi2_stats = (((log_total_expected_ibd_len_each_gen.array().exp() - 
            log_expectation.array().exp()).square())/log_total_expected_ibd_len_each_gen.array().exp()).sum();
    return chi2_stats + penalty;
}


void lossFunc::gradient(const VectorXd& x, VectorXd& grad){
    // now calculate the gradient
    int G = x.rows();
    MatrixXd jacMatrix = MatrixXd::Zero(G+1,G);
    
    //calculate diagonal elements
    RowVectorXd gen = VectorXd::LinSpaced(G, 1, G);
    RowVectorXd aux(G+1);
    aux(0) = 0;
    aux.tail(G) = log((1 - 1.0/(2.0*x.array())));
    RowVectorXd sum_log_prob_not_coalesce(G+1);
    cumsum_eigen_rowvector(aux, sum_log_prob_not_coalesce);
    RowVectorXd log_common_terms(G);
    log_common_terms = log(n_p) + sum_log_prob_not_coalesce.head(G).array() 
            - minIBD*gen.array()/50 + log_term3.array();
    for(int i = 0; i < G; i++){jacMatrix(i,i) = -exp(log_common_terms(i) + log(0.5) - 2.0*log(x(i)));}
    //calculate lower triangular elements
    for (int g = 2; g < G+1; g++){
        jacMatrix.row(g-1).head(g-1) = (log_common_terms(g-1) - log(2.0*x(g-1)) + log(0.5)
            - (1.0 - 1.0/(2.0*x.head(g-1).array())).log() - 2.0*((x.head(g-1).array()).log())).exp();
    }

    // calculate last row of jacMatrix
    double log_IBD_beyond_maxGen = log_expectedIBD_beyond_maxGen_given_Ne(x, chr_len_cM, n_p, minIBD);
    jacMatrix.row(G).head(G-1) = (log_IBD_beyond_maxGen - (1.0 - 1.0/(2.0*x.head(G-1).array())).log()
        + log(0.5) - 2.0*(x.head(G-1).array().log())).exp();
    double epsilon = 1e-6;
    VectorXd x_up(x);
    x_up(G-1) += epsilon;
    VectorXd x_low(x);
    x_low(G-1) -= epsilon;
    double upper = exp(log_expectedIBD_beyond_maxGen_given_Ne(x_up, chr_len_cM, n_p, minIBD));
    double lower = exp(log_expectedIBD_beyond_maxGen_given_Ne(x_low,chr_len_cM, n_p, minIBD));
    jacMatrix(G, G-1) = (upper - lower)/(2.0*epsilon);
    
    //summing up
    VectorXd log_expectation(G+1);
    log_expectation.head(G) = log_common_terms.transpose().array() - (2.0*x.array()).log();
    log_expectation(G) = log_IBD_beyond_maxGen;

    VectorXd chain_part1(G+1);
    chain_part1 = 2.0*(log_expectation.array().exp() - 
        log_total_expected_ibd_len_each_gen.array().exp())/(log_total_expected_ibd_len_each_gen.array().exp());
    //jacMatrix.colwise() *= chain_part1; this is a invalid operation in Eigen; use the following for-loop instead
    for(int c = 0; c < jacMatrix.cols(); c++){
        jacMatrix.col(c) = jacMatrix.col(c).array()*chain_part1.array();
    }
    RowVectorXd chi2_term = jacMatrix.colwise().sum();

    VectorXd N_left2(G), N_left1(G), N_right2(G), N_right1(G), penalty_term(G);
    shift(x, N_left2, -2);
    shift(x, N_left1, -1);
    shift(x, N_right2, 2);
    shift(x, N_right1, 1);
    penalty_term = 12.0*x - 8.0*(N_left1 + N_right1) + 2.0*(N_left2 + N_right2);
    penalty_term(0) = 2*x(0)-4*x(1)+2*x(2);
    penalty_term(1) = 10*x(1)-4*x(0)-8*x(2)+2*x(3);
    penalty_term(G-1) = 2*x(G-1)-4*x(G-2)+2*x(G-3);
    penalty_term(G-2) = 10*x(G-2)-4*x(G-1)-8*x(G-3)+2*x(G-4);
    for(int i = 0; i < G; i++){grad(i) = chi2_term(i) + alpha*penalty_term(i);}

}

// void lossFunc::grad_numeric(const VectorXd &x, VectorXd &grad){
//     double epsilon = 1e-6;
//     // approximate gradient numerically
//     for(int i = 0; i < x.rows(); i++){
//         VectorXd x_up(x);
//         VectorXd x_low(x);
//         x_up(i) += epsilon;
//         x_low(i) -= epsilon;
//         double upper = evaluate(x_up);
//         double lower = evaluate(x_low);
//         grad(i) = (upper - lower)/(2*epsilon);
//     }
// }

double second_diff_sum(const VectorXd &N){
    VectorXd first_diff(N.rows()-1);
    for(int i = 1; i < N.rows(); i++){
        first_diff(i-1) = N(i) - N(i-1);
    }
    VectorXd second_diff(first_diff.rows()-1);
    for(int i = 1; i < first_diff.rows(); i++){
        second_diff(i-1) = first_diff(i) - first_diff(i-1);
    }
    return second_diff.array().square().sum();
}

void shift(const VectorXd &source, VectorXd &dest, int shift, double replace){
    assert(source.rows() == dest.rows());
    for(int i = 0; i < dest.rows(); i++){
        int index = i - shift;
        dest(i) = (index < 0 || index >= source.rows())? replace : source(index);
    }
}