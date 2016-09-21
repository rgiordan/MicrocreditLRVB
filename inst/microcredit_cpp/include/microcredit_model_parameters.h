# ifndef MICROCREDIT_MODEL_PARAMETERS_H
# define MICROCREDIT_MODEL_PARAMETERS_H

# include <Eigen/Dense>
# include <vector>
# include <iostream>

# include "variational_parameters.h"
# include "monte_carlo_parameters.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::MatrixXi;
using Eigen::VectorXi;

using Eigen::Matrix;
using Eigen::Dynamic;

using Eigen::SparseMatrix;

template <typename T> using VectorXT = Eigen::Matrix<T, Dynamic, 1>;
template <typename T> using MatrixXT = Eigen::Matrix<T, Dynamic, Dynamic>;

using std::vector;


/////////////////////////////
// Offset types

struct Offsets {
    int mu;
    int lambda;
    vector<int> tau;
    vector<int> mu_g;

    int local_offset;
    int encoded_size;
    int local_encoded_size;

    Offsets() {
        mu = 0;
        lambda = 0;
        tau.resize(0);
        mu_g.resize(0);
        local_offset = 0;
        local_encoded_size = 0;
        encoded_size = 0;
    }
};


struct PriorOffsets {
    // We will encode all the prior parameters whether we use them or not.
    int mu;
    int mu_t_loc;
    int mu_t_scale;
    int mu_t_df;
    int tau;

    int lambda_eta;
    int lambda_alpha;
    int lambda_beta;

    int encoded_size;

    PriorOffsets() {
        mu = mu_t_loc = mu_t_scale = mu_t_df = 0;
        tau = 0;
        lambda_eta = 0;
        lambda_alpha = 0;
        lambda_beta =0;
    }
};


//////////////////////////////////////
// VariationalParameters
//////////////////////////////////////

template <class T>
class VariationalParameters {
private:
    void Initialize(int _k, int _n_g, int n_sim, bool _unconstrained) {
        k = _k;
        n_g = _n_g;
        unconstrained = _unconstrained;

        mu = MultivariateNormalNatural<T>(k);
        lambda = WishartNatural<T>(k);
        mu.info.scale_cholesky = lambda.v.scale_cholesky = false;
        mu_draws = MonteCarloNormalParameter(n_sim);

        // Per-observation parameters
        mu_g.resize(n_g);
        tau.resize(n_g);

        for (int g = 0; g < n_g; g++) {
            mu_g[g] = MultivariateNormalNatural<T>(k);
            mu_g[g].info.scale_cholesky = false;
            tau[g] = GammaNatural<T>();
            // tau[g].alpha_max = 1e6;
            // tau[g].beta_max = 1e6;
            tau[g].beta_min = 1e-6;
        }
        offsets = GetOffsets(*this);
    }

    void Initialize(int _k, int _n_g, bool _unconstrained) {
        Initialize(_k, _n_g, 20, _unconstrained);
    }

public:
    // Parameters:
    int n_g;    // The number of groups
    int k;      // The dimension of the means

    bool unconstrained;
    Offsets offsets;

    // The global mean parameter.
    MultivariateNormalNatural<T> mu;
    MonteCarloNormalParameter mu_draws;

    // The precision matrix of the grou p means.
    WishartNatural<T> lambda;

    // A vector of per-group, E(tau), the observation noise precision
    vector<GammaNatural<T>> tau;

    // Vectors of the per-group means.
    vector<MultivariateNormalNatural<T>> mu_g;

    // Methods:
    VariationalParameters(int k, int n_g, bool unconstrained):
    k(k), n_g(n_g), unconstrained(unconstrained) {
        Initialize(k, n_g, unconstrained);
    };

    VariationalParameters() {
        Initialize(1, 1, false);
    }

    /////////////////
    template <typename Tnew>
    operator VariationalParameters<Tnew>() const {
        VariationalParameters<Tnew> vp =
                VariationalParameters<Tnew>(k, n_g, unconstrained);

        vp.mu = mu;
        vp.lambda = lambda;
        vp.mu_draws = mu_draws;

        for (int g = 0; g < n_g; g++) {
            vp.mu_g[g] = mu_g[g];
            vp.tau[g] = tau[g];
        }
        return vp;
    }
};


template <class T>
class MomentParameters {
private:
    void Initialize(int _k, int _n_g) {
        k = _k;
        n_g = _n_g;
        unconstrained = true;

        mu = MultivariateNormalMoments<T>(k);
        lambda = WishartMoments<T>(k);

        // Per-observation parameters
        mu_g.resize(n_g);
        tau.resize(n_g);

        for (int g = 0; g < n_g; g++) {
            mu_g[g] = MultivariateNormalMoments<T>(k);
            tau[g] = GammaMoments<T>();
        }
        offsets = GetOffsets(*this);
    }
public:
    // Parameters:
    int n_g;    // The number of groups
    int k;      // The dimension of the means
    bool unconstrained;

    Offsets offsets;

    // The global mean parameter.
    MultivariateNormalMoments<T> mu;

    // The precision matrix of the grou p means.
    WishartMoments<T> lambda;

    // A vector of per-group, E(tau), the observation noise precision
    vector<GammaMoments<T>> tau;

    // Vectors of the per-group means.
    vector<MultivariateNormalMoments<T>> mu_g;

    // Methods:
    MomentParameters(int k, int n_g): k(k), n_g(n_g) {
        Initialize(k, n_g);
    };

    MomentParameters(VariationalParameters<T> vp) {
            Initialize(vp.k, vp.n_g);
            mu = MultivariateNormalMoments<T>(vp.mu);
            lambda = WishartMoments<T>(vp.lambda);
            for (int g = 0; g < n_g; g++) {
                    mu_g[g] = MultivariateNormalMoments<T>(vp.mu_g[g]);
                    tau[g] = GammaMoments<T>(vp.tau[g]);
            }
    };

    MomentParameters() {
        Initialize(1, 1);
    }

    /////////////////
    template <typename Tnew>
    operator MomentParameters<Tnew>() const {
        MomentParameters<Tnew> mp = MomentParameters<Tnew>(k, n_g);

        mp.mu = mu;
        mp.lambda = lambda;

        for (int g = 0; g < n_g; g++) {
            mp.mu_g[g] = mu_g[g];
            mp.tau[g] = tau[g];
        }
        return mp;
    }
};


//////////////////////////////
// Priors

template <class T> class PriorParameters {
private:
    void Initialize(int k) {
        k = k;
        mu_student_t_prior = false;
        mu = MultivariateNormalNatural<T>(k);
        mu_t_loc = 0;
        mu_t_scale = 1;
        mu_t_df = 50;

        lambda_eta = 1;
        lambda_alpha = 1;
        lambda_beta = 1;

        tau = GammaNatural<T>();

        offsets = GetPriorOffsets(*this);
    }
public:
    // Parameters:
    int k;      // The dimension of the means
    PriorOffsets offsets;

    // Mu prior which can be student t or multivariate normal.
    bool mu_student_t_prior;    // If true, use a t prior.  Otherwise use a MVN prior.

    // Mu MVN parameters
    MultivariateNormalNatural<T> mu;

    // Mu student t parameters.  Each location, scale, and df are the same.
    T mu_t_loc;
    T mu_t_scale;
    T mu_t_df;

    GammaNatural<T> tau;

    // lambda ~ LKJ(eta), scale ~ Gamma(alpha, beta)
    T lambda_eta;
    T lambda_alpha;
    T lambda_beta;

    // Methods:
    PriorParameters(int k): k(k) {
        Initialize(k);
    };


    PriorParameters() {
        Initialize(1);
    };


    template <typename Tnew> operator PriorParameters<Tnew>() const {
        PriorParameters<Tnew> pp = PriorParameters<Tnew>(k);

        pp.mu = mu;
        pp.mu_student_t_prior = mu_student_t_prior;
        pp.mu_t_loc = mu_t_loc;
        pp.mu_t_scale = mu_t_scale;
        pp.mu_t_df = mu_t_df;

        pp.lambda_eta = lambda_eta;
        pp.lambda_alpha = lambda_alpha;
        pp.lambda_beta = lambda_beta;

        pp.tau = tau;

        return pp;
    };
};


//////////////////////////////
// Model data

struct MicroCreditData {

    VectorXd y;           // The observations.
    VectorXi y_g;         // The one-indexed group indices.
    MatrixXd x;           // The explanatory variables.

    int n;                // The number of observations.
    int k;                // The dimension of X
    int n_g;              // The number of groups

    // Constructor
    MicroCreditData(MatrixXd x, VectorXd y, VectorXi y_g): x(x), y(y), y_g(y_g) {
        if (x.rows() != y.size() || x.rows() != y_g.size()) {
            throw std::runtime_error("Wrong number of rows");
        }

        int min_g_index, max_g_index;
        int min_g = y_g.minCoeff(&min_g_index);
        int max_g = y_g.maxCoeff(&max_g_index);
        if (min_g < 1) {
            std::ostringstream error_msg;
            error_msg <<
                "Error -- y_g must have integers between 1 and n_groups. " <<
                "Got min(y_g) = "  << min_g << " and max(y_g)  = " << max_g;
            throw std::runtime_error(error_msg.str());
        }

        n = y.size();
        n_g = max_g;
        k = x.cols();
    };

    MicroCreditData() {
        MatrixXd x(1, 1);
        x.setZero();
        VectorXd y(1);
        y.setZero();
        VectorXi y_g(1);
        y_g(0) = 0;
        MicroCreditData(x, y, y_g);
    };

    void print() {
        printf("--------------------------\n");
        std::cout << "X matrix:" << std::endl << x << std::endl << std::endl;
        std::cout << "Y matrix:" << std::endl << y << std::endl << std::endl;
        std::cout << "Y_g matrix:" << std::endl << y_g << std::endl << std::endl;
        printf("--------------------------\n");
    }

};

# include "microcredit_encoders.h"

# endif
