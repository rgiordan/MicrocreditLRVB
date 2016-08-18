
// This is intended to be included in microcredit_model_parameters.h

/////////////////////////////////////
// Offsets

template <typename T, template<typename> class VPType>
Offsets GetOffsets(VPType<T> vp) {
    Offsets offsets;
    int encoded_size = 0;

    offsets.mu = encoded_size;
    encoded_size += vp.mu.encoded_size;

    offsets.lambda = encoded_size;
    encoded_size += vp.lambda.encoded_size;

    // This indicates the start of the local parameters.
    offsets.local_offset = encoded_size;

    offsets.tau.resize(vp.n_g);
    offsets.mu_g.resize(vp.n_g);
    for (int g = 0; g < vp.n_g; g++) {
        offsets.tau[g] = encoded_size;
        encoded_size += vp.tau[g].encoded_size;
        offsets.mu_g[g] = encoded_size;
        encoded_size += vp.mu_g[g].encoded_size;
    }

    // Store the sizes of the parameters.
    if (vp.n_g > 0) {
        offsets.local_encoded_size =
            vp.tau[0].encoded_size + vp.mu_g[0].encoded_size;
    } else {
        offsets.local_encoded_size = 0;
    }
    offsets.encoded_size = encoded_size;

    return offsets;
};


template <typename T> PriorOffsets GetPriorOffsets(PriorParameters<T> pp) {
        PriorOffsets offsets;
        int encoded_size = 0;

        offsets.mu = encoded_size; encoded_size += pp.mu.encoded_size;
        offsets.tau = encoded_size; encoded_size += pp.tau.encoded_size;
        offsets.lambda_eta = encoded_size; encoded_size += 1;
        offsets.lambda_alpha = encoded_size; encoded_size += 1;
        offsets.lambda_beta = encoded_size; encoded_size += 1;

        offsets.encoded_size = encoded_size;

        return offsets;
};





///////////////////////////////
// Converting to and from vectors

template <typename T, template<typename> class VPType>
VectorXT<T> GetParameterVector(VPType<T> vp) {
    VectorXT<T> theta(vp.offsets.encoded_size);

        theta.segment(vp.offsets.mu, vp.mu.encoded_size) =
                vp.mu.encode_vector(vp.unconstrained);

        theta.segment(vp.offsets.lambda, vp.lambda.encoded_size) =
                vp.lambda.encode_vector(vp.unconstrained);

        for (int g = 0; g < vp.tau.size(); g++) {
                theta.segment(vp.offsets.tau[g], vp.tau[g].encoded_size) =
                        vp.tau[g].encode_vector(vp.unconstrained);
        }

        for (int g = 0; g < vp.mu_g.size(); g++) {
                theta.segment(vp.offsets.mu_g[g], vp.mu_g[g].encoded_size) =
                        vp.mu_g[g].encode_vector(vp.unconstrained);
        }

        return theta;
}


template <typename T, template<typename> class VPType>
void SetFromVector(VectorXT<T> const &theta, VPType<T> &vp) {
    if (theta.size() != vp.offsets.encoded_size) {
        throw std::runtime_error("SetFromVector vp: Vector is wrong size.");
    }

    VectorXT<T> theta_sub;

    theta_sub = theta.segment(vp.offsets.mu, vp.mu.encoded_size);
    vp.mu.decode_vector(theta_sub, vp.unconstrained);

    theta_sub = theta.segment(vp.offsets.lambda, vp.lambda.encoded_size);
    vp.lambda.decode_vector(theta_sub, vp.unconstrained);

    for (int g = 0; g < vp.tau.size(); g++) {
        theta_sub = theta.segment(vp.offsets.tau[g], vp.tau[g].encoded_size);
        vp.tau[g].decode_vector(theta_sub, vp.unconstrained);
    }

    for (int g = 0; g < vp.mu_g.size(); g++) {
        theta_sub = theta.segment(vp.offsets.mu_g[g], vp.mu_g[g].encoded_size);
        vp.mu_g[g].decode_vector(theta_sub, vp.unconstrained);
    }
}


template <typename T, template<typename> class VPType>
VPType<T> GetSingleGroupVariationalParameters(VPType<T> vp, int const g) {
    if (g < 0 || g > vp.n_g) {
            throw std::runtime_error("g out of bounds");
    }

    VPType<T> vp_g(vp.k, 1, vp.unconstrained);
    vp_g.mu = vp.mu;
    vp_g.lambda = vp.lambda;
    vp_g.mu_g[0] = vp.mu_g[g];
    vp_g.tau[0] = vp.tau[g];

    return vp_g;
}


// For global parameters and a single group
template <typename T, template<typename> class VPType>
VectorXT<T> GetGroupParameterVector(VPType<T> vp, int const g) {
        if (g < 0 || g > vp.n_g) {
                throw std::runtime_error("g out of bounds");
        }

        VPType<T> vp_g = GetSingleGroupVariationalParameters(vp, g);
        return GetParameterVector(vp_g);
}


template <typename T, template<typename> class VPType>
void SetFromGroupVector(
        VectorXT<T> const &theta, VPType<T> &vp, int const g) {
        if (g < 0 || g > vp.n_g) {
                throw std::runtime_error("g out of bounds");
        }

        VPType<T> vp_g = GetSingleGroupVariationalParameters(vp, g);
        if (theta.size() != vp_g.offsets.encoded_size) {
                throw std::runtime_error("SetFromGroupVector: Vector is wrong size.");
        }

        SetFromVector(theta, vp_g);
        vp.mu = vp_g.mu;
        vp.lambda = vp_g.lambda;
        vp.mu_g[g] = vp_g.mu_g[0];
        vp.tau[g] = vp_g.tau[0];
}


// For global parameters
template <typename T, template<typename> class VPType>
VectorXT<T> GetGlobalParameterVector(VPType<T> vp) {
        int encoded_size = vp.mu.encoded_size + vp.lambda.encoded_size;
        VectorXT<T> theta(encoded_size);

        // Make sure that SetFromGlobalVector and SetFromVector follows the
        // same ordering.
        int offset = 0;
        theta.segment(offset, vp.mu.encoded_size) = vp.mu.encode_vector(vp.unconstrained);
        offset += vp.mu.encoded_size;
        theta.segment(offset, vp.lambda.encoded_size) = vp.lambda.encode_vector(vp.unconstrained);

        return theta;
}


template <typename T, template<typename> class VPType>
void SetFromGlobalVector(VectorXT<T> const &theta, VPType<T> &vp) {

                int encoded_size = vp.mu.encoded_size + vp.lambda.encoded_size;
        if (theta.size() != encoded_size) {
                throw std::runtime_error("SetFromGlobalVector: Vector is wrong size.");
        }

        int offset = 0;
        VectorXT<T> theta_sub;

        // Make sure that GetGlobalParameterVector  and
        // GetParameterVector follows the same ordering.
        theta_sub = theta.segment(offset, vp.mu.encoded_size);
        vp.mu.decode_vector(theta_sub, vp.unconstrained);
        offset += vp.mu.encoded_size;

        theta_sub = theta.segment(offset, vp.lambda.encoded_size);
        vp.lambda.decode_vector(theta_sub, vp.unconstrained);
}


///////////////////
// Priors

template <typename T> VectorXT<T> GetParameterVector(PriorParameters<T> pp) {
    VectorXT<T> theta(pp.offsets.encoded_size);

    theta.segment(pp.offsets.mu, pp.mu.encoded_size) = pp.mu.encode_vector(false);
    theta.segment(pp.offsets.tau, pp.tau.encoded_size) = pp.tau.encode_vector(false);

    theta(pp.offsets.lambda_eta) = pp.lambda_eta;
    theta(pp.offsets.lambda_alpha) = pp.lambda_eta;
    theta(pp.offsets.lambda_beta) = pp.lambda_eta;

    return theta;
}


template <typename T>
void SetFromVector(VectorXT<T> const &theta, PriorParameters<T> &pp) {
    if (theta.size() != pp.offsets.encoded_size) {
            throw std::runtime_error("SetFromVector pp:  is wrong size.");
    }

    VectorXT<T> theta_sub;

    theta_sub = theta.segment(pp.offsets.mu, pp.mu.encoded_size);
    pp.mu.decode_vector(theta_sub, false);

    theta_sub = theta.segment(pp.offsets.tau, pp.tau.encoded_size);
    pp.tau.decode_vector(theta_sub, false);

    pp.lambda_eta = theta(pp.offsets.lambda_eta);
    pp.lambda_alpha = theta(pp.offsets.lambda_alpha);
    pp.lambda_beta = theta(pp.offsets.lambda_beta);
}


/////////////
// Both priors and parameters in a single vector.
template <typename T>
VectorXT<T> GetParameterVector(VariationalParameters<T> &vp,
                               PriorParameters<T> pp) {

    int encoded_size = vp.offsets.encoded_size + pp.offsets.encoded_size;
    VectorXT<T> theta(encoded_size);
    theta.segment(0, vp.offsets.encoded_size) = GetParameterVector(vp);
    theta.segment(vp.offsets.encoded_size, pp.offsets.encoded_size) =
        GetParameterVector(pp);

    return theta;
}


template <typename T>
void SetFromVector(VectorXT<T> const &theta,
                   VariationalParameters<T> &vp, PriorParameters<T> &pp) {

    int encoded_size = vp.offsets.encoded_size + pp.offsets.encoded_size;
    if (theta.size() != encoded_size) {
        throw std::runtime_error("SetFromVector vp and pp: Vector is wrong size.");
    }

    VectorXT<T> theta_sub;

    theta_sub = theta.segment(0, vp.offsets.encoded_size);
    SetFromVector(theta_sub, vp);

    theta_sub = theta.segment(vp.offsets.encoded_size, pp.offsets.encoded_size);
    SetFromVector(theta_sub, pp);
}
