#include "et_gm_phd_cv_pose.hpp"

#include <random>

namespace mot {
  std::random_device r;
  std::default_random_engine e(r());

  std::uniform_real_distribution<double> pose_dist(-10.0, 10.0);
  std::uniform_real_distribution<double> velocity_dist(-1.0, 1.0);

  EtGmPhdCvPose::EtGmPhdCvPose(const GmPhdCalibrations<4u, 2u> & calibrations)
    : EtGmPhd<4u, 2u>(calibrations) {
    PredictBirths();
  }

  EtGmPhdCvPose::Hypothesis EtGmPhdCvPose::PredictHypothesis(const Hypothesis & hypothesis) {
    static Hypothesis predicted_hypothesis;

    predicted_hypothesis.weight = calibrations_.ps * hypothesis.weight;
    predicted_hypothesis.state = transition_matrix_ * hypothesis.state;
    predicted_hypothesis.covariance = transition_matrix_ * hypothesis.covariance * transition_matrix_.transpose()
      + time_delta * process_noise_covariance_matrix_;

    return predicted_hypothesis;
  }

  void EtGmPhdCvPose::PrepareTransitionMatrix(void) {
    transition_matrix_ = StateSizeMatrix::Zero();

    transition_matrix_(0u, 0u) = 1.0;
    transition_matrix_(0u, 2u) = time_delta;

    transition_matrix_(1u, 1u) = 1.0;
    transition_matrix_(1u, 3u) = time_delta;

    transition_matrix_(2u, 2u) = 1.0;

    transition_matrix_(3u, 3u) = 1.0;
  }

  void EtGmPhdCvPose::PrepareProcessNoiseMatrix(void) {
    process_noise_covariance_matrix_ = StateSizeMatrix::Zero();
    for (size_t i = 0; i < 4; ++i)
      process_noise_covariance_matrix_(i, i) = calibrations_.process_noise_diagonal(i);
  }

} // namespace mot
