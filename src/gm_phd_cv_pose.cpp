#include "ghm_phd_cv_pose.hpp"

#include <random>

namespace mot {
  std::random_device r;
  std::default_random_engine e(r());

  std::uniform_real_distribution<double> pose_dist(-10.0, 10.0);
  std::uniform_real_distribution<double> velocity_dist(-1.0, 1.0);

  GmPhdCvPose::GmPhdCvPose(const GmPhdCalibrations<4u, 2u> & calibrations)
    : GmPhd<4u, 2u>(calibrations) {}

  GmPhdCvPose::Hypothesis GmPhdCvPose::PredictHypothesis(const Hypothesis & hypothesis) {
    static Hypothesis predicted_hypothesis;

    predicted_hypothesis.weight = calibrations_.pd * hypothesis.weight;
    predicted_hypothesis.state = transition_matrix_ * hypothesis.state;
    predicted_hypothesis.covariance = transition_matrix_ * hypothesis.covariance * transition_matrix_.transpose()
      + time_delta * process_noise_covariance_matrix_;

    return predicted_hypothesis;
  }

  void GmPhdCvPose::PrepareTransitionMatrix(void) {
    transition_matrix_ = StateSizeMatrix::Zero();

    transition_matrix_(0u, 0u) = 1.0;
    transition_matrix_(0u, 2u) = time_delta;

    transition_matrix_(1u, 1u) = 1.0;
    transition_matrix_(1u, 3u) = time_delta;

    transition_matrix_(2u, 2u) = 1.0;

    transition_matrix_(3u, 3u) = 1.0;
  }

  void GmPhdCvPose::PrepareProcessNoiseMatrix(void) {
    process_noise_covariance_matrix_ = StateSizeMatrix::Zero();

    for (auto index = 0u; index < calibrations_.process_noise_diagonal.size(); index++)
      process_noise_covariance_matrix_(index, index) = calibrations_.process_noise_diagonal.at(index);
  }

  void GmPhdCvPose::PredictBirths(void) {
    for (auto index = 0; index < 1000u; index++) {
      Hypothesis birth_hypothesis;

      birth_hypothesis.weight = 1.0;

      birth_hypothesis.state(0u) = pose_dist(e);
      birth_hypothesis.state(1u) = pose_dist(e);
      birth_hypothesis.state(2u) = velocity_dist(e);
      birth_hypothesis.state(3u) = velocity_dist(e);

      birth_hypothesis.covariance = 0.01 * StateSizeMatrix::Identity();

      const auto predicted_measurement = calibrations_.observation_matrix * birth_hypothesis.state;
      const auto innovation_covariance = calibrations_.measurement_covariance + calibrations_.observation_matrix * birth_hypothesis.covariance * calibrations_.observation_matrix.transpose();
      const auto kalman_gain = birth_hypothesis.covariance * calibrations_.observation_matrix.transpose() * innovation_covariance.inverse();
      const auto predicted_covariance = (StateSizeMatrix::Identity() - kalman_gain * calibrations_.observation_matrix) * birth_hypothesis.covariance;

      predicted_hypothesis_.push_back(PredictedHypothesis(birth_hypothesis, predicted_measurement, innovation_covariance, kalman_gain, predicted_covariance));
    }
  }
} // namespace mot
