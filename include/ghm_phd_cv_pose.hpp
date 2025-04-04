#ifndef GM_PHD_INCLUDE_GM_PHD_CV_HPP_
#define GM_PHD_INCLUDE_GM_PHD_CV_HPP_

#include "gm_phd.hpp"
#include <Eigen/src/Core/Diagonal.h>
#include <random>
#include <iostream>

namespace mot {
  class GmPhdCvPose : public GmPhd<4u, 2u> {
    public:
      explicit GmPhdCvPose(const GmPhdCalibrations<4u, 2u> & calibrations);
      virtual ~GmPhdCvPose(void) = default;

    protected:
      Hypothesis PredictHypothesis(const Hypothesis & hypothesis);
      void PrepareTransitionMatrix(void);
      void PrepareProcessNoiseMatrix(void);

      void PredictBirths(void);
    private:
      std::random_device _rand_device;
      std::default_random_engine _rand_engine{_rand_device()};
      std::uniform_real_distribution<double> _pose_dist{-1.0, 1.0};
      std::uniform_real_distribution<double> _velocity_dist{-1.0, 1.0};
  };
  GmPhdCvPose::GmPhdCvPose(const GmPhdCalibrations<4u, 2u> & calibrations)
  : GmPhd<4u, 2u>(calibrations)
  , _rand_engine{_rand_device()}
  {
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
    constexpr auto birth_objects_number = 100u;
    for (auto index = 0; index < birth_objects_number; index++) {
      Hypothesis birth_hypothesis;
      birth_hypothesis.weight = 2.0 / static_cast<double>(birth_objects_number);
      birth_hypothesis.state(0u) = _pose_dist(_rand_engine) * calibrations_.init_pose_range_spread(0) + calibrations_.init_pose_range_mean(0);
      birth_hypothesis.state(1u) = _pose_dist(_rand_engine) * calibrations_.init_pose_range_spread(1) + calibrations_.init_pose_range_mean(1);
      birth_hypothesis.state(2u) = _velocity_dist(_rand_engine);
      birth_hypothesis.state(3u) = _velocity_dist(_rand_engine);
      birth_hypothesis.covariance = calibrations_.init_state_covariance.asDiagonal();
      const auto predicted_measurement = calibrations_.observation_matrix * birth_hypothesis.state;
      const auto innovation_covariance = calibrations_.measurement_covariance + calibrations_.observation_matrix * birth_hypothesis.covariance * calibrations_.observation_matrix.transpose();
      const auto kalman_gain = birth_hypothesis.covariance * calibrations_.observation_matrix.transpose() * innovation_covariance.inverse();
      const auto predicted_covariance = (StateSizeMatrix::Identity() - kalman_gain * calibrations_.observation_matrix) * birth_hypothesis.covariance;
      predicted_hypothesis_.push_back(PredictedHypothesis(birth_hypothesis, predicted_measurement, innovation_covariance, kalman_gain, predicted_covariance));
    }
  }
} //  namespace mot

#endif  //  GM_PHD_INCLUDE_GM_PHD_CV_HPP_
