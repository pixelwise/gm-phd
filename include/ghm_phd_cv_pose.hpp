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

      void PrepareTransitionMatrix(double time_delta) override;
      void PrepareProcessNoiseMatrix(void) override;
      void PredictBirths(void) override;

    private:
      std::random_device _rand_device;
      std::default_random_engine _rand_engine{_rand_device()};
      std::uniform_real_distribution<double> _pose_dist{-1.0, 1.0};
      std::uniform_real_distribution<double> _velocity_dist{-1.0, 1.0};
  };

  inline GmPhdCvPose::GmPhdCvPose(const GmPhdCalibrations<4u, 2u> & calibrations)
  : GmPhd<4u, 2u>(calibrations)
  , _rand_engine{_rand_device()}
  {
  }

  inline void GmPhdCvPose::PrepareTransitionMatrix(double time_delta)
  {
    transition_matrix_ = StateSizeMatrix::Zero();

    transition_matrix_(0u, 0u) = 1.0;
    transition_matrix_(0u, 2u) = time_delta;

    transition_matrix_(1u, 1u) = 1.0;
    transition_matrix_(1u, 3u) = time_delta;

    transition_matrix_(2u, 2u) = 1.0;

    transition_matrix_(3u, 3u) = 1.0;
  }

  inline void GmPhdCvPose::PrepareProcessNoiseMatrix(void) {
    process_noise_covariance_matrix_ = StateSizeMatrix::Zero();

    for (auto index = 0u; index < calibrations_.process_noise_diagonal.size(); index++)
      process_noise_covariance_matrix_(index, index) = calibrations_.process_noise_diagonal.at(index);
  }

  inline void GmPhdCvPose::PredictBirths(void) {
    constexpr auto birth_objects_number = 100u;
    for (auto index = 0; index < birth_objects_number; index++) {
      double weight = 2.0 / static_cast<double>(birth_objects_number);
      StateSizeVector state;
      state(0u) = _pose_dist(_rand_engine) * calibrations_.init_pose_range_spread(0) + calibrations_.init_pose_range_mean(0);
      state(1u) = _pose_dist(_rand_engine) * calibrations_.init_pose_range_spread(1) + calibrations_.init_pose_range_mean(1);
      state(2u) = _velocity_dist(_rand_engine);
      state(3u) = _velocity_dist(_rand_engine);
      Spawn(Hypothesis{weight, state, calibrations_.init_state_covariance.asDiagonal()});
    }
  }
} //  namespace mot

#endif  //  GM_PHD_INCLUDE_GM_PHD_CV_HPP_
