#ifndef GM_PHD_INCLUDE_GM_PHD_CV_HPP_
#define GM_PHD_INCLUDE_GM_PHD_CV_HPP_

#include "gm_phd.hpp"
#include <Eigen/src/Core/Diagonal.h>

namespace mot {

  class GmPhdCvPose : public GmPhd<4u, 2u> {
    public:
      explicit GmPhdCvPose(const GmPhdCalibrations<4u, 2u> & calibrations);
      virtual ~GmPhdCvPose(void) = default;

    protected:
      using float_type_t = GmPhd<4u, 2u>::float_type_t;
      void PrepareTransitionMatrix(float_type_t time_delta) override;
      void PrepareProcessNoiseMatrix(void) override;
  };

  inline GmPhdCvPose::GmPhdCvPose(const GmPhdCalibrations<4u, 2u> & calibrations)
  : GmPhd<4u, 2u>(calibrations)
  {
  }

  inline void GmPhdCvPose::PrepareTransitionMatrix(float_type_t time_delta)
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
    for (size_t i = 0; i < 4; ++i)
      process_noise_covariance_matrix_(i, i) = calibrations_.process_noise_diagonal(i);
  }

} //  namespace mot

#endif  //  GM_PHD_INCLUDE_GM_PHD_CV_HPP_
