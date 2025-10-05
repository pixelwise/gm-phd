#ifndef GM_PHD_INCLUDE_GM_PHD_CALIBRATIONS_HPP_
#define GM_PHD_INCLUDE_GM_PHD_CALIBRATIONS_HPP_

#include <array>

#include <Eigen/Dense>

namespace mot {
  enum class merge_mode_t {moment_match, absorb};
  template <size_t state_size, size_t measurement_size>
  struct GmPhdCalibrations {
    Eigen::Vector<float, state_size> process_noise_diagonal = {};                 // Process noise covariance matrix

    Eigen::Matrix<float, measurement_size, state_size> observation_matrix;     // Observation matrix

    float pd = 0.8;                                                       // Probability of detection
    float ps = 0.8;                                                        // Probability of survival
    float kappa = 1.0e-9;
    float lambda = 1.0e-5; // birth measurement likelihood threshold

    float truncation_threshold = 0.1;
    float merging_threshold = 3.0;
    float extraction_threshold = 3.0;

    merge_mode_t merge_mode = merge_mode_t::absorb;
  };
} //  namespace mot

#endif  //  GM_PHD_INCLUDE_GM_PHD_CALIBRATIONS_HPP_
