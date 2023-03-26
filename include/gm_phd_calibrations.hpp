#ifndef GM_PHD_INCLUDE_GM_PHD_CALIBRATIONS_HPP_
#define GM_PHD_INCLUDE_GM_PHD_CALIBRATIONS_HPP_

#include <array>

#include <Eigen/Dense>

namespace mot {
  template <size_t state_size, size_t measurement_size>
  struct GmPhdCalibrations {
    std::array<double, state_size> process_noise_diagonal = {};                 // Process noise covariance matrix

    Eigen::Matrix<double, measurement_size, state_size> observation_matrix;     // Observation matrix

    double pd = 1.0;                                                            // Probability of detection
    double ps = 1.0;                                                            // Probability of survival
  };
} //  namespace mot

#endif  //  GM_PHD_INCLUDE_GM_PHD_CALIBRATIONS_HPP_
