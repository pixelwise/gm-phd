#ifndef GM_PHD_INCLUDE_GM_PHD_HPP_
#define GM_PHD_INCLUDE_GM_PHD_HPP_

#include <algorithm>
#include <cmath>
#include <numbers>
#include <numeric>
#include <ranges>
#include <tuple>
#include <vector>
#include <iostream>

#include <Eigen/Dense>

#include "gm_phd_calibrations.hpp"
#include "value_with_covariance.hpp"

namespace mot {
  template <size_t state_size, size_t measurement_size>
  class GmPhd {
    public:

      using StateSizeVector = Eigen::Vector<double, state_size>;
      using StateSizeMatrix = Eigen::Matrix<double, state_size, state_size>;
      using MeasurementSizeVector = Eigen::Vector<double, measurement_size>;
      using MeasurementSizeMatrix = Eigen::Matrix<double, measurement_size, measurement_size>;
      using SensorPoseVector = Eigen::Vector<double, 3u>;
      using SensorPoseMatrix = Eigen::Matrix<double, 3u, 3u>;

      using Object = ValueWithCovariance<state_size>;
      using Measurement = ValueWithCovariance<measurement_size>;

      struct Hypothesis {
        double weight;
        StateSizeVector state;
        StateSizeMatrix covariance;
      };

      struct PredictedHypothesis {
        Hypothesis hypothesis;
        MeasurementSizeVector predicted_measurement;
        MeasurementSizeMatrix innovation_matrix;
        Eigen::Matrix<double, state_size, measurement_size> kalman_gain;
        StateSizeMatrix updated_covariance;
      };

      explicit GmPhd(const GmPhdCalibrations<state_size, measurement_size> & calibrations)
        : calibrations_{calibrations} {}

      virtual ~GmPhd(void) = default;

      void Run(const double time_delta, const std::vector<Measurement> & measurements) {
        predicted_hypothesis_.clear();
        PredictExistingTargets(time_delta);
        hypothesis_.clear();
        UpdateExistedHypothesis();
        PredictBirths();
        MakeMeasurementUpdate(measurements);
        Prune();
        ExtractObjects();
      }

      const std::vector<Object> & GetObjects(void) const {
        return objects_;
      }

      double GetWeightsSum(void) const {
        return std::accumulate(hypothesis_.begin(), hypothesis_.end(),
          0.0,
          [](double sum, const Hypothesis & hypothesis) {
            return sum + hypothesis.weight;
          }
        );
      }

      void Spawn(Hypothesis birth_hypothesis)
      {
        AddPredictedHypothesis(birth_hypothesis);
      }

    protected:

      virtual void PrepareTransitionMatrix(double time_delta) = 0;
      virtual void PrepareProcessNoiseMatrix(void) = 0;
      virtual void PredictBirths(void) = 0;

      std::vector<PredictedHypothesis> predicted_hypothesis_;

      GmPhdCalibrations<state_size, measurement_size> calibrations_;
      StateSizeMatrix transition_matrix_ = StateSizeMatrix::Zero();
      StateSizeMatrix process_noise_covariance_matrix_ = StateSizeMatrix::Zero();

    private:

      void PredictExistingTargets(double time_delta)
      {
        PrepareTransitionMatrix(time_delta);
        PrepareProcessNoiseMatrix();
        for (auto& hypothesis : hypothesis_)
          AddPredictedHypothesis(PredictHypothesis(hypothesis, time_delta));
      }

      void AddPredictedHypothesis(Hypothesis hypothesis)
      {
        const auto predicted_measurement = calibrations_.observation_matrix * hypothesis.state;
        const auto innovation_covariance = calibrations_.measurement_covariance + calibrations_.observation_matrix * hypothesis.covariance * calibrations_.observation_matrix.transpose();
        const auto kalman_gain = hypothesis.covariance * calibrations_.observation_matrix.transpose() * innovation_covariance.inverse();
        const auto predicted_covariance = (StateSizeMatrix::Identity() - kalman_gain * calibrations_.observation_matrix) * hypothesis.covariance;
        predicted_hypothesis_.push_back(PredictedHypothesis(hypothesis, predicted_measurement, innovation_covariance, kalman_gain, predicted_covariance));        
      }

      Hypothesis PredictHypothesis(const Hypothesis & hypothesis, double time_delta) {
        return Hypothesis{
          calibrations_.ps * hypothesis.weight,
          transition_matrix_ * hypothesis.state,
          transition_matrix_ * hypothesis.covariance * transition_matrix_.transpose() + time_delta * process_noise_covariance_matrix_
        };
      }

      void UpdateExistedHypothesis(void) {
        for (auto& predicted : predicted_hypothesis_)
        {
          double weight = predicted.hypothesis.weight * (1.0 - calibrations_.pd);
          if (weight > calibrations_.truncation_threshold)
          {
            auto hypothesis = predicted.hypothesis;
            hypothesis.weight = weight;
            hypothesis_.push_back(hypothesis);
          }
        }
      }

      void MakeMeasurementUpdate(const std::vector<Measurement> & measurements) {
        std::vector<Hypothesis> new_hypothesises;
        for (const auto & measurement : measurements) {
          new_hypothesises.clear();
          double Z = 0;
          for (const auto & predicted_hypothesis : predicted_hypothesis_) {
            const auto weight = calibrations_.pd * predicted_hypothesis.hypothesis.weight * NormPdf(measurement.value, predicted_hypothesis.predicted_measurement, predicted_hypothesis.innovation_matrix);
            const auto state = predicted_hypothesis.hypothesis.state + predicted_hypothesis.kalman_gain * (measurement.value - predicted_hypothesis.predicted_measurement);
            const auto covariance = predicted_hypothesis.hypothesis.covariance;
            Z += weight;
            new_hypothesises.push_back(Hypothesis(weight, state, covariance));
          }
          for (auto & hypothesis : new_hypothesises)
          {
            hypothesis.weight /= calibrations_.kappa + Z;
            if (hypothesis.weight > calibrations_.truncation_threshold)
              hypothesis_.push_back(hypothesis);
          }
        }
      }

      void Prune(void) {
        // Select elements with weigths over turncation threshold
        enum class mark_t {unmerged, merging, merged};
        std::vector<std::pair<Hypothesis, mark_t>> pruned_hypothesis_marked;
        for (auto& h : hypothesis_)
          if (h.weight >= calibrations_.truncation_threshold)
            pruned_hypothesis_marked.push_back({h, mark_t::unmerged});
        std::vector<Hypothesis> merged_hypothesises;
        while (
          std::ranges::count_if(
            pruned_hypothesis_marked,
            [](auto& marked_hypothesis){return marked_hypothesis.second == mark_t::unmerged;}
          )
        ) {
          auto unmerged_hypothesises = pruned_hypothesis_marked | std::views::filter([](auto& marked_hypothesis) { return marked_hypothesis.second == mark_t::unmerged; });
          const auto maximum_weight_hypothesis = *std::ranges::max_element(
            unmerged_hypothesises,
            [](const auto& a, const auto& b) {
              return a.first.weight < b.first.weight;
            }
          );

          for (auto& [hypothesis, mark] : pruned_hypothesis_marked)
            if (mark == mark_t::unmerged)
            {
              const auto diff = hypothesis.state - maximum_weight_hypothesis.first.state;
              const auto distance_matrix = diff.transpose() * hypothesis.covariance.inverse() * diff;
              if (distance_matrix(0) < calibrations_.merging_threshold)
                mark = mark_t::merging;
            }

          // Calculate new merged element
          double merged_weight = 0;
          for (auto& [hypothesis, mark] : pruned_hypothesis_marked)
            if (mark == mark_t::merging)
              merged_weight += hypothesis.weight;          
          StateSizeVector merged_state = StateSizeVector::Zero();
          for (auto& [hypothesis, mark] : pruned_hypothesis_marked)
            if (mark == mark_t::merging)
              merged_state += (hypothesis.weight * hypothesis.state) / merged_weight;
          StateSizeMatrix merged_covariance = StateSizeMatrix::Zero();
          for (auto& [hypothesis, mark] : pruned_hypothesis_marked)
            if (mark == mark_t::merging)
            {
              const auto diff = merged_state - hypothesis.state;
              merged_covariance += (hypothesis.covariance + diff * diff.transpose()) / merged_weight;
            }
          merged_hypothesises.push_back(Hypothesis(merged_weight, merged_state, merged_covariance));

          for (auto& [hypothesis, mark] : pruned_hypothesis_marked)
            if (mark == mark_t::merging)
              mark = mark_t::merged;
        }
        // Set final hypothesis
        std::cerr << "merged from " << pruned_hypothesis_marked.size() << " down to " << merged_hypothesises.size() << " hypothesises" << std::endl;
        hypothesis_ = std::move(merged_hypothesises);
      }

      void ExtractObjects(void) {
        objects_.clear();
        for (const auto & hypothesis : hypothesis_) {
          if (hypothesis.weight > 0.5) {
            Object object;
            object.value = hypothesis.state;
            object.covariance = hypothesis.covariance;
            objects_.push_back(object);
          }
        }
      }

      static double NormPdf(const MeasurementSizeVector & z, const MeasurementSizeVector & nu, const MeasurementSizeMatrix & cov) {
        const auto diff = z - nu;
        const auto c = 1.0 / (std::sqrt(std::pow(std::numbers::pi, measurement_size) * cov.determinant()));
        const auto e = std::exp(-0.5 * diff.transpose() * cov.inverse() * diff);
        return c * e;
      }

      double prev_timestamp_ = 0.0;
      std::vector<Object> objects_;
      std::vector<Hypothesis> hypothesis_;
  };
};  //  namespace mot

#endif  //  GM_PHD_INCLUDE_GM_PHD_HPP_
