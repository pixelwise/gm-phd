#ifndef GM_PHD_INCLUDE_GM_PHD_HPP_
#define GM_PHD_INCLUDE_GM_PHD_HPP_

#include <algorithm>
#include <cmath>
#include <numbers>
#include <numeric>
#include <optional>
#include <ranges>
#include <vector>
#include <iostream>

#include <Eigen/Dense>

#include "gm_phd_calibrations.hpp"
#include "value_with_covariance.hpp"

namespace mot {
  template <size_t state_size, size_t measurement_size>
  class GmPhd {
    public:
      using float_type_t = float;
      using StateSizeVector = Eigen::Vector<float_type_t, state_size>;
      using StateSizeMatrix = Eigen::Matrix<float_type_t, state_size, state_size>;
      using MeasurementSizeVector = Eigen::Vector<float_type_t, measurement_size>;
      using MeasurementSizeMatrix = Eigen::Matrix<float_type_t, measurement_size, measurement_size>;
      using SensorPoseVector = Eigen::Vector<float_type_t, 3u>;
      using SensorPoseMatrix = Eigen::Matrix<float_type_t, 3u, 3u>;

      using Object = ValueWithCovariance<state_size>;
      using Measurement = ValueWithCovariance<measurement_size>;

      struct Hypothesis {
        float_type_t weight;
        StateSizeVector state;
        StateSizeMatrix covariance;
      };

      explicit GmPhd(const GmPhdCalibrations<state_size, measurement_size> & calibrations)
        : calibrations_{calibrations} {}

      virtual ~GmPhd(void) = default;

      void Predict(const float_type_t time_delta)
      {
        PredictExistingTargets(time_delta);
        UpdateExistedHypothesis();
      }

      void Update(const std::vector<Measurement> & measurements)
      {
        MakeMeasurementUpdate(measurements);
        Prune();
        ExtractObjects();
      }

      const std::vector<Object> & GetObjects(void) const {
        return objects_;
      }

      float_type_t GetWeightsSum(void) const {
        return std::accumulate(hypothesises_.begin(), hypothesises_.end(),
          0.0,
          [](float_type_t sum, const Hypothesis & hypothesis) {
            return sum + hypothesis.weight;
          }
        );
      }

      void Spawn(Hypothesis birth_hypothesis)
      {
        AddPredictedHypothesis(birth_hypothesis);
      }

      void Reset()
      {
        hypothesises_.clear();
      }

    protected:

      virtual void PrepareTransitionMatrix(float_type_t time_delta) = 0;
      virtual void PrepareProcessNoiseMatrix(void) = 0;

      std::vector<Hypothesis> predicted_hypothesis_;

      GmPhdCalibrations<state_size, measurement_size> calibrations_;
      StateSizeMatrix transition_matrix_ = StateSizeMatrix::Zero();
      StateSizeMatrix process_noise_covariance_matrix_ = StateSizeMatrix::Zero();
      std::vector<Object> objects_;
      std::vector<Hypothesis> hypothesises_;

    private:

      enum class merging_state_t {unmerged, merging, merged};
      using merge_candidate_t = std::pair<Hypothesis, merging_state_t>;

      void PredictExistingTargets(float_type_t time_delta)
      {
        predicted_hypothesis_.clear();
        PrepareTransitionMatrix(time_delta);
        PrepareProcessNoiseMatrix();
        for (auto& hypothesis : hypothesises_)
          AddPredictedHypothesis(PredictHypothesis(hypothesis, time_delta));
        hypothesises_.clear();
      }

      void AddPredictedHypothesis(Hypothesis hypothesis)
      {
        predicted_hypothesis_.push_back(hypothesis);        
      }

      Hypothesis PredictHypothesis(const Hypothesis & hypothesis, float_type_t time_delta) {
        return Hypothesis{
          calibrations_.ps * hypothesis.weight,
          transition_matrix_ * hypothesis.state,
          transition_matrix_ * hypothesis.covariance * transition_matrix_.transpose() + time_delta * process_noise_covariance_matrix_
        };
      }

      void UpdateExistedHypothesis(void) {
        for (auto& predicted : predicted_hypothesis_)
        {
          float_type_t weight = predicted.weight * (1.0 - calibrations_.pd);
          if (weight > calibrations_.truncation_threshold)
          {
            auto hypothesis = predicted;
            hypothesis.weight = weight;
            hypothesises_.push_back(hypothesis);
          }
        }
      }

      void MakeMeasurementUpdate(const std::vector<Measurement> & measurements) {
        std::vector<Hypothesis> new_hypothesises;
        const size_t n = predicted_hypothesis_.size();
        std::vector<float> weights(n);
        for (const auto & measurement : measurements) {
          new_hypothesises.clear();
          float_type_t Z = 0;
          for (size_t i = 0; i < n; ++i)
          {
            auto& prediction = predicted_hypothesis_[i];
            weights[i] = DetectionWeight(prediction, measurement);
            Z += weights[i];
          }
          double H = 0;
          size_t n_assigned = 0;
          for (size_t i = 0; i < n; ++i)
          {
            auto weight = weights[i]; 
            auto p = double(weight) / Z;
            if (p > 1e-10)
              H -= p * std::log(p);
            weight /= calibrations_.kappa + Z;
            if (weight > calibrations_.truncation_threshold)
            {
              n_assigned++;
              auto& prediction = predicted_hypothesis_[i];
              hypothesises_.push_back(KalmanUpdatedHypothesis(prediction, measurement, weight));
            }
          }
          std::cerr << "measurement assignment entropy " << H << " <n> " << std::exp(H) << " n " << n_assigned << std::endl;
        }
      }

      float DetectionWeight(const Hypothesis& prediction, const Measurement& measurement)
      {
        const auto predicted_measurement = calibrations_.observation_matrix * prediction.state;
        const auto predicted_covariance = measurement.covariance + calibrations_.observation_matrix * prediction.covariance * calibrations_.observation_matrix.transpose();
        const auto weight = calibrations_.pd * prediction.weight * NormPdf(measurement.value, predicted_measurement, predicted_covariance);        
      }

      Hypothesis KalmanUpdatedHypothesis(const Hypothesis& prediction, const Measurement& measurement, float weight) const
      {
        const auto innovation = measurement.value - calibrations_.observation_matrix * prediction.state;
        const auto innovation_covariance = measurement.covariance + calibrations_.observation_matrix * prediction.covariance * calibrations_.observation_matrix.transpose();
        const auto kalman_gain = prediction.covariance * calibrations_.observation_matrix.transpose() * innovation_covariance.inverse();
        const auto state = prediction.state + kalman_gain * innovation;
        const auto covariance = (StateSizeMatrix::Identity() - kalman_gain * calibrations_.observation_matrix) * prediction.covariance;
        return Hypothesis(weight, state, covariance);
      }

      void Prune(void)
      {
        auto merge_candidates = GetMergeCandidates();
        hypothesises_.clear();
        while (SelectMergeSet(merge_candidates))
        {
          hypothesises_.push_back(MergeSelected(merge_candidates));
          MarkMerged(merge_candidates);
        }
        std::cerr << "merged " << merge_candidates.size() << " -> " << hypothesises_.size() << std::endl;
      }

      std::vector<merge_candidate_t> GetMergeCandidates()
      {
        std::vector<merge_candidate_t> merge_candidates;
        for (auto& h : hypothesises_)
          if (h.weight >= calibrations_.truncation_threshold)
            merge_candidates.push_back({h, merging_state_t::unmerged});
        return merge_candidates;
      }

      bool SelectMergeSet(std::vector<merge_candidate_t>& candidates)
      {
        auto merge_center = SelectMergeCenter(candidates);
        if (!merge_center)
          return false;
        MarkMergeSet(candidates, *merge_center);
        return true;
      }

      std::optional<StateSizeVector> SelectMergeCenter(std::vector<merge_candidate_t>& candidates)
      {
        auto unmerged_candidates = candidates | std::views::filter([](auto& candidate) {return candidate.second == merging_state_t::unmerged;});
        if (std::ranges::empty(unmerged_candidates))
          return std::nullopt;
        auto imax = std::ranges::max_element(
          unmerged_candidates,
          [](const auto& a, const auto& b) {
            return a.first.weight < b.first.weight;
          }
        );
        imax->second = merging_state_t::merging;
        return imax->first.state;        
      }

      void MarkMergeSet(std::vector<merge_candidate_t>& candidates, StateSizeVector merge_center)
      {
        auto unmerged_candidates = candidates | std::views::filter([](auto& candidate) {return candidate.second == merging_state_t::unmerged;});
        for (auto& [hypothesis, mark] : unmerged_candidates)
        {
          const auto diff = hypothesis.state - merge_center;
          const auto distance_matrix = diff.transpose() * hypothesis.covariance.inverse() * diff;
          if (distance_matrix(0) < calibrations_.merging_threshold)
            mark = merging_state_t::merging;
        };
      }

      Hypothesis MergeSelected(std::vector<merge_candidate_t>& candidates)
      {
        auto merge_set = candidates | std::views::filter([](auto& candidate) {return candidate.second == merging_state_t::merging;});
        float_type_t merged_weight = 0;
        StateSizeVector merged_state = StateSizeVector::Zero();
        for (auto& [hypothesis, mark] : merge_set)
        {
          merged_weight += hypothesis.weight;
          merged_state += hypothesis.weight * hypothesis.state;
        }
        merged_state /= merged_weight;
        StateSizeMatrix merged_covariance = StateSizeMatrix::Zero();
        for (auto& [hypothesis, mark] : merge_set)
        {
          const auto diff = merged_state - hypothesis.state;
          merged_covariance += (hypothesis.covariance + diff * diff.transpose()) / merged_weight;
        }
        return Hypothesis(merged_weight, merged_state, merged_covariance);
      }

      void MarkMerged(std::vector<merge_candidate_t>& candidates)
      {
          for (auto& [hypothesis, mark] : candidates)
            if (mark == merging_state_t::merging)
              mark = merging_state_t::merged;        
      }

      void ExtractObjects(void) {
        objects_.clear();
        for (const auto & hypothesis : hypothesises_) {
          if (hypothesis.weight > calibrations_.extraction_threshold) {
            Object object;
            object.value = hypothesis.state;
            object.covariance = hypothesis.covariance;
            objects_.push_back(object);
          }
        }
      }

      static float_type_t NormPdf(const MeasurementSizeVector & z, const MeasurementSizeVector & nu, const MeasurementSizeMatrix & cov) {
        const auto diff = z - nu;
        const auto c = 1.0 / (std::sqrt(std::pow(std::numbers::pi, measurement_size) * cov.determinant()));
        const auto e = std::exp(-0.5 * diff.transpose() * cov.inverse() * diff);
        return c * e;
      }

  };
};  //  namespace mot

#endif  //  GM_PHD_INCLUDE_GM_PHD_HPP_
