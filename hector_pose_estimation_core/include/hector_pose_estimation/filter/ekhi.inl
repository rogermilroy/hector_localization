//=================================================================================================

// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//     * Neither the name of the Flight Systems and Automatic Control group,
//       TU Darmstadt, nor the names of its contributors may be used to
//       endorse or promote products derived from this software without
//       specific prior written permission.

// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//=================================================================================================

//=================================================================================================
// The below work is derived from the EKF class written by Johannes Meyer TU Darmstadt.
//
// Modifications carried out by Roger Milroy.
//=================================================================================================


#ifndef HECTOR_POSE_ESTIMATION_FILTER_EKHI_INL
#define HECTOR_POSE_ESTIMATION_FILTER_EKHI_INL

#include <hector_pose_estimation/filter/ekhi.h>
#include <boost/utility/enable_if.hpp>

namespace hector_pose_estimation {
  namespace filter {

    template<class ConcreteModel, typename Enabled>
    bool EKHI::Predictor_<ConcreteModel, Enabled>::predict(double dt) {
      // update F
      ROS_ERROR("predicting finding F");
      this->model_->getStateJacobian(F, state(), dt, this->init_);
//      ROS_ERROR_STREAM("System F = [" << std::endl << F << "]");

      // Prediction should happen only in main EKHi
      this->init_ = false;
      return true;
    }

    template<class ConcreteModel, typename Enabled>
    bool EKHI::Corrector_<ConcreteModel, Enabled>::correct(const typename ConcreteModel::MeasurementVector &y,
                                                           const typename ConcreteModel::NoiseVariance &R) {
      // this directly updates the base filters y
      this->model_->getCorrectedValue(y, filter_->yt, state());
//      ROS_ERROR_STREAM("base filter yt = [" << filter_->yt << "]");

      this->init_ = false;
      return true;
    }

  } // namespace filter
} // namespace hector_pose_estimation

#endif // HECTOR_POSE_ESTIMATION_FILTER_EKF_INL
