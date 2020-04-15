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


#ifndef HECTOR_POSE_ESTIMATION_FILTER_EKHI_H  // TODO check that these definitions are a good
#define HECTOR_POSE_ESTIMATION_FILTER_EKHI_H

#include <hector_pose_estimation/filter.h>
#include <torch/torch.h>

#include <ros/console.h>

namespace hector_pose_estimation {
  namespace filter {

    class EKHI : public Filter {
    public:
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW

      EKHI(State &state);

      virtual ~EKHI();

      virtual std::string getType() const { return "EKHI"; }

      virtual bool init(PoseEstimation &estimator);

      virtual bool preparePredict(double dt);

      virtual bool predict(const SystemPtr &system, double dt);

      virtual bool doPredict(double dt);

      virtual bool prepareCorrect();

      virtual bool correct(const MeasurementPtr &measurement);

      virtual bool doCorrect();

      at::Tensor systemMatrixToTensor(const State::SystemMatrix &matrix, at::Device device) {
        auto eig_mat = hector_pose_estimation::State::systemMatrixToVector(matrix);

        // Converts each sub vector to tensor and stacks into a container vector
        std::vector <at::Tensor> tensors;
        for (const auto &x : eig_mat) {
          tensors.push_back(torch::tensor(x,torch::dtype(at::kFloat).device(device)));
        }
        // This stacks those tensors to make a 2d tensor. Must be transposed
        // due to mapping taking columnwise and stack going rowwise.
        return torch::stack(tensors).t();
      }

      void tensorToVector(const at::Tensor &tensor, State::Vector &vec) {
        double *temp = tensor.to(at::kDouble).data_ptr<double>();

        vec = Eigen::Map<State::Vector>(temp, torch::size(tensor, 0), 1);
      }

      void modelTensorToStateVector(const at::Tensor &tensor, State::Vector &vec) {
        ROS_WARN_STREAM("input tensor = " << tensor);
        State::Vector temp;
        tensorToVector(tensor, temp);
        ROS_WARN_STREAM("converted tensor = " << temp.transpose());
        State::Vector y_eig(15);
        // to [orientation, position, velocity, blank, blank]
        y_eig << temp.head(3), temp.segment(6, 6), temp.segment(3, 3), temp.segment(12, 3);
        ROS_WARN_STREAM("converted tensor, reformatted = " << y_eig.transpose());
        y_eig = y_eig.unaryExpr([](double v) { return std::isfinite(v)? v : 0.0; });
        ROS_WARN_STREAM("converted tensor, reformatted = " << y_eig.transpose());
        vec = y_eig;
      }

      void getStateAsEuler(State::Vector &vec) {
        // get state vector and convert to have euler..
        State::Vector curr = state().getVector();
        // from [orientation, rate, position, velocity]
        State::Vector orientation = state().getEuler();
        State::Vector curr_eul(15);
        curr_eul << orientation, curr.tail(12);
        ROS_WARN_STREAM("curr_eul = " << curr_eul.transpose());
        vec = curr_eul;
      }

      class Predictor {
      public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        Predictor(EKHI *filter)
          : filter_(filter),
            F(filter->state().getCovarianceDimension(), filter->state().getCovarianceDimension()),
            Q(filter->state().getCovarianceDimension(), filter->state().getCovarianceDimension()) {
          F.setZero();
          Q.setZero();
        }

        virtual ~Predictor() {}

        virtual bool predict(double dt) = 0;

      protected:
        EKHI *filter_;

      public:
        State::SystemMatrix F;
        State::Covariance Q;
      };

      template<class ConcreteModel, typename Enabled = void>
      class Predictor_ : public Filter::template Predictor_<ConcreteModel>, public Predictor {
      public:
        typedef ConcreteModel Model;
        typedef typename Filter::template Predictor_<ConcreteModel> Base;
        using Filter::template Predictor_<ConcreteModel>::state;

        Predictor_(EKHI *filter, Model *model)
          : Base(filter, model), Predictor(filter) {}

        virtual ~Predictor_() {}

        virtual bool predict(double dt);
      };

      class Corrector {
      public:
        Corrector(EKHI *filter) : filter_(filter) {}

        virtual ~Corrector() {}

      protected:
        EKHI *filter_;
      };

      template<class ConcreteModel, typename Enabled = void>
      class Corrector_ : public Filter::template Corrector_<ConcreteModel>, public Corrector {
      public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        typedef ConcreteModel
        Model;
        typedef typename Filter::template Corrector_<ConcreteModel> Base;
        using Filter::template Corrector_<ConcreteModel>::state;

        Corrector_(EKHI *filter, Model *model) : Base(filter, model), Corrector(filter) {}

        virtual ~Corrector_() {}

        virtual bool correct(const typename ConcreteModel::MeasurementVector &y,
                             const typename ConcreteModel::NoiseVariance &R);

      };

    public:
      State::SystemMatrix F; // This is the current F
      State::Covariance Q; // Q
      at::Tensor Ft; // This is the accumulated F between corrects.
      at::Tensor Fs;  // tensor            // This is the accumulated Fs (last 100 ish?) as a tensor
      at::Tensor yt;  // this is for assembling the y.
      at::Tensor ys;  // Should store these every time the main correct is called.
      int predict_steps; // This is for padded ys for prediction. TODO add this in.
      at::Tensor xs;  // this stores a full representation of state space
      torch::jit::script::Module model;
      torch::Device dev;
    };

  } // namespace filter
} // namespace hector_pose_estimation

#include "ekhi.inl"

#endif // HECTOR_POSE_ESTIMATION_FILTER_HI_H
