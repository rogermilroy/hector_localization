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

#include <hector_pose_estimation/filter/ekhi.h>
#include <hector_pose_estimation/system.h>

#include <boost/pointer_cast.hpp>
#include <torch/script.h>



#ifdef USE_HECTOR_TIMING
#include <hector_diagnostics/timing.h>
#endif

namespace hector_pose_estimation {
  namespace filter {

    EKHI::EKHI(State &state)
      : Filter(state),
      device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU)
      {}

    EKHI::~EKHI() {}


    bool EKHI::init(PoseEstimation &estimator) {

      F = State::SystemMatrix(state_.getCovarianceDimension(), state_.getCovarianceDimension());
      Q = State::Covariance(state_.getCovarianceDimension(), state_.getCovarianceDimension());
      Ft = EKHI::systemMatrixToTensor(F, device);
      ROS_ERROR_STREAM("USING " << torch::get_device(Ft));
      Fs = torch::zeros_like(Ft);
      Fs = torch::unsqueeze(Fs, 0); // make 3d so cat works correctly.
      yt = torch::zeros(state().getCovarianceDimension());

      ys = torch::zeros_like(yt, device);
      ys = torch::unsqueeze(ys, 0); // make 2d so cat works correctly.
      // Load EKHI model here

      model = torch::jit::load(
        "/home/r/Documents/FinalProject/FullUnit_1920_RogerMilroy/Code/hybrid_inference/src"
        "/torchscript/ekhi_model_larger.pt");
      model.to(device);

      ROS_WARN(
        "+++++++++++++++++++++++++++++++++ INIT EKHI +++++++++++++++++++=======================");

      return true;
    }

    bool EKHI::preparePredict(double dt) {
//   Here maybe select the most recent 100?
      // This resets the current F which will then be updated with Jacobians.
      // Set up the right number of steps based on dt....... PAD YS (maybe just call predict)
//  predict_steps = 1;   // TODO see if this can be dynamic based on the rate.
      F.setIdentity();
      Q.setZero();

      return Filter::preparePredict(dt); // this just returns true.
    }

    bool EKHI::predict(const SystemPtr &system, double dt) {
      // This if statement calls the systems Predictors predict ( updates their F and Q )
      if (!Filter::predict(system, dt)) return false;
      auto *predictor = boost::dynamic_pointer_cast<EKHI::Predictor>(system->predictor());
      // this then updates the global F and Q before the doPredict step
      F += predictor->F;
      Q += predictor->Q;

      return true;
    }

    bool EKHI::doPredict(double dt) {
      ROS_WARN("EKHI prediction (dt = %f):", dt);

      ROS_WARN_STREAM("F      = [" << std::endl << F << "]");
      ROS_WARN_STREAM("Q      = [" << std::endl << Q << "]");

      // update accumulated F
      Ft = Ft.matmul(EKHI::systemMatrixToTensor(F, device));

      // CALL THE MODEL.. GET the last element (predicted state)
      std::vector <torch::jit::IValue> inputs;
      // here I replace the predict by just adding zeros to the end here instead of in predict.
      inputs.emplace_back(torch::unsqueeze(torch::cat({ys, torch::unsqueeze(torch::zeros_like(yt,
        device), 0)}), 0));
      // add Ft to the end of Fs for predictions.
      inputs.emplace_back(torch::cat({Fs, torch::unsqueeze(Ft, 0)}));

      torch::NoGradGuard no_grad_guard;
      using namespace std::chrono;
      auto start = high_resolution_clock::now();
      xs = model.forward(inputs).toTensor();
      auto stop = high_resolution_clock::now();
      auto duration = duration_cast<microseconds>(stop - start);
      std::cout << "Model execution time: " << duration.count() << std::endl;

      xs = torch::squeeze(xs, 0);

      ROS_WARN_STREAM("xs = [" << xs << "]");

      State::Vector curr_eul = getStateAsEuler();

      // convert to same format..
      State::Vector pred_x = modelTensorToStateVector(xs.slice(/*dim*/ 1, /*start*/ xs.size(1) - 1, /*end*/ xs.size(1)));

      // calculate difference between predicted x and state
      State::Vector diff = curr_eul - pred_x;

      // THEN UPDATE THE STATE..
      state().update(diff);

//  ROS_WARN_STREAM("x_pred = [" << state().getVector().transpose() << "]");

      return true;
    }

    bool EKHI::prepareCorrect() {
      if (!Filter::prepareCorrect()) return false;

      // Add Ft to Fs  - This is done here because the F matrix is updated only in predict steps so
      // this is the most recent F.
      Fs = torch::cat({Fs, torch::unsqueeze(Ft, 0)});
      // reset Ft
      Ft = torch::eye(Ft.size(0)); // TODO check this doesn't have weird reset issues like y did...

      // make sure only 100 in Fs (most recent 100)
      if (Fs.size(0) > 100) {
        Fs = torch::slice(Fs, /*dim*/ 0, /*start*/ Fs.size(0) - 100, /*end*/ Fs.size(0));
      }

//  ROS_WARN_STREAM("y pre reset = [" << y << "]");
      return true;
    }

    bool EKHI::correct(const MeasurementPtr &measurement) {
      // this is just for clarity so that it is easier to trace the calls.
      // Filter::correct(Measurements) just iterates over the measurements and calls this method on
      // them each. then doCorrect.
      return Filter::correct(measurement);
    }

    bool EKHI::doCorrect() {

      ROS_WARN("EKHI correction");
      // manage the ys, add new y to ys  -- This is done here because the correct steps update y
      // so only now is y the most recent y.
      ys = torch::cat({ys, torch::unsqueeze(yt, 0)});

      // make sure ys aren't longer than 100
      if (ys.size(0) > 100) {
        ys = torch::slice(ys, /*dim*/ 0, /*start*/ ys.size(0) - 100, /*end*/ ys.size(0));
      }

      at::print(std::cout, ys, 130);
//      at::print(std::cout, Fs, 130);

      // CALL THE MODEL
      std::vector <torch::jit::IValue> inputs;
      inputs.emplace_back(torch::unsqueeze(ys, 0));
      inputs.emplace_back(Fs);

      torch::NoGradGuard no_grad_guard;
      using namespace std::chrono;
      auto start = high_resolution_clock::now();
      xs = model.forward(inputs).toTensor();
      auto stop = high_resolution_clock::now();
      auto duration = duration_cast<microseconds>(stop - start);
      std::cout << "Model execution time: " << duration.count() << std::endl;

      xs = torch::squeeze(xs, 0);
      ROS_WARN_STREAM("xs = [" << xs << "]");

      // extract the current state
      State::Vector curr_eul = getStateAsEuler();

      // convert to same format..
      State::Vector pred_x = modelTensorToStateVector(xs.slice(/*dim*/ 1, /*start*/ xs.size(1) - 1, /*end*/ xs.size(1)));

      // calculate difference between predicted x and state
      State::Vector diff = curr_eul - pred_x;

      // THEN UPDATE THE STATE..
      state().update(diff);

//  ROS_WARN_STREAM("diff = [" << diff.transpose() << "]");
      return true;
    }
  } // namespace filter
} // namespace hector_pose_estimation
