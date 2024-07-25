/*
 * MIT License
 *
 * Copyright (c) 2021 Yifu Zhang
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

// Copyright 2023 TIER IV, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "strack.h"

#include <ament_index_cpp/get_package_share_directory.hpp>

#include <boost/uuid/uuid_generators.hpp>

#include <yaml-cpp/yaml.h>

// init static variable
bool STrack::_parameters_loaded = false;
STrack::KfParams STrack::_kf_parameters;

STrack::STrack(std::vector<float> in_pose, std::vector<float> in_lwh, float score, int label)
{
  original_pose.resize(4);
  original_pose.assign(in_pose.begin(), in_pose.end());
  lwh.resize(3);
  lwh.assign(in_lwh.begin(), in_lwh.end());

  is_activated = false;
  track_id = 0;
  state = TrackState::New;

  pose.resize(4);

  static_pose();  // update object size
  frame_id = 0;
  tracklet_len = 0;
  this->score = score;
  start_frame = 0;
  this->label = label;

  // load static kf parameters: initialized once in program
  const std::string package_share_directory =
    ament_index_cpp::get_package_share_directory("bytetrack3d");
  const std::string default_config_path =
    package_share_directory + "/config/kalman_filter.param.yaml";
  if (!_parameters_loaded) {
    load_parameters(default_config_path);
    _parameters_loaded = true;
  }
}

STrack::~STrack()
{
}

void STrack::init_kalman_filter()
{
  // assert parameter is loaded
  assert(_parameters_loaded);

  // init kalman filter state
  Eigen::MatrixXd X0 = Eigen::MatrixXd::Zero(_kf_parameters.dim_x, 1);
  Eigen::MatrixXd P0 = Eigen::MatrixXd::Zero(_kf_parameters.dim_x, _kf_parameters.dim_x);
  X0(IDX::X) = this->original_pose[0];
  X0(IDX::Y) = this->original_pose[1];
  X0(IDX::Z) = this->original_pose[2];
  X0(IDX::Yaw) = this->original_pose[3];
  X0(IDX::L) = this->lwh[0];
  X0(IDX::W) = this->lwh[1];
  X0(IDX::H) = this->lwh[2];
  X0(IDX::VX) = 0;
  X0(IDX::VY) = 0;
  X0(IDX::VZ) = 0;
  X0(IDX::VYaw) = 0;


  P0(IDX::X, IDX::X) = _kf_parameters.p0_cov_p;
  P0(IDX::Y, IDX::Y) = _kf_parameters.p0_cov_p;
  P0(IDX::Z, IDX::Z) = _kf_parameters.p0_cov_p;
  P0(IDX::Yaw, IDX::Yaw) = _kf_parameters.p0_cov_pyaw;
  P0(IDX::L, IDX::L) = _kf_parameters.p0_cov_d;
  P0(IDX::W, IDX::W) = _kf_parameters.p0_cov_d;
  P0(IDX::H, IDX::H) = _kf_parameters.p0_cov_d;
  P0(IDX::VX, IDX::VX) = _kf_parameters.p0_cov_v;
  P0(IDX::VY, IDX::VY) = _kf_parameters.p0_cov_v;
  P0(IDX::VZ, IDX::VZ) = _kf_parameters.p0_cov_v;
  P0(IDX::VYaw, IDX::VYaw) = _kf_parameters.p0_cov_vyaw;
  this->kalman_filter_.init(X0, P0);
}

/** init a tracklet */
void STrack::activate(int frame_id)
{
  this->track_id = this->next_id();
  this->unique_id = boost::uuids::random_generator()();

  // init kf
  init_kalman_filter();
  // reflect state
  reflect_state();

  this->tracklet_len = 0;
  this->state = TrackState::Tracked;
  this->is_activated = true;
  this->frame_id = frame_id;
  this->start_frame = frame_id;
}

void STrack::re_activate(STrack & new_track, int frame_id, bool new_id)
{
  // TODO(me): write kf update
  Eigen::MatrixXd measurement = Eigen::MatrixXd::Zero(_kf_parameters.dim_z, 1);
  measurement << new_track.pose[0], new_track.pose[1], new_track.pose[2], new_track.pose[3];
  update_kalman_filter(measurement);

  reflect_state();

  this->tracklet_len = 0;
  this->state = TrackState::Tracked;
  this->is_activated = true;
  this->frame_id = frame_id;
  this->score = new_track.score;
  if (new_id) {
    this->track_id = next_id();
    this->unique_id = boost::uuids::random_generator()();
  }
}

void STrack::update(STrack & new_track, int frame_id)
{
  this->frame_id = frame_id;
  this->tracklet_len++;

  // update
  Eigen::MatrixXd measurement = Eigen::MatrixXd::Zero(_kf_parameters.dim_z, 1);
  measurement << new_track.pose[0], new_track.pose[1], new_track.pose[2], new_track.pose[3];
  update_kalman_filter(measurement);

  reflect_state();

  this->state = TrackState::Tracked;
  this->is_activated = true;

  this->score = new_track.score;
}

/** reflect kalman filter state to current object variables*/
void STrack::reflect_state()
{
  static_pose();
}

void STrack::static_pose()
{
  if (this->state == TrackState::New) {
    pose[0] = original_pose[0];
    pose[1] = original_pose[1];
    pose[2] = original_pose[2];
    pose[3] = original_pose[3];
    return;
  }
  // put kf state to pose
  Eigen::MatrixXd X = Eigen::MatrixXd::Zero(_kf_parameters.dim_x, 1);
  this->kalman_filter_.getX(X);
  pose[0] = X(IDX::X);
  pose[1] = X(IDX::Y);
  pose[2] = X(IDX::Z);
  pose[3] = X(IDX::Yaw);
}


void STrack::mark_lost()
{
  state = TrackState::Lost;
}

void STrack::mark_removed()
{
  state = TrackState::Removed;
}

int STrack::next_id()
{
  static int _count = 0;
  _count++;
  return _count;
}

int STrack::end_frame()
{
  return this->frame_id;
}

void STrack::multi_predict(std::vector<STrack *> & stracks)
{
  for (size_t i = 0; i < stracks.size(); i++) {
    if (stracks[i]->state != TrackState::Tracked) {
      // not tracked
    }
    // prediction
    stracks[i]->predict(stracks[i]->frame_id + 1);
    stracks[i]->static_pose();
  }
}

void STrack::update_kalman_filter(const Eigen::MatrixXd & measurement)
{
  // assert parameter is loaded
  assert(_parameters_loaded);

  // get C matrix
  // 测量矩阵
  Eigen::MatrixXd C = Eigen::MatrixXd::Zero(_kf_parameters.dim_z, _kf_parameters.dim_x);
  C(IDX::X, IDX::X) = 1;
  C(IDX::Y, IDX::Y) = 1;
  C(IDX::Z, IDX::Z) = 1;
  C(IDX::Yaw, IDX::Yaw) = 1;
  C(IDX::L, IDX::L) = 1;
  C(IDX::W, IDX::W) = 1;
  C(IDX::H, IDX::H) = 1;

  // get R matrix(__Observation Noise Covariance Matrix__)
  Eigen::MatrixXd R = Eigen::MatrixXd::Zero(_kf_parameters.dim_z, _kf_parameters.dim_z);
  R(IDX::X, IDX::X) = _kf_parameters.r_cov_p;
  R(IDX::Y, IDX::Y) = _kf_parameters.r_cov_p;
  R(IDX::Z, IDX::Z) = _kf_parameters.r_cov_p;
  R(IDX::Yaw, IDX::Yaw) = _kf_parameters.r_cov_pyaw;
  R(IDX::L, IDX::L) = _kf_parameters.r_cov_d;
  R(IDX::W, IDX::W) = _kf_parameters.r_cov_d;
  R(IDX::H, IDX::H) = _kf_parameters.r_cov_d;

  // update
  if (!this->kalman_filter_.update(measurement, C, R)) {
    std::cerr << "Cannot update" << std::endl;
  }
}

void STrack::predict(const int frame_id)
{
  // check state is Tracked
  if (this->state != TrackState::Tracked) {
    // not tracked
    return;
  }

  // else do prediction
  float time_elapsed = _kf_parameters.dt * (frame_id - this->frame_id);
  // A matrix(__State Transition Matrix__)
  Eigen::MatrixXd A = Eigen::MatrixXd::Identity(_kf_parameters.dim_x, _kf_parameters.dim_x);
  A(IDX::X, IDX::VX) = time_elapsed;
  A(IDX::Y, IDX::VY) = time_elapsed;
  A(IDX::Z, IDX::VZ) = time_elapsed;
  A(IDX::Yaw, IDX::VYaw) = time_elapsed;

  // u and B matrix(____ and __Control Input Matrix__)
  Eigen::MatrixXd u = Eigen::MatrixXd::Zero(_kf_parameters.dim_x, 1);
  Eigen::MatrixXd B = Eigen::MatrixXd::Zero(_kf_parameters.dim_x, _kf_parameters.dim_x);

  // get P_t
  Eigen::MatrixXd P_t = Eigen::MatrixXd::Zero(_kf_parameters.dim_x, _kf_parameters.dim_x);
  this->kalman_filter_.getP(P_t);

  // Q matrix(__Process Noise Covariance Matrix__)
  Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(_kf_parameters.dim_x, _kf_parameters.dim_x);
  Q(IDX::X, IDX::X) = _kf_parameters.q_cov_p;
  Q(IDX::Y, IDX::Y) = _kf_parameters.q_cov_p;
  Q(IDX::Z, IDX::Z) = _kf_parameters.q_cov_p;
  Q(IDX::Yaw, IDX::Yaw) = _kf_parameters.q_cov_pyaw;
  Q(IDX::L, IDX::L) = _kf_parameters.q_cov_d;
  Q(IDX::W, IDX::W) = _kf_parameters.q_cov_d;
  Q(IDX::H, IDX::H) = _kf_parameters.q_cov_d;
  Q(IDX::VX, IDX::VX) = _kf_parameters.q_cov_v;
  Q(IDX::VY, IDX::VY) = _kf_parameters.q_cov_v;
  Q(IDX::VZ, IDX::VZ) = _kf_parameters.q_cov_v;
  Q(IDX::VYaw, IDX::VYaw) = _kf_parameters.q_cov_vyaw;

  // prediction
  if (!this->kalman_filter_.predict(u, A, B, Q)) {
    std::cerr << "Cannot predict" << std::endl;
  }
}

void STrack::load_parameters(const std::string & path)
{
  YAML::Node config = YAML::LoadFile(path);
  // initialize ekf params
  _kf_parameters.dim_x = config["dim_x"].as<int>();
  _kf_parameters.dim_z = config["dim_z"].as<int>();
  _kf_parameters.q_cov_p = config["q_cov_p"].as<float>();
  _kf_parameters.q_cov_pyaw = config["q_cov_pyaw"].as<float>();
  _kf_parameters.q_cov_d = config["q_cov_d"].as<float>();
  _kf_parameters.q_cov_v = config["q_cov_v"].as<float>();
  _kf_parameters.q_cov_vyaw = config["q_cov_vyaw"].as<float>();
  _kf_parameters.r_cov_p = config["r_cov_p"].as<float>();
  _kf_parameters.r_cov_pyaw = config["r_cov_pyaw"].as<float>();
  _kf_parameters.r_cov_d = config["r_cov_d"].as<float>();
  _kf_parameters.p0_cov_p = config["p0_cov_p"].as<float>();
  _kf_parameters.p0_cov_pyaw = config["p0_cov_pyaw"].as<float>();
  _kf_parameters.p0_cov_d = config["p0_cov_d"].as<float>();
  _kf_parameters.p0_cov_v = config["p0_cov_v"].as<float>();
  _kf_parameters.p0_cov_vyaw = config["p0_cov_vyaw"].as<float>();

  _kf_parameters.dt = config["dt"].as<float>();
}
