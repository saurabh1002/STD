#include "STDdesc.h"

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <pcl/common/io.h>
#include <pcl/kdtree/kdtree_flann.h>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/StdVector>
#include <algorithm>
#include <iostream>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "omp.h"

void down_sampling_voxel(std::vector<Eigen::Vector3d> &pl_feat, const double voxel_size) {
    if (voxel_size < 0.01) {
        return;
    }
    auto Discretize = [&](const Eigen::Vector3d &p) -> VOXEL_LOC {
        auto voxel = (p / voxel_size).array().floor().cast<int64_t>();
        return VOXEL_LOC(voxel.x(), voxel.y(), voxel.z());
    };

    std::unordered_map<VOXEL_LOC, M_POINT> voxel_map;
    std::for_each(pl_feat.cbegin(), pl_feat.cend(), [&](const Eigen::Vector3d &point) {
        VOXEL_LOC position = Discretize(point);
        auto iter = voxel_map.find(position);
        if (iter != voxel_map.end()) {
            iter->second.point += point;
            iter->second.count += 1;
        } else {
            M_POINT anp;
            anp.point = point;
            anp.intensity = 0;
            anp.count = 1;
            voxel_map[position] = anp;
        }
    });

    pl_feat.resize(voxel_map.size());
    std::transform(voxel_map.cbegin(), voxel_map.cend(), pl_feat.begin(),
                   [](const auto &pair) { return pair.second.point / pair.second.count; });
}

pcl::PointCloud<pcl::PointXYZI>::Ptr EigenToPCL(const std::vector<Eigen::Vector3d> &pointcloud) {
    pcl::PointCloud<pcl::PointXYZI>::Ptr pcl(
        new pcl::PointCloud<pcl::PointXYZI>(pointcloud.size(), 1));
    std::transform(pointcloud.cbegin(), pointcloud.cend(), pcl->begin(),
                   [&](const Eigen::Vector3d &point_eigen) { return vec2point(point_eigen); });
    return pcl;
}

inline pcl::PointXYZI vec2point(const Eigen::Vector3d &vec) {
    pcl::PointXYZI pi;
    pi.x = vec[0];
    pi.y = vec[1];
    pi.z = vec[2];
    return pi;
}

inline Eigen::Vector3d point2vec(const pcl::PointXYZ &pi) {
    return Eigen::Vector3d(pi.x, pi.y, pi.z);
}

inline Eigen::Vector3d point2vec(const pcl::PointXYZI &pi) {
    return Eigen::Vector3d(pi.x, pi.y, pi.z);
}

inline Eigen::Vector3d point2vec(const pcl::PointXYZINormal &pi) {
    return Eigen::Vector3d(pi.x, pi.y, pi.z);
}

inline Eigen::Vector3d normal2vec(const pcl::PointXYZINormal &pi) {
    return Eigen::Vector3d(pi.normal_x, pi.normal_y, pi.normal_z);
}

void STDescManager::GenerateSTDescs(const pcl::PointCloud<pcl::PointXYZI>::Ptr &input_cloud,
                                    std::vector<STDesc> &stds_vec) {
    // step1, voxelization and plane dection
    std::unordered_map<VOXEL_LOC, OctoTree *> voxel_map;
    init_voxel_map(input_cloud, voxel_map);
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr plane_cloud(
        new pcl::PointCloud<pcl::PointXYZINormal>);
    getPlane(voxel_map, plane_cloud);
    plane_cloud_vec_.emplace_back(plane_cloud);

    // step2, build connection for planes in the voxel map
    build_connection(voxel_map);

    // step3, extraction corner points
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr corner_points(
        new pcl::PointCloud<pcl::PointXYZINormal>);
    corner_extractor(voxel_map, corner_points);
    corner_cloud_vec_.emplace_back(corner_points);

    // step4, generate stable triangle descriptors
    stds_vec.clear();
    build_stdesc(corner_points, stds_vec);

    // step5, clear memory
    std::for_each(voxel_map.begin(), voxel_map.end(), [](auto &pair) { delete (pair.second); });
}

void STDescManager::SearchLoop(const std::vector<STDesc> &stds_vec) {
    if (stds_vec.size() == 0) {
        std::cerr << "No STDescs!" << std::endl;
        return;
    }
    // step1, select candidates, default number 50
    std::vector<STDMatchList> candidate_matcher_vec;
    candidate_selector(stds_vec, candidate_matcher_vec);

    loop_match_ids_.resize(candidate_matcher_vec.size());
    loop_match_scores_.resize(candidate_matcher_vec.size());
    loop_trs_.resize(candidate_matcher_vec.size());
    loop_rots_.resize(candidate_matcher_vec.size());
    // step2, select best candidates from rough candidates
    for (size_t i = 0; i < candidate_matcher_vec.size(); i++) {
        double verify_score = -1;
        std::pair<Eigen::Vector3d, Eigen::Matrix3d> relative_pose;
        std::vector<std::pair<STDesc, STDesc>> success_match_vec;
        candidate_verify(candidate_matcher_vec[i], verify_score, relative_pose, success_match_vec);
        loop_match_ids_[i] = candidate_matcher_vec[i].match_id_.second;
        loop_match_scores_[i] = verify_score;
        loop_rots_[i] = relative_pose.second;
        loop_trs_[i] = relative_pose.first;
    }
}

void STDescManager::AddSTDescs(const std::vector<STDesc> &stds_vec) {
    // update frame id
    current_frame_id_++;
    std::for_each(stds_vec.cbegin(), stds_vec.cend(), [&](const STDesc &single_std) {
        // calculate the position of single std
        STDesc_LOC position;
        position.x = (int)(single_std.side_length_[0] + 0.5);
        position.y = (int)(single_std.side_length_[1] + 0.5);
        position.z = (int)(single_std.side_length_[2] + 0.5);
        position.a = (int)(single_std.angle_[0]);
        position.b = (int)(single_std.angle_[1]);
        position.c = (int)(single_std.angle_[2]);
        auto iter = data_base_.find(position);
        if (iter != data_base_.end()) {
            data_base_[position].emplace_back(single_std);
        } else {
            data_base_[position] = std::vector<STDesc>{single_std};
        }
    });
}

void STDescManager::init_voxel_map(const pcl::PointCloud<pcl::PointXYZI>::Ptr &input_cloud,
                                   std::unordered_map<VOXEL_LOC, OctoTree *> &voxel_map) {
    auto Discretize = [&](const Eigen::Vector3d &p) -> VOXEL_LOC {
        auto voxel = (p / config_setting_.voxel_size_).array().floor().cast<int64_t>();
        return VOXEL_LOC(voxel.x(), voxel.y(), voxel.z());
    };
    std::for_each(input_cloud->points.cbegin(), input_cloud->points.cend(),
                  [&](const pcl::PointXYZI &point) {
                      Eigen::Vector3d p_c = point2vec(point);
                      VOXEL_LOC position = Discretize(p_c);
                      auto iter = voxel_map.find(position);
                      if (iter != voxel_map.end()) {
                          voxel_map[position]->voxel_points_.emplace_back(p_c);
                      } else {
                          voxel_map[position] = new OctoTree(config_setting_);
                          voxel_map[position]->voxel_points_.emplace_back(p_c);
                      }
                  });

    std::for_each(voxel_map.begin(), voxel_map.end(),
                  [](auto &pair) { pair.second->init_octo_tree(); });
}

void STDescManager::build_connection(std::unordered_map<VOXEL_LOC, OctoTree *> &voxel_map) {
    std::for_each(voxel_map.cbegin(), voxel_map.cend(), [&](const auto &pair) {
        if (pair.second->plane_ptr_->is_plane_) {
            OctoTree *current_octo = pair.second;
            for (int i = 0; i < 6; i++) {
                VOXEL_LOC neighbor = pair.first;
                if (i == 0) {
                    neighbor.x = neighbor.x + 1;
                } else if (i == 1) {
                    neighbor.y = neighbor.y + 1;
                } else if (i == 2) {
                    neighbor.z = neighbor.z + 1;
                } else if (i == 3) {
                    neighbor.x = neighbor.x - 1;
                } else if (i == 4) {
                    neighbor.y = neighbor.y - 1;
                } else if (i == 5) {
                    neighbor.z = neighbor.z - 1;
                }
                auto near = voxel_map.find(neighbor);
                if (near == voxel_map.end()) {
                    current_octo->is_check_connect_[i] = true;
                    current_octo->connect_[i] = false;
                } else {
                    if (!current_octo->is_check_connect_[i]) {
                        OctoTree *near_octo = near->second;
                        current_octo->is_check_connect_[i] = true;
                        int j;
                        if (i >= 3) {
                            j = i - 3;
                        } else {
                            j = i + 3;
                        }
                        near_octo->is_check_connect_[j] = true;
                        if (near_octo->plane_ptr_->is_plane_) {
                            // merge near octo
                            Eigen::Vector3d normal_diff =
                                current_octo->plane_ptr_->normal_ - near_octo->plane_ptr_->normal_;
                            Eigen::Vector3d normal_add =
                                current_octo->plane_ptr_->normal_ + near_octo->plane_ptr_->normal_;
                            if (normal_diff.norm() < config_setting_.plane_merge_normal_thre_ ||
                                normal_add.norm() < config_setting_.plane_merge_normal_thre_) {
                                current_octo->connect_[i] = true;
                                near_octo->connect_[j] = true;
                                current_octo->connect_tree_[i] = near_octo;
                                near_octo->connect_tree_[j] = current_octo;
                            } else {
                                current_octo->connect_[i] = false;
                                near_octo->connect_[j] = false;
                            }
                        } else {
                            current_octo->connect_[i] = false;
                            near_octo->connect_[j] = true;
                            near_octo->connect_tree_[j] = current_octo;
                        }
                    }
                }
            }
        }
    });
}

void STDescManager::getPlane(const std::unordered_map<VOXEL_LOC, OctoTree *> &voxel_map,
                             pcl::PointCloud<pcl::PointXYZINormal>::Ptr &plane_cloud) {
    plane_cloud->reserve(voxel_map.size());
    std::for_each(voxel_map.cbegin(), voxel_map.cend(), [&](const auto &pair) {
        if (pair.second->plane_ptr_->is_plane_) {
            pcl::PointXYZINormal pi;
            pi.x = pair.second->plane_ptr_->center_[0];
            pi.y = pair.second->plane_ptr_->center_[1];
            pi.z = pair.second->plane_ptr_->center_[2];
            pi.normal_x = pair.second->plane_ptr_->normal_[0];
            pi.normal_y = pair.second->plane_ptr_->normal_[1];
            pi.normal_z = pair.second->plane_ptr_->normal_[2];
            plane_cloud->emplace_back(pi);
        }
    });
}

void STDescManager::corner_extractor(std::unordered_map<VOXEL_LOC, OctoTree *> &voxel_map,
                                     pcl::PointCloud<pcl::PointXYZINormal>::Ptr &corner_points) {
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr prepare_corner_points(
        new pcl::PointCloud<pcl::PointXYZINormal>);

    // Avoid inconsistent voxel cutting caused by different view point
    std::vector<Eigen::Vector3i> voxel_round;
    voxel_round.reserve(27);
    for (int x = -1; x <= 1; x++) {
        for (int y = -1; y <= 1; y++) {
            for (int z = -1; z <= 1; z++) {
                Eigen::Vector3i voxel_inc(x, y, z);
                voxel_round.emplace_back(voxel_inc);
            }
        }
    }
    std::for_each(voxel_map.begin(), voxel_map.end(), [&](auto &pair) {
        if (!pair.second->plane_ptr_->is_plane_) {
            VOXEL_LOC current_position = pair.first;
            OctoTree *current_octo = pair.second;
            int connect_index = -1;
            for (int i = 0; i < 6; i++) {
                if (current_octo->connect_[i]) {
                    connect_index = i;
                    OctoTree *connect_octo = current_octo->connect_tree_[connect_index];
                    bool use = false;
                    for (int j = 0; j < 6; j++) {
                        if (connect_octo->is_check_connect_[j]) {
                            if (connect_octo->connect_[j]) {
                                use = true;
                            }
                        }
                    }
                    // if no plane near the voxel, skip
                    if (use == false) {
                        continue;
                    }
                    // only project voxels with points num > 10
                    if (current_octo->voxel_points_.size() > 10) {
                        Eigen::Vector3d projection_normal =
                            current_octo->connect_tree_[connect_index]->plane_ptr_->normal_;
                        Eigen::Vector3d projection_center =
                            current_octo->connect_tree_[connect_index]->plane_ptr_->center_;
                        std::vector<Eigen::Vector3d> proj_points;
                        // proj the boundary voxel and nearby voxel onto adjacent plane
                        for (auto voxel_inc : voxel_round) {
                            VOXEL_LOC connect_project_position = current_position;
                            connect_project_position.x += voxel_inc[0];
                            connect_project_position.y += voxel_inc[1];
                            connect_project_position.z += voxel_inc[2];
                            auto iter_near = voxel_map.find(connect_project_position);
                            if (iter_near != voxel_map.end()) {
                                bool skip_flag = false;
                                if (!voxel_map[connect_project_position]->plane_ptr_->is_plane_) {
                                    if (voxel_map[connect_project_position]->is_project_) {
                                        for (auto normal : voxel_map[connect_project_position]
                                                               ->proj_normal_vec_) {
                                            Eigen::Vector3d normal_diff =
                                                projection_normal - normal;
                                            Eigen::Vector3d normal_add = projection_normal + normal;
                                            // check if repeated project
                                            if (normal_diff.norm() < 0.5 ||
                                                normal_add.norm() < 0.5) {
                                                skip_flag = true;
                                            }
                                        }
                                    }
                                    if (skip_flag) {
                                        continue;
                                    }
                                    std::for_each(
                                        voxel_map[connect_project_position]->voxel_points_.cbegin(),
                                        voxel_map[connect_project_position]->voxel_points_.cend(),
                                        [&](const auto &point) {
                                            proj_points.emplace_back(point);
                                            voxel_map[connect_project_position]
                                                ->proj_normal_vec_.emplace_back(projection_normal);
                                        });
                                    voxel_map[connect_project_position]->is_project_ = true;
                                }
                            }
                        }
                        // here do the 2D projection and corner extraction
                        extract_corner(projection_center, projection_normal, proj_points,
                                       prepare_corner_points);
                    }
                }
            }
        }
    });
    non_maxi_suppression(prepare_corner_points);

    if (config_setting_.maximum_corner_num_ > prepare_corner_points->size()) {
        corner_points = prepare_corner_points;
    } else {
        std::vector<std::pair<double, int>> attach_vec;
        attach_vec.reserve(prepare_corner_points->size());
        for (size_t i = 0; i < prepare_corner_points->size(); i++) {
            attach_vec.emplace_back(prepare_corner_points->points[i].intensity, i);
        }
        std::sort(attach_vec.begin(), attach_vec.end(),
                  [&](std::pair<double, int> a, std::pair<double, int> b) {
                      return (a.first > b.first);
                  });
        for (size_t i = 0; i < config_setting_.maximum_corner_num_; i++) {
            corner_points->points.emplace_back(prepare_corner_points->points[attach_vec[i].second]);
        }
    }
}

void STDescManager::extract_corner(const Eigen::Vector3d &proj_center,
                                   const Eigen::Vector3d proj_normal,
                                   const std::vector<Eigen::Vector3d> proj_points,
                                   pcl::PointCloud<pcl::PointXYZINormal>::Ptr &corner_points) {
    double resolution = config_setting_.proj_image_resolution_;
    double dis_threshold_min = config_setting_.proj_dis_min_;
    double dis_threshold_max = config_setting_.proj_dis_max_;
    double A = proj_normal[0];
    double B = proj_normal[1];
    double C = proj_normal[2];
    double D = -(A * proj_center[0] + B * proj_center[1] + C * proj_center[2]);
    Eigen::Vector3d x_axis(1, 1, 0);
    if (C != 0) {
        x_axis[2] = -(A + B) / C;
    } else if (B != 0) {
        x_axis[1] = -A / B;
    } else {
        x_axis[0] = 0;
        x_axis[1] = 1;
    }
    x_axis.normalize();
    Eigen::Vector3d y_axis = proj_normal.cross(x_axis);
    y_axis.normalize();
    double ax = x_axis[0];
    double bx = x_axis[1];
    double cx = x_axis[2];
    double dx = -(ax * proj_center[0] + bx * proj_center[1] + cx * proj_center[2]);
    double ay = y_axis[0];
    double by = y_axis[1];
    double cy = y_axis[2];
    double dy = -(ay * proj_center[0] + by * proj_center[1] + cy * proj_center[2]);
    std::vector<Eigen::Vector2d> point_list_2d;
    point_list_2d.reserve(proj_points.size());
    std::for_each(proj_points.cbegin(), proj_points.cend(), [&](const auto &point) {
        double x = point[0];
        double y = point[1];
        double z = point[2];
        double dis = fabs(x * A + y * B + z * C + D);
        if (dis >= dis_threshold_min && dis <= dis_threshold_max) {
            Eigen::Vector3d cur_project;
            cur_project[0] =
                (-A * (B * y + C * z + D) + x * (B * B + C * C)) / (A * A + B * B + C * C);
            cur_project[1] =
                (-B * (A * x + C * z + D) + y * (A * A + C * C)) / (A * A + B * B + C * C);
            cur_project[2] =
                (-C * (A * x + B * y + D) + z * (A * A + B * B)) / (A * A + B * B + C * C);

            pcl::PointXYZ p;
            p.x = cur_project[0];
            p.y = cur_project[1];
            p.z = cur_project[2];
            double project_x = cur_project[0] * ay + cur_project[1] * by + cur_project[2] * cy + dy;
            double project_y = cur_project[0] * ax + cur_project[1] * bx + cur_project[2] * cx + dx;
            point_list_2d.emplace_back(project_x, project_y);
        }
    });
    point_list_2d.shrink_to_fit();
    if (point_list_2d.size() <= 5) {
        return;
    }

    double min_x = 10;
    double max_x = -10;
    double min_y = 10;
    double max_y = -10;
    for (auto pi : point_list_2d) {
        if (pi[0] < min_x) {
            min_x = pi[0];
        }
        if (pi[0] > max_x) {
            max_x = pi[0];
        }
        if (pi[1] < min_y) {
            min_y = pi[1];
        }
        if (pi[1] > max_y) {
            max_y = pi[1];
        }
    }
    // segment project cloud with a fixed resolution
    int segmen_base_num = 5;
    double segmen_len = segmen_base_num * resolution;
    int x_segment_num = (max_x - min_x) / segmen_len + 1;
    int y_segment_num = (max_y - min_y) / segmen_len + 1;
    int x_axis_len = (int)((max_x - min_x) / resolution + segmen_base_num);
    int y_axis_len = (int)((max_y - min_y) / resolution + segmen_base_num);
    std::vector<Eigen::Vector2d> img_container[x_axis_len][y_axis_len];
    double img_count_array[x_axis_len][y_axis_len] = {0};
    double gradient_array[x_axis_len][y_axis_len] = {0};
    double mean_x_array[x_axis_len][y_axis_len] = {0};
    double mean_y_array[x_axis_len][y_axis_len] = {0};
    for (int x = 0; x < x_axis_len; x++) {
        for (int y = 0; y < y_axis_len; y++) {
            img_count_array[x][y] = 0;
            mean_x_array[x][y] = 0;
            mean_y_array[x][y] = 0;
            gradient_array[x][y] = 0;
            std::vector<Eigen::Vector2d> single_container;
            img_container[x][y] = single_container;
        }
    }
    for (size_t i = 0; i < point_list_2d.size(); i++) {
        int x_index = (int)((point_list_2d[i][0] - min_x) / resolution);
        int y_index = (int)((point_list_2d[i][1] - min_y) / resolution);
        mean_x_array[x_index][y_index] += point_list_2d[i][0];
        mean_y_array[x_index][y_index] += point_list_2d[i][1];
        img_count_array[x_index][y_index]++;
        img_container[x_index][y_index].push_back(point_list_2d[i]);
    }
    // calc gradient
    for (int x = 0; x < x_axis_len; x++) {
        for (int y = 0; y < y_axis_len; y++) {
            double gradient = 0;
            int cnt = 0;
            int inc = 1;
            for (int x_inc = -inc; x_inc <= inc; x_inc++) {
                for (int y_inc = -inc; y_inc <= inc; y_inc++) {
                    int xx = x + x_inc;
                    int yy = y + y_inc;
                    if (xx >= 0 && xx < x_axis_len && yy >= 0 && yy < y_axis_len) {
                        if (xx != x || yy != y) {
                            if (img_count_array[xx][yy] >= 0) {
                                gradient += img_count_array[x][y] - img_count_array[xx][yy];
                                cnt++;
                            }
                        }
                    }
                }
            }
            if (cnt != 0) {
                gradient_array[x][y] = gradient * 1.0 / cnt;
            } else {
                gradient_array[x][y] = 0;
            }
        }
    }
    // extract corner by gradient
    std::vector<int> max_gradient_vec;
    max_gradient_vec.reserve(x_segment_num * y_segment_num);
    std::vector<int> max_gradient_x_index_vec;
    max_gradient_x_index_vec.reserve(x_segment_num * y_segment_num);
    std::vector<int> max_gradient_y_index_vec;
    max_gradient_y_index_vec.reserve(x_segment_num * y_segment_num);
    for (int x_segment_index = 0; x_segment_index < x_segment_num; x_segment_index++) {
        for (int y_segment_index = 0; y_segment_index < y_segment_num; y_segment_index++) {
            double max_gradient = 0;
            int max_gradient_x_index = -10;
            int max_gradient_y_index = -10;
            for (int x_index = x_segment_index * segmen_base_num;
                 x_index < (x_segment_index + 1) * segmen_base_num; x_index++) {
                for (int y_index = y_segment_index * segmen_base_num;
                     y_index < (y_segment_index + 1) * segmen_base_num; y_index++) {
                    if (img_count_array[x_index][y_index] > max_gradient) {
                        max_gradient = img_count_array[x_index][y_index];
                        max_gradient_x_index = x_index;
                        max_gradient_y_index = y_index;
                    }
                }
            }
            if (max_gradient >= config_setting_.corner_thre_) {
                max_gradient_vec.emplace_back(max_gradient);
                max_gradient_x_index_vec.emplace_back(max_gradient_x_index);
                max_gradient_y_index_vec.emplace_back(max_gradient_y_index);
            }
        }
    }
    // filter out line
    // calc line or not
    std::vector<Eigen::Vector2i> direction_list = {{0, 1}, {1, 0}, {1, 1}, {1, -1}};
    corner_points->reserve(max_gradient_vec.size());
    for (size_t i = 0; i < max_gradient_vec.size(); i++) {
        for (const auto &direction : direction_list) {
            Eigen::Vector2i p(max_gradient_x_index_vec[i], max_gradient_y_index_vec[i]);
            Eigen::Vector2i p1 = p + direction;
            Eigen::Vector2i p2 = p - direction;
            int threshold = img_count_array[p[0]][p[1]] / 2;
            if (img_count_array[p1[0]][p1[1]] >= threshold &&
                img_count_array[p2[0]][p2[1]] >= threshold) {
            } else {
                continue;
            }
        }
        double px = mean_x_array[max_gradient_x_index_vec[i]][max_gradient_y_index_vec[i]] /
                    img_count_array[max_gradient_x_index_vec[i]][max_gradient_y_index_vec[i]];
        double py = mean_y_array[max_gradient_x_index_vec[i]][max_gradient_y_index_vec[i]] /
                    img_count_array[max_gradient_x_index_vec[i]][max_gradient_y_index_vec[i]];
        // reproject on 3D space
        Eigen::Vector3d coord = py * x_axis + px * y_axis + proj_center;
        pcl::PointXYZINormal pi;
        pi.x = coord[0];
        pi.y = coord[1];
        pi.z = coord[2];
        pi.intensity = max_gradient_vec[i];
        pi.normal_x = proj_normal[0];
        pi.normal_y = proj_normal[1];
        pi.normal_z = proj_normal[2];
        corner_points->points.emplace_back(pi);
    }
}

void STDescManager::non_maxi_suppression(
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr &corner_points) {
    std::vector<bool> is_add_vec;
    is_add_vec.reserve(corner_points->size());
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr prepare_key_cloud(
        new pcl::PointCloud<pcl::PointXYZINormal>);
    prepare_key_cloud->reserve(corner_points->size());
    for (auto pi : corner_points->points) {
        prepare_key_cloud->push_back(pi);
        is_add_vec.push_back(true);
    }
    pcl::KdTreeFLANN<pcl::PointXYZINormal> kd_tree;
    kd_tree.setInputCloud(prepare_key_cloud);
    std::vector<int> pointIdxRadiusSearch;
    std::vector<float> pointRadiusSquaredDistance;
    double radius = config_setting_.non_max_suppression_radius_;
    for (size_t i = 0; i < prepare_key_cloud->size(); i++) {
        pcl::PointXYZINormal searchPoint = prepare_key_cloud->points[i];
        if (kd_tree.radiusSearch(searchPoint, radius, pointIdxRadiusSearch,
                                 pointRadiusSquaredDistance) > 0) {
            for (const auto &search_index : pointIdxRadiusSearch) {
                if (search_index == i) {
                    continue;
                }
                if (prepare_key_cloud->points[i].intensity <=
                    prepare_key_cloud->points[search_index].intensity) {
                    is_add_vec[i] = false;
                }
            }
        }
    }
    corner_points->clear();
    corner_points->reserve(is_add_vec.size());
    for (size_t i = 0; i < is_add_vec.size(); i++) {
        if (is_add_vec[i]) {
            corner_points->points.push_back(prepare_key_cloud->points[i]);
        }
    }
    corner_points->points.shrink_to_fit();
}

void STDescManager::build_stdesc(const pcl::PointCloud<pcl::PointXYZINormal>::Ptr &corner_points,
                                 std::vector<STDesc> &stds_vec) {
    double scale = 1.0 / config_setting_.std_side_resolution_;
    int near_num = config_setting_.descriptor_near_num_;
    double max_dis_threshold = config_setting_.descriptor_max_len_;
    double min_dis_threshold = config_setting_.descriptor_min_len_;
    stds_vec.clear();
    stds_vec.reserve(corner_points->size() * (near_num - 1) * (near_num - 1));
    std::unordered_map<VOXEL_LOC, bool> feat_map;
    pcl::KdTreeFLANN<pcl::PointXYZINormal>::Ptr kd_tree(new pcl::KdTreeFLANN<pcl::PointXYZINormal>);
    kd_tree->setInputCloud(corner_points);
    std::vector<int> pointIdxNKNSearch(near_num);
    std::vector<float> pointNKNSquaredDistance(near_num);
    // Search N nearest corner points to form stds.
    std::for_each(corner_points->cbegin(), corner_points->cend(), [&](const auto &searchPoint) {
        if (kd_tree->nearestKSearch(searchPoint, near_num, pointIdxNKNSearch,
                                    pointNKNSquaredDistance) > 0) {
            for (int m = 1; m < near_num - 1; m++) {
                for (int n = m + 1; n < near_num; n++) {
                    pcl::PointXYZINormal p1 = searchPoint;
                    pcl::PointXYZINormal p2 = corner_points->points[pointIdxNKNSearch[m]];
                    pcl::PointXYZINormal p3 = corner_points->points[pointIdxNKNSearch[n]];
                    Eigen::Vector3d normal_inc1 = normal2vec(p1) - normal2vec(p2);
                    Eigen::Vector3d normal_inc2 = normal2vec(p3) - normal2vec(p2);
                    Eigen::Vector3d normal_add1 = normal2vec(p1) + normal2vec(p2);
                    Eigen::Vector3d normal_add2 = normal2vec(p3) + normal2vec(p2);
                    double a =
                        sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2) + pow(p1.z - p2.z, 2));
                    double b =
                        sqrt(pow(p1.x - p3.x, 2) + pow(p1.y - p3.y, 2) + pow(p1.z - p3.z, 2));
                    double c =
                        sqrt(pow(p3.x - p2.x, 2) + pow(p3.y - p2.y, 2) + pow(p3.z - p2.z, 2));
                    if (a > max_dis_threshold || b > max_dis_threshold || c > max_dis_threshold ||
                        a < min_dis_threshold || b < min_dis_threshold || c < min_dis_threshold) {
                        continue;
                    }
                    // re-range the vertex by the side length
                    double temp;
                    Eigen::Vector3d A, B, C;
                    Eigen::Vector3i l1, l2, l3;
                    Eigen::Vector3i l_temp;
                    l1 << 1, 2, 0;
                    l2 << 1, 0, 3;
                    l3 << 0, 2, 3;
                    if (a > b) {
                        temp = a;
                        a = b;
                        b = temp;
                        l_temp = l1;
                        l1 = l2;
                        l2 = l_temp;
                    }
                    if (b > c) {
                        temp = b;
                        b = c;
                        c = temp;
                        l_temp = l2;
                        l2 = l3;
                        l3 = l_temp;
                    }
                    if (a > b) {
                        temp = a;
                        a = b;
                        b = temp;
                        l_temp = l1;
                        l1 = l2;
                        l2 = l_temp;
                    }
                    // check augnmentation
                    pcl::PointXYZ d_p;
                    d_p.x = a * 1000;
                    d_p.y = b * 1000;
                    d_p.z = c * 1000;
                    VOXEL_LOC position((int64_t)d_p.x, (int64_t)d_p.y, (int64_t)d_p.z);
                    auto iter = feat_map.find(position);
                    Eigen::Vector3d normal_1, normal_2, normal_3;
                    if (iter == feat_map.end()) {
                        Eigen::Vector3d vertex_attached;
                        if (l1[0] == l2[0]) {
                            A = point2vec(p1);
                            normal_1 = normal2vec(p1);
                            vertex_attached[0] = p1.intensity;
                        } else if (l1[1] == l2[1]) {
                            A = point2vec(p2);
                            normal_1 = normal2vec(p2);
                            ;
                            vertex_attached[0] = p2.intensity;
                        } else {
                            A = point2vec(p3);
                            normal_1 = normal2vec(p3);
                            ;
                            vertex_attached[0] = p3.intensity;
                        }
                        if (l1[0] == l3[0]) {
                            B = point2vec(p1);
                            normal_2 = normal2vec(p1);
                            vertex_attached[1] = p1.intensity;
                        } else if (l1[1] == l3[1]) {
                            B = point2vec(p2);
                            normal_2 = normal2vec(p2);
                            vertex_attached[1] = p2.intensity;
                        } else {
                            B = point2vec(p3);
                            normal_2 = normal2vec(p3);
                            vertex_attached[1] = p3.intensity;
                        }
                        if (l2[0] == l3[0]) {
                            C = point2vec(p1);
                            normal_3 = normal2vec(p1);
                            vertex_attached[2] = p1.intensity;
                        } else if (l2[1] == l3[1]) {
                            C = point2vec(p2);
                            normal_3 = normal2vec(p2);
                            vertex_attached[2] = p2.intensity;
                        } else {
                            C = point2vec(p3);
                            normal_3 = normal2vec(p3);
                            vertex_attached[2] = p3.intensity;
                        }
                        STDesc single_descriptor;
                        single_descriptor.vertex_A_ = A;
                        single_descriptor.vertex_B_ = B;
                        single_descriptor.vertex_C_ = C;
                        single_descriptor.center_ = (A + B + C) / 3;
                        single_descriptor.vertex_attached_ = vertex_attached;
                        single_descriptor.side_length_ << scale * a, scale * b, scale * c;
                        single_descriptor.angle_[0] = fabs(5 * normal_1.dot(normal_2));
                        single_descriptor.angle_[1] = fabs(5 * normal_1.dot(normal_3));
                        single_descriptor.angle_[2] = fabs(5 * normal_3.dot(normal_2));
                        // single_descriptor.angle << 0, 0, 0;
                        single_descriptor.frame_id_ = current_frame_id_;
                        feat_map[position] = true;
                        stds_vec.emplace_back(single_descriptor);
                    }
                }
            }
        }
    });
}

void STDescManager::candidate_selector(const std::vector<STDesc> &stds_vec,
                                       std::vector<STDMatchList> &candidate_matcher_vec) {
    double match_array[MAX_FRAME_N] = {0};
    std::vector<Eigen::Vector3i> voxel_round;
    voxel_round.reserve(27);
    for (int x = -1; x <= 1; x++) {
        for (int y = -1; y <= 1; y++) {
            for (int z = -1; z <= 1; z++) {
                voxel_round.emplace_back(x, y, z);
            }
        }
    }

    std::vector<bool> useful_match(stds_vec.size(), false);
    std::vector<std::vector<size_t>> useful_match_index(stds_vec.size());
    std::vector<std::vector<STDesc_LOC>> useful_match_position(stds_vec.size());
    // speed up matching
#ifdef MP_EN
    omp_set_num_threads(MP_PROC_NUM);
#pragma omp parallel for
#endif
    for (size_t i = 0; i < stds_vec.size(); i++) {
        STDesc src_std = stds_vec[i];
        STDesc_LOC position;
        STDesc_LOC best_position;
        double dis_threshold = src_std.side_length_.norm() * config_setting_.rough_dis_threshold_;
        for (auto voxel_inc : voxel_round) {
            position.x = (int)(src_std.side_length_[0] + voxel_inc[0]);
            position.y = (int)(src_std.side_length_[1] + voxel_inc[1]);
            position.z = (int)(src_std.side_length_[2] + voxel_inc[2]);
            Eigen::Vector3d voxel_center((double)position.x + 0.5, (double)position.y + 0.5,
                                         (double)position.z + 0.5);
            if ((src_std.side_length_ - voxel_center).norm() < 1.5) {
                auto iter = data_base_.find(position);
                if (iter != data_base_.end()) {
                    for (size_t j = 0; j < data_base_[position].size(); j++) {
                        if ((src_std.frame_id_ - data_base_[position][j].frame_id_) >
                            config_setting_.skip_near_num_) {
                            double dis =
                                (src_std.side_length_ - data_base_[position][j].side_length_)
                                    .norm();
                            // rough filter with side lengths
                            if (dis < dis_threshold) {
                                // rough filter with vertex attached info
                                double vertex_attach_diff =
                                    2.0 *
                                    (src_std.vertex_attached_ -
                                     data_base_[position][j].vertex_attached_)
                                        .norm() /
                                    (src_std.vertex_attached_ +
                                     data_base_[position][j].vertex_attached_)
                                        .norm();
                                if (vertex_attach_diff < config_setting_.vertex_diff_threshold_) {
                                    useful_match[i] = true;
                                    useful_match_position[i].emplace_back(position);
                                    useful_match_index[i].emplace_back(j);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // record match index
    std::vector<int> match_index_vec;
    std::vector<Eigen::Vector2i, Eigen::aligned_allocator<Eigen::Vector2i>> index_recorder;
    for (size_t i = 0; i < useful_match.size(); i++) {
        if (useful_match[i]) {
            for (size_t j = 0; j < useful_match_index[i].size(); j++) {
                match_array[data_base_[useful_match_position[i][j]][useful_match_index[i][j]]
                                .frame_id_] += 1;
                index_recorder.emplace_back(i, j);
                match_index_vec.emplace_back(
                    data_base_[useful_match_position[i][j]][useful_match_index[i][j]].frame_id_);
            }
        }
    }

    // select candidate according to the matching score
    for (int cnt = 0; cnt < config_setting_.candidate_num_; cnt++) {
        double max_vote = 1;
        int max_vote_index = -1;
        for (int i = 0; i < MAX_FRAME_N; i++) {
            if (match_array[i] > max_vote) {
                max_vote = match_array[i];
                max_vote_index = i;
            }
        }
        if (max_vote_index >= 0 && max_vote >= 5) {
            STDMatchList match_triangle_list;
            match_triangle_list.match_list_.reserve(index_recorder.size());
            match_triangle_list.match_id_.first = current_frame_id_;
            match_triangle_list.match_id_.second = max_vote_index;
            match_array[max_vote_index] = 0;
            for (size_t i = 0; i < index_recorder.size(); i++) {
                if (match_index_vec[i] == max_vote_index) {
                    match_triangle_list.match_list_.emplace_back(
                        stds_vec[index_recorder[i][0]],
                        data_base_[useful_match_position[index_recorder[i][0]]
                                                        [index_recorder[i][1]]]
                                  [useful_match_index[index_recorder[i][0]][index_recorder[i][1]]]);
                }
            }
            candidate_matcher_vec.emplace_back(match_triangle_list);
        } else {
            break;
        }
    }
}

// Get the best candidate frame by geometry check
void STDescManager::candidate_verify(const STDMatchList &candidate_matcher,
                                     double &verify_score,
                                     std::pair<Eigen::Vector3d, Eigen::Matrix3d> &relative_pose,
                                     std::vector<std::pair<STDesc, STDesc>> &success_match_vec) {
    success_match_vec.clear();
    int skip_len = (int)(candidate_matcher.match_list_.size() / 50) + 1;
    int use_size = candidate_matcher.match_list_.size() / skip_len;
    double dis_threshold = 3.0;
    std::vector<int> vote_list(use_size);
    std::mutex mylock;

#ifdef MP_EN
    omp_set_num_threads(MP_PROC_NUM);
#pragma omp parallel for
#endif
    for (size_t i = 0; i < use_size; i++) {
        auto single_pair = candidate_matcher.match_list_[i * skip_len];
        int vote = 0;
        Eigen::Matrix3d test_rot;
        Eigen::Vector3d test_t;
        triangle_solver(single_pair, test_t, test_rot);
        std::for_each(
            candidate_matcher.match_list_.cbegin(), candidate_matcher.match_list_.cend(),
            [&](const auto &verify_pair) {
                Eigen::Vector3d A = verify_pair.first.vertex_A_;
                Eigen::Vector3d A_transform = test_rot * A + test_t;
                Eigen::Vector3d B = verify_pair.first.vertex_B_;
                Eigen::Vector3d B_transform = test_rot * B + test_t;
                Eigen::Vector3d C = verify_pair.first.vertex_C_;
                Eigen::Vector3d C_transform = test_rot * C + test_t;
                double dis_A = (A_transform - verify_pair.second.vertex_A_).norm();
                double dis_B = (B_transform - verify_pair.second.vertex_B_).norm();
                double dis_C = (C_transform - verify_pair.second.vertex_C_).norm();
                if (dis_A < dis_threshold && dis_B < dis_threshold && dis_C < dis_threshold) {
                    vote++;
                }
            });
        mylock.lock();
        vote_list[i] = vote;
        mylock.unlock();
    }
    auto max_vote_iter = std::max_element(vote_list.begin(), vote_list.end());
    int max_vote_index = std::distance(vote_list.begin(), max_vote_iter);
    int max_vote = *max_vote_iter;
    if (max_vote >= 4) {
        auto best_pair = candidate_matcher.match_list_[max_vote_index * skip_len];
        Eigen::Matrix3d best_rot;
        Eigen::Vector3d best_t;
        triangle_solver(best_pair, best_t, best_rot);
        relative_pose.first = best_t;
        relative_pose.second = best_rot;
        success_match_vec.reserve(candidate_matcher.match_list_.size());
        std::for_each(
            candidate_matcher.match_list_.cbegin(), candidate_matcher.match_list_.cend(),
            [&](const auto &verify_pair) {
                Eigen::Vector3d A = verify_pair.first.vertex_A_;
                Eigen::Vector3d A_transform = best_rot * A + best_t;
                Eigen::Vector3d B = verify_pair.first.vertex_B_;
                Eigen::Vector3d B_transform = best_rot * B + best_t;
                Eigen::Vector3d C = verify_pair.first.vertex_C_;
                Eigen::Vector3d C_transform = best_rot * C + best_t;
                double dis_A = (A_transform - verify_pair.second.vertex_A_).norm();
                double dis_B = (B_transform - verify_pair.second.vertex_B_).norm();
                double dis_C = (C_transform - verify_pair.second.vertex_C_).norm();
                if (dis_A < dis_threshold && dis_B < dis_threshold && dis_C < dis_threshold) {
                    success_match_vec.emplace_back(verify_pair);
                }
            });
        verify_score = plane_geometric_verify(plane_cloud_vec_.back(),
                                              plane_cloud_vec_[candidate_matcher.match_id_.second],
                                              relative_pose);
    } else {
        verify_score = -1;
    }
}

void STDescManager::triangle_solver(const std::pair<STDesc, STDesc> &std_pair,
                                    Eigen::Vector3d &t,
                                    Eigen::Matrix3d &rot) {
    Eigen::Matrix3d src = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d ref = Eigen::Matrix3d::Zero();
    src.col(0) = std_pair.first.vertex_A_ - std_pair.first.center_;
    src.col(1) = std_pair.first.vertex_B_ - std_pair.first.center_;
    src.col(2) = std_pair.first.vertex_C_ - std_pair.first.center_;
    ref.col(0) = std_pair.second.vertex_A_ - std_pair.second.center_;
    ref.col(1) = std_pair.second.vertex_B_ - std_pair.second.center_;
    ref.col(2) = std_pair.second.vertex_C_ - std_pair.second.center_;
    Eigen::Matrix3d covariance = src * ref.transpose();
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(covariance, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::Matrix3d V = svd.matrixV();
    Eigen::Matrix3d U = svd.matrixU();
    rot = V * U.transpose();
    if (rot.determinant() < 0) {
        Eigen::Matrix3d K;
        K << 1, 0, 0, 0, 1, 0, 0, 0, -1;
        rot = V * K * U.transpose();
    }
    t = -rot * std_pair.first.center_ + std_pair.second.center_;
}

double STDescManager::plane_geometric_verify(
    const pcl::PointCloud<pcl::PointXYZINormal>::Ptr &source_cloud,
    const pcl::PointCloud<pcl::PointXYZINormal>::Ptr &target_cloud,
    const std::pair<Eigen::Vector3d, Eigen::Matrix3d> &transform) {
    Eigen::Vector3d t = transform.first;
    Eigen::Matrix3d rot = transform.second;
    pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr kd_tree(new pcl::KdTreeFLANN<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    for (size_t i = 0; i < target_cloud->size(); i++) {
        pcl::PointXYZ pi;
        pi.x = target_cloud->points[i].x;
        pi.y = target_cloud->points[i].y;
        pi.z = target_cloud->points[i].z;
        input_cloud->push_back(pi);
    }
    kd_tree->setInputCloud(input_cloud);
    std::vector<int> pointIdxNKNSearch(1);
    std::vector<float> pointNKNSquaredDistance(1);
    double useful_match = 0;
    double normal_threshold = config_setting_.normal_threshold_;
    double dis_threshold = config_setting_.dis_threshold_;
    std::for_each(source_cloud->cbegin(), source_cloud->cend(), [&](const auto &searchPoint) {
        Eigen::Vector3d pi(searchPoint.x, searchPoint.y, searchPoint.z);
        pi = rot * pi + t;
        pcl::PointXYZ use_search_point;
        use_search_point.x = pi[0];
        use_search_point.y = pi[1];
        use_search_point.z = pi[2];
        Eigen::Vector3d ni(searchPoint.normal_x, searchPoint.normal_y, searchPoint.normal_z);
        ni = rot * ni;
        int K = 3;
        if (kd_tree->nearestKSearch(use_search_point, K, pointIdxNKNSearch,
                                    pointNKNSquaredDistance) > 0) {
            for (size_t j = 0; j < K; j++) {
                pcl::PointXYZINormal nearstPoint = target_cloud->points[pointIdxNKNSearch[j]];
                Eigen::Vector3d tpi = point2vec(nearstPoint);
                Eigen::Vector3d tni = normal2vec(nearstPoint);
                Eigen::Vector3d normal_inc = ni - tni;
                Eigen::Vector3d normal_add = ni + tni;
                double point_to_plane = fabs(tni.transpose() * (pi - tpi));
                if ((normal_inc.norm() < normal_threshold ||
                     normal_add.norm() < normal_threshold) &&
                    point_to_plane < dis_threshold) {
                    useful_match++;
                    break;
                }
            }
        }
    });
    return useful_match / source_cloud->size();
}

int STDescManager::ProcessNewScan(const std::vector<Eigen::Vector3d> &pcl) {
    pcl::PointCloud<pcl::PointXYZI>::Ptr current_cloud = EigenToPCL(pcl);

    std::vector<STDesc> stds_vec;
    this->GenerateSTDescs(current_cloud, stds_vec);

    if (keyCloudInd > config_setting_.skip_near_num_) {
        this->SearchLoop(stds_vec);
    }
    this->AddSTDescs(stds_vec);
    keyCloudInd++;
    return loop_match_ids_.size();
}

void STDescManager::AddToDatabase(const std::vector<Eigen::Vector3d> &pcl) {
    pcl::PointCloud<pcl::PointXYZI>::Ptr current_cloud = EigenToPCL(pcl);

    std::vector<STDesc> stds_vec;
    this->GenerateSTDescs(current_cloud, stds_vec);
    this->AddSTDescs(stds_vec);
    keyCloudInd++;
}

int STDescManager::ComputeClosure(const std::vector<Eigen::Vector3d> &pcl) {
    pcl::PointCloud<pcl::PointXYZI>::Ptr current_cloud = EigenToPCL(pcl);

    std::vector<STDesc> stds_vec;
    this->GenerateSTDescs(current_cloud, stds_vec);
    this->SearchLoop(stds_vec);
    return loop_match_ids_.size();
}

std::tuple<int, double, Eigen::Vector3d, Eigen::Matrix3d> STDescManager::GetClosureDataAtIdx(
    int idx) {
    return {loop_match_ids_[idx], loop_match_scores_[idx], loop_trs_[idx], loop_rots_[idx]};
}

void OctoTree::init_plane() {
    plane_ptr_->covariance_ = Eigen::Matrix3d::Zero();
    plane_ptr_->center_ = Eigen::Vector3d::Zero();
    plane_ptr_->normal_ = Eigen::Vector3d::Zero();
    plane_ptr_->points_size_ = voxel_points_.size();
    plane_ptr_->radius_ = 0;
    for (auto pi : voxel_points_) {
        plane_ptr_->covariance_ += pi * pi.transpose();
        plane_ptr_->center_ += pi;
    }
    plane_ptr_->center_ = plane_ptr_->center_ / plane_ptr_->points_size_;
    plane_ptr_->covariance_ = plane_ptr_->covariance_ / plane_ptr_->points_size_ -
                              plane_ptr_->center_ * plane_ptr_->center_.transpose();
    Eigen::EigenSolver<Eigen::Matrix3d> es(plane_ptr_->covariance_);
    Eigen::Matrix3cd evecs = es.eigenvectors();
    Eigen::Vector3cd evals = es.eigenvalues();
    Eigen::Vector3d evalsReal;
    evalsReal = evals.real();
    Eigen::Matrix3d::Index evalsMin, evalsMax;
    evalsReal.rowwise().sum().minCoeff(&evalsMin);
    evalsReal.rowwise().sum().maxCoeff(&evalsMax);
    if (evalsReal(evalsMin) < config_setting_.plane_detection_thre_) {
        plane_ptr_->normal_ << evecs.real()(0, evalsMin), evecs.real()(1, evalsMin),
            evecs.real()(2, evalsMin);
        plane_ptr_->min_eigen_value_ = evalsReal(evalsMin);
        plane_ptr_->radius_ = sqrt(evalsReal(evalsMax));
        plane_ptr_->is_plane_ = true;

        plane_ptr_->intercept_ = -(plane_ptr_->normal_(0) * plane_ptr_->center_(0) +
                                   plane_ptr_->normal_(1) * plane_ptr_->center_(1) +
                                   plane_ptr_->normal_(2) * plane_ptr_->center_(2));
        plane_ptr_->p_center_.x = plane_ptr_->center_(0);
        plane_ptr_->p_center_.y = plane_ptr_->center_(1);
        plane_ptr_->p_center_.z = plane_ptr_->center_(2);
        plane_ptr_->p_center_.normal_x = plane_ptr_->normal_(0);
        plane_ptr_->p_center_.normal_y = plane_ptr_->normal_(1);
        plane_ptr_->p_center_.normal_z = plane_ptr_->normal_(2);
    } else {
        plane_ptr_->is_plane_ = false;
    }
}

void OctoTree::init_octo_tree() {
    if (voxel_points_.size() > config_setting_.voxel_init_num_) {
        init_plane();
    }
}
