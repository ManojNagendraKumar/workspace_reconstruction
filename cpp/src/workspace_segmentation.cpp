// Workspace segmentation for a cobot table: turn a fused depth point cloud into separate object clusters.
//
// Pipeline (all in the robot base frame, metres):
//   1. crop box                      keep only the workspace volume (cheapest step, so it goes first)
//   2. voxel grid                    5 mm downsampling: fewer, evenly spaced points
//   3. RANSAC plane                  find the table (constrained to be horizontal) and remove it
//   4. statistical outlier removal   drop flying pixels and speckle, on the object points only
//   5. Euclidean clustering          kd tree region growing: one cluster per object
//
// Profiling showed outlier removal (a kd tree query for every point) took over 80% of the time when it ran
// on the whole cloud. Most points belong to the table, and RANSAC is already robust to outliers, so the
// optimized order removes the table first and only filters the remaining object points.
// Pass "baseline" as the 4th argument to run the textbook order (outlier removal before the plane).
//
// Usage: workspace_segmentation <input.pcd> <output_dir> [timing_repeats=20] [optimized|baseline]
// Writes <output_dir>/segmented.pcd (label: 0 = table, k >= 1 = cluster k, 4294967295 = unclustered),
//        <output_dir>/clusters.json and <output_dir>/timing.json.

#include <pcl/common/centroid.h>
#include <pcl/common/common.h>
#include <pcl/filters/crop_box.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/sac_segmentation.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

using Cloud = pcl::PointCloud<pcl::PointXYZ>;
using Clock = std::chrono::steady_clock;

struct Params {
  Eigen::Vector4f crop_min{0.20f, -0.50f, -0.05f, 1.0f};
  Eigen::Vector4f crop_max{1.00f, 0.50f, 0.40f, 1.0f};
  float leaf = 0.005f;            // voxel size
  int sor_k = 30;                 // neighbours for the outlier statistics
  double sor_std = 1.0;           // keep points within mean + 1 sigma
  double plane_dist = 0.006;      // RANSAC inlier threshold for the table
  double plane_eps_deg = 10.0;    // table normal must be within 10 degrees of +z
  int plane_iters = 200;
  double cluster_tol = 0.010;     // points closer than 1 cm join the same cluster
  int min_cluster = 40;
  int max_cluster = 30000;
  bool sor_after_plane = true;    // optimized order (see header comment)
};

struct Result {
  Cloud::Ptr filtered;                       // after crop, voxel and outlier removal
  pcl::PointIndices::Ptr table;              // indices into filtered
  Cloud::Ptr objects;                        // filtered minus table
  std::vector<pcl::PointIndices> clusters;   // indices into objects
  Eigen::Vector4f plane = Eigen::Vector4f::Zero();
  std::map<std::string, double> ms;          // time per stage
};

static double ms_since(Clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

Result run_pipeline(const Cloud::ConstPtr& input, const Params& P) {
  Result r;
  auto t = Clock::now();

  Cloud::Ptr cropped(new Cloud);
  pcl::CropBox<pcl::PointXYZ> crop;
  crop.setInputCloud(input);
  crop.setMin(P.crop_min);
  crop.setMax(P.crop_max);
  crop.filter(*cropped);
  r.ms["1_crop"] = ms_since(t);

  t = Clock::now();
  Cloud::Ptr voxel(new Cloud);
  pcl::VoxelGrid<pcl::PointXYZ> vg;
  vg.setInputCloud(cropped);
  vg.setLeafSize(P.leaf, P.leaf, P.leaf);
  vg.filter(*voxel);
  r.ms["2_voxel"] = ms_since(t);

  auto remove_outliers = [&P](const Cloud::Ptr& in) {
    Cloud::Ptr out(new Cloud);
    pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
    sor.setInputCloud(in);
    sor.setMeanK(P.sor_k);
    sor.setStddevMulThresh(P.sor_std);
    sor.filter(*out);
    return out;
  };

  if (!P.sor_after_plane) {
    t = Clock::now();
    r.filtered = remove_outliers(voxel);
    r.ms["3_outliers"] = ms_since(t);
  } else {
    r.filtered = voxel;
  }

  t = Clock::now();
  pcl::ModelCoefficients coeffs;
  r.table.reset(new pcl::PointIndices);
  pcl::SACSegmentation<pcl::PointXYZ> seg;
  seg.setOptimizeCoefficients(true);                    // least squares refit on the inliers
  seg.setModelType(pcl::SACMODEL_PERPENDICULAR_PLANE);  // plane perpendicular to the axis below
  seg.setAxis(Eigen::Vector3f(0.f, 0.f, 1.f));
  seg.setEpsAngle(P.plane_eps_deg * M_PI / 180.0);
  seg.setMethodType(pcl::SAC_RANSAC);
  seg.setDistanceThreshold(P.plane_dist);
  seg.setMaxIterations(P.plane_iters);
  seg.setInputCloud(r.filtered);
  seg.segment(*r.table, coeffs);
  if (coeffs.values.size() == 4)
    r.plane = Eigen::Vector4f(coeffs.values[0], coeffs.values[1], coeffs.values[2], coeffs.values[3]);
  r.objects.reset(new Cloud);
  pcl::ExtractIndices<pcl::PointXYZ> ex;
  ex.setInputCloud(r.filtered);
  ex.setIndices(r.table);
  ex.setNegative(true);
  ex.filter(*r.objects);
  r.ms["4_plane"] = ms_since(t);

  if (P.sor_after_plane) {
    t = Clock::now();
    r.objects = remove_outliers(r.objects);
    r.ms["3_outliers"] = ms_since(t);
  }

  t = Clock::now();
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
  tree->setInputCloud(r.objects);
  pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
  ec.setClusterTolerance(P.cluster_tol);
  ec.setMinClusterSize(P.min_cluster);
  ec.setMaxClusterSize(P.max_cluster);
  ec.setSearchMethod(tree);
  ec.setInputCloud(r.objects);
  ec.extract(r.clusters);
  r.ms["5_clusters"] = ms_since(t);

  double total = 0;
  for (const auto& kv : r.ms) total += kv.second;
  r.ms["total"] = total;
  return r;
}

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "usage: " << argv[0] << " <input.pcd> <output_dir> [timing_repeats] [baseline]\n";
    return 1;
  }
  const std::string in_path = argv[1], out_dir = argv[2];
  const int repeats = argc > 3 ? std::stoi(argv[3]) : 20;
  std::filesystem::create_directories(out_dir);

  pcl::PointCloud<pcl::PointXYZL>::Ptr labelled(new pcl::PointCloud<pcl::PointXYZL>);
  if (pcl::io::loadPCDFile(in_path, *labelled) < 0) return 2;
  Cloud::Ptr input(new Cloud);
  pcl::copyPointCloud(*labelled, *input);  // the pipeline never sees the ground truth labels

  Params P;
  if (argc > 4 && std::string(argv[4]) == "baseline") P.sor_after_plane = false;
  const Result r = run_pipeline(input, P);

  std::map<std::string, double> mean_ms;  // repeat the whole pipeline, report the mean per stage
  for (int i = 0; i < repeats; ++i) {
    const Result ri = run_pipeline(input, P);
    for (const auto& kv : ri.ms) mean_ms[kv.first] += kv.second / repeats;
  }

  pcl::PointCloud<pcl::PointXYZL> out;
  out.reserve(r.filtered->size());
  for (int idx : r.table->indices) {
    const auto& p = r.filtered->points[idx];
    pcl::PointXYZL q;
    q.x = p.x; q.y = p.y; q.z = p.z; q.label = 0;
    out.push_back(q);
  }
  std::vector<uint32_t> label_of(r.objects->size(), 4294967295u);
  for (size_t k = 0; k < r.clusters.size(); ++k)
    for (int idx : r.clusters[k].indices) label_of[idx] = static_cast<uint32_t>(k + 1);
  for (size_t i = 0; i < r.objects->size(); ++i) {
    const auto& p = r.objects->points[i];
    pcl::PointXYZL q;
    q.x = p.x; q.y = p.y; q.z = p.z; q.label = label_of[i];
    out.push_back(q);
  }
  pcl::io::savePCDFileBinary(out_dir + "/segmented.pcd", out);

  std::ofstream cj(out_dir + "/clusters.json");
  cj << "[\n";
  for (size_t k = 0; k < r.clusters.size(); ++k) {
    Cloud c;
    pcl::copyPointCloud(*r.objects, r.clusters[k], c);
    Eigen::Vector4f centroid;
    pcl::compute3DCentroid(c, centroid);
    pcl::PointXYZ mn, mx;
    pcl::getMinMax3D(c, mn, mx);
    cj << "  {\"id\": " << k + 1 << ", \"points\": " << c.size() << ", \"centroid\": [" << centroid[0] << ", "
       << centroid[1] << ", " << centroid[2] << "], \"aabb_min\": [" << mn.x << ", " << mn.y << ", " << mn.z
       << "], \"aabb_max\": [" << mx.x << ", " << mx.y << ", " << mx.z << "]}"
       << (k + 1 < r.clusters.size() ? ",\n" : "\n");
  }
  cj << "]\n";

  std::ofstream tj(out_dir + "/timing.json");
  tj << "{\n  \"mode\": \"" << (P.sor_after_plane ? "optimized" : "baseline") << "\",\n  \"input_points\": " << input->size() << ",\n  \"filtered_points\": " << r.filtered->size()
     << ",\n  \"table_points\": " << r.table->indices.size() << ",\n  \"clusters\": " << r.clusters.size()
     << ",\n  \"plane\": [" << r.plane[0] << ", " << r.plane[1] << ", " << r.plane[2] << ", " << r.plane[3]
     << "],\n  \"repeats\": " << repeats << ",\n  \"mean_ms\": {";
  size_t i = 0;
  for (const auto& kv : mean_ms)
    tj << "\"" << kv.first << "\": " << kv.second << (++i < mean_ms.size() ? ", " : "");
  tj << "}\n}\n";

  std::cout << in_path << ": " << input->size() << " points -> " << r.filtered->size() << " filtered, "
            << r.clusters.size() << " clusters, " << mean_ms["total"] << " ms (" << 1000.0 / mean_ms["total"]
            << " Hz)\n";
  return 0;
}
