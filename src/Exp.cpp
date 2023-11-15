#include <iostream>
#include <fstream>
#include <iterator>
#include <algorithm>
#include <vector>
#include <string>
#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include "utils/FileReader.h"
// #include "indices/ZM.h"
#include "indices/RSMI.h"
#include "utils/ExpRecorder.h"
#include "utils/Constants.h"
#include "utils/FileWriter.h"
#include "utils/util.h"
#include <torch/torch.h>

#include <xmmintrin.h>
#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>

using namespace std;

int ks[] = {1, 5, 25, 125, 625};
float areas[] = {0.000006, 0.000025, 0.0001, 0.0004, 0.0016};
float ratios[] = {0.25, 0.5, 1, 2, 4};
int Ns[] = {5000, 2500, 500};

int k_length = sizeof(ks) / sizeof(ks[0]);
int window_length = sizeof(areas) / sizeof(areas[0]);
int ratio_length = sizeof(ratios) / sizeof(ratios[0]);

int n_length = sizeof(Ns) / sizeof(Ns[0]);

int query_window_num = 1000;
int query_k_num = 1000;


double knn_diff(vector<Point> acc, vector<Point> pred) {
    int num = 0;
    for (Point point: pred) {
        for (Point point1: acc) {
            if (point.x == point1.x && point.y == point1.y) {
                num++;
            }
        }
    }
    return num * 1.0 / pred.size();
}

void exp_RSMI(FileWriter file_writer, ExpRecorder exp_recorder, const vector<Point> &points,
              map<string, vector<Mbr>> mbrs_map,
              const vector<Point> &query_points, const vector<Point> &insert_points, const string &model_path) {
    exp_recorder.clean();
    exp_recorder.structure_name = "RSMI";
    RSMI *partition = new RSMI(0, Constants::MAX_WIDTH);
    auto start = chrono::high_resolution_clock::now();
    partition->model_path = model_path;
    cout << "Start building models " << model_path << endl;
    partition->build(exp_recorder, points);
    auto finish = chrono::high_resolution_clock::now();
    exp_recorder.time = chrono::duration_cast<chrono::seconds>(finish - start).count();
    cout << "build time: " << exp_recorder.time << " s" << endl;
    exp_recorder.size =
            (2 * Constants::HIDDEN_LAYER_WIDTH + Constants::HIDDEN_LAYER_WIDTH * 1 + Constants::HIDDEN_LAYER_WIDTH * 1 +
             1) * Constants::EACH_DIM_LENGTH * exp_recorder.non_leaf_node_num +
            (Constants::DIM * Constants::PAGESIZE + Constants::PAGESIZE + Constants::DIM * Constants::DIM) *
            Constants::EACH_DIM_LENGTH * exp_recorder.leaf_node_num;
    file_writer.write_build(exp_recorder);
    exp_recorder.clean();

//    partition->point_query(exp_recorder, points);
//    cout << "finish point_query: pageaccess:" << exp_recorder.page_access << endl;
//    cout << "finish point_query time: " << exp_recorder.time << endl;
//    file_writer.write_point_query(exp_recorder);
//    exp_recorder.clean();

    exp_recorder.window_size = areas[2];
    exp_recorder.window_ratio = ratios[2];
    partition->acc_window_query(exp_recorder, mbrs_map[to_string(areas[2]) + to_string(ratios[2])]);
    cout << "RSMI::acc_window_query time: " << exp_recorder.time << endl;
    cout << "RSMI::acc_window_query page_access: " << exp_recorder.page_access << endl;
    file_writer.write_acc_window_query(exp_recorder);
    partition->window_query(exp_recorder, mbrs_map[to_string(areas[2]) + to_string(ratios[2])]);
    exp_recorder.accuracy =
            ((double) exp_recorder.window_query_result_size) / exp_recorder.acc_window_query_qesult_size;
    cout << "window_query time: " << exp_recorder.time << endl;
    cout << "window_query page_access: " << exp_recorder.page_access << endl;
    cout << "window_query accuracy: " << exp_recorder.accuracy << endl;
    file_writer.write_window_query(exp_recorder);

    /*
    exp_recorder.clean();
    exp_recorder.k_num = ks[2];
    partition->acc_kNN_query(exp_recorder, query_points, ks[2]);
    cout << "exp_recorder.time: " << exp_recorder.time << endl;
    cout << "exp_recorder.page_access: " << exp_recorder.page_access << endl;
    file_writer.write_acc_kNN_query(exp_recorder);
    partition->kNN_query(exp_recorder, query_points, ks[2]);
    cout << "exp_recorder.time: " << exp_recorder.time << endl;
    cout << "exp_recorder.page_access: " << exp_recorder.page_access << endl;
    exp_recorder.accuracy = knn_diff(exp_recorder.acc_knn_query_results, exp_recorder.knn_query_results);
    cout << "exp_recorder.accuracy: " << exp_recorder.accuracy << endl;
    file_writer.write_kNN_query(exp_recorder);
    exp_recorder.clean();

    partition->insert(exp_recorder, insert_points);
    cout << "exp_recorder.insert_time: " << exp_recorder.insert_time << endl;
    exp_recorder.clean();
    partition->point_query(exp_recorder, points);
    cout << "finish point_query: pageaccess:" << exp_recorder.page_access << endl;
    cout << "finish point_query time: " << exp_recorder.time << endl;
    exp_recorder.clean();
     */
}


int main(int argc, char **argv) {
    int c;
    static struct option long_options[] =
            {
                    {"file", required_argument, NULL, 'f'},
            };
    std::string file_path;
    while (1) {
        int opt_index = 0;
        c = getopt_long(argc, argv, "f:", long_options, &opt_index);

        if (-1 == c) {
            break;
        }
        switch (c) {
            case 'f':
                file_path.assign(optarg);
                break;
        }
    }
    auto exec_path = std::string(argv[0]);
    std::string model_root_path = exec_path.substr(0, exec_path.find_last_of('/'));

    if (access(file_path.c_str(), R_OK) != 0) {
        printf("Cannot open %s\n", file_path.c_str());
        exit(1);
    }

    auto file_name = file_path;

    std::replace(file_name.begin(), file_name.end(), '/', '_');

    ExpRecorder exp_recorder;

    FileReader filereader(file_path, ",");
    vector<Point> points = filereader.get_points();
    auto inserted_num = points.size() / 2;
    exp_recorder.insert_num = inserted_num;
    vector<Point> query_points;
    vector<Point> insert_points;
    //***********************write query data*********************
    FileWriter query_file_writer(Constants::QUERYPROFILES, file_name);
    query_points = Point::get_points(points, query_k_num);
    query_file_writer.write_points(query_points, exp_recorder);
    insert_points = Point::get_inserted_points(exp_recorder.insert_num);
    query_file_writer.write_inserted_points(insert_points, exp_recorder);

    for (size_t i = 0; i < window_length; i++) {
        for (size_t j = 0; j < ratio_length; j++) {
            exp_recorder.window_size = areas[i];
            exp_recorder.window_ratio = ratios[j];
            vector<Mbr> mbrs = Mbr::get_mbrs(points, exp_recorder.window_size, query_window_num,
                                             exp_recorder.window_ratio);
            query_file_writer.write_mbrs(mbrs, exp_recorder);
        }
    }
    //**************************prepare knn, window query, and insertion data******************
    FileReader knn_reader(
            (Constants::QUERYPROFILES + Constants::KNN + file_name + "_" + to_string(exp_recorder.k_num) + ".csv"),
            ",");
    map<string, vector<Mbr>> mbrs_map;
    FileReader query_filereader;

    query_points = query_filereader.get_points(
            (Constants::QUERYPROFILES + Constants::KNN + file_name + ".csv"), ",");
    insert_points = query_filereader.get_points(
            (Constants::QUERYPROFILES + Constants::UPDATE + file_name + "_" +
             to_string(exp_recorder.insert_num) + ".csv"), ",");

    for (size_t i = 0; i < window_length; i++) {
        for (size_t j = 0; j < ratio_length; j++) {
            exp_recorder.window_size = areas[i];
            exp_recorder.window_ratio = ratios[j];
            vector<Mbr> mbrs = query_filereader.get_mbrs(
                    (Constants::QUERYPROFILES + Constants::WINDOW + file_name + "_" +
                     to_string(exp_recorder.window_size) + "_" + to_string(exp_recorder.window_ratio) + ".csv"), ",");
            mbrs_map.insert(pair<string, vector<Mbr>>(to_string(areas[i]) + to_string(ratios[j]), mbrs));
        }
    }
    model_root_path += "/" + Constants::TORCH_MODELS + file_name;
    file_utils::check_dir(model_root_path);
    std::string model_path = model_root_path + "/N=" + to_string(exp_recorder.N);
    FileWriter file_writer(Constants::RECORDS, file_name);
    exp_RSMI(file_writer, exp_recorder, points, mbrs_map, query_points, insert_points, model_path);
}
