#include <opencv2/opencv.hpp>
#include <cmath>
#include <iostream>
#include "smoothing.h"

using namespace cv;
using namespace std;

void mycv::movingAverage(vector<Trajectory> trajectory, vector<Trajectory>& smoothed_trajectory) {
	vector <TransformParam> new_prev_to_cur_transform;
	double sum_x = 0;
	double sum_y = 0;
	double sum_a = 0;
	double avg_x, avg_x_prev=NULL;
	double avg_y, avg_y_prev=NULL;
	double avg_a, avg_a_prev=NULL;

	
	for (size_t i = 0; i < trajectory.size(); i++) {		
		int count = 0;
		for (int j = -SMOOTHING_RADIUS; j < SMOOTHING_RADIUS; j++) {
			if (i + j >= 0 && i + j < trajectory.size()
				&& avg_x_prev==NULL) {
				sum_x += trajectory[i + j].x;
				sum_y += trajectory[i + j].y;
				sum_a += trajectory[i + j].a;

				count++;
			}
		}

		if (avg_x_prev == NULL && avg_y_prev == NULL && avg_a_prev == NULL){
			avg_x = sum_x / count;
			avg_y = sum_y / count;
			avg_a = sum_a / count;
		}
		else if(i+SMOOTHING_RADIUS<trajectory.size()){
			avg_x = (avg_x_prev *(SMOOTHING_RADIUS - 1) + trajectory[i + SMOOTHING_RADIUS].x)/SMOOTHING_RADIUS;
			avg_y = (avg_y_prev *(SMOOTHING_RADIUS - 1) + trajectory[i + SMOOTHING_RADIUS].y) / SMOOTHING_RADIUS;
			avg_a = (avg_a_prev *(SMOOTHING_RADIUS - 1) + trajectory[i + SMOOTHING_RADIUS].a) / SMOOTHING_RADIUS;
		}

		smoothed_trajectory.push_back(Trajectory(avg_x, avg_y, avg_a));
		avg_x_prev = avg_x;
		avg_y_prev = avg_y;
		avg_a_prev = avg_a;
		
	}

}