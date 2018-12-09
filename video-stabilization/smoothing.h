#pragma once

#include <opencv2/opencv.hpp>
#include "mystab.h"

namespace mycv {

	void movingAverage(vector<Trajectory> trajectory, vector<Trajectory>& smoothed_trajectory);
}