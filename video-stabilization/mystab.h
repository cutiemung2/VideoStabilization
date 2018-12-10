#pragma once

#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

const int SMOOTHING_RADIUS = 30; // In frames. The larger the more stable the video, but less reactive to sudden panning
const int HORIZONTAL_BORDER_CROP = 20; // In pixels. Crops the border to reduce the black borders from stabilisation being too noticeable.

namespace mycv {
	struct TransformParam
	{
		TransformParam() {}
		TransformParam(double _dx, double _dy, double _da) {
			dx = _dx;
			dy = _dy;
			da = _da;
		}

		double dx;
		double dy;
		double da; // angle
	};

	struct Trajectory
	{
		Trajectory() {}
		Trajectory(double _x, double _y, double _a) {
			x = _x;
			y = _y;
			a = _a;
		}

		double x;
		double y;
		double a; // angle
	};

	void goodFeaturesToTrack(Mat image, vector<Point2f>& corners,
		int maxCorners, double qualityLevel, double minDistance,
		InputArray mask = noArray(), int blockSize = 3,
		bool useHarrisDetector = false, double k = 0.04);

	void calcOpticalFlowPyrLK(Mat prevImg, Mat nextImg,
		vector<Point2f> prevPts, vector<Point2f>& nextPts,
		vector<uchar>& status, vector<float>& err,
		Size winSize = Size(21, 21), int maxLevel = 3,
		TermCriteria criteria = TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 30, 0.01),
		int flags = 0, double minEigThreshold = 1e-4);

	void extractInliers(const vector<Point2f>& previousFeaturesIn, const vector<Point2f>& currentFeaturesIn,
						vector<Point2f>* previousInliersOut, vector<Point2f>* currentInliersOut);

	Mat estimateRigidTransform(vector<Point2f> src, vector<Point2f> dst, bool fullAffine);
	void plot(const vector<Point2f>& previousFeaturesIn, 
		const vector<Point2f>& currentFeaturesIn,
		const int* inliersIndices, int inliersCount,
		Mat xCoef, Mat yCoef);

	void warpAffine(Mat src, Mat& dst,
		Mat M, Size dsize,
		int flags = INTER_LINEAR,
		int borderMode = BORDER_CONSTANT,
		const Scalar& borderValue = Scalar());

	void cvtColor(InputArray src, OutputArray dst, int code, int dstCn = 0);

	void resize(InputArray src, OutputArray dst,
		Size dsize, double fx = 0, double fy = 0,
		int interpolation = INTER_LINEAR);

	void movingAverage(vector<Trajectory> trajectory, vector<Trajectory>& smoothed_trajectory);
}
