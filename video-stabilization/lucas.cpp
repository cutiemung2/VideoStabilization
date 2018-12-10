#include <iostream>
#include <algorithm>
#include "lucas.h"

using namespace cv;
using namespace std;

/*
	subimage의 크기가 최소한 (2 * window + 1)이 되도록 피라미드 레벨 조정 

	subimage의 크기가 window 크기의 두배 이하로 작아질 경우
	guess의 값이 영상에 비해 급격히 커지면서 정확한 값을 찾을 수 없는 문제로 인해
	입력 영상의 최대 피라미드 크기를 조절.
*/
int mycv::calcMaximumLevel(const Mat img, const Size winSize, const int hopeThisLevel) {
	const int width = img.cols;
	const int height = img.rows;
	const int searchWidth = 2 * winSize.width + 1;
	const int searchHeight = 2 * winSize.height + 1;

	for (int l = hopeThisLevel; l > 0; l--) {
		const int w = width >> l;
		const int h = height >> l;

		if (w >= 2 * searchWidth && h >= 2 * searchHeight) {
			return l;
		}
	}
	return 0;
}

/*
	영상을 maxLevel의 subimage로 만들기 용이하게 하기 위해
	width, height를 (2^maxLevel)의 배수가 되게 함.
*/
Mat mycv::resizeForPyramid(const Mat src, const int maxLevel) {
	const int rest = 1 << maxLevel;
	const int width = src.cols - src.cols % rest;
	const int height = src.rows - src.rows % rest;
	const unsigned char* srcData = src.data;

	Mat dest = Mat::zeros(Size(width, height), src.type());
	unsigned char* destData = dest.data;

	for (int y = 0; y < height; y++) {
		memcpy(destData + y * width, srcData + y * src.cols, sizeof(unsigned char) * width);
	}

	return dest;
}

/*
	width, height를 입력 영상 대비 1/2로 만든다.
*/
Mat mycv::makeNextLevelImage(const Mat src) {
	const int width = src.cols >> 1;
	const int height = src.rows >> 1;
	const unsigned char* srcData = src.data;
	
	Mat dest = Mat::zeros(Size(width, height), src.type());
	unsigned char* destData = dest.data;

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			destData[y * width + x] = srcData[2 * y * src.rows + 2 * x];
		}
	}
	return dest;
}
	
/*
	입력 영상의 각 픽셀마다 해당 픽셀의 상하, 좌우 미분값을 계산함
*/
void mycv::calcDerivatives(const Mat img, vector<double>& Ix, vector<double>& Iy) {
	const int width = img.cols;
	const int height = img.rows;
	const int capacity = width * height;
	const unsigned char* imgData = img.data;

	Ix.resize(capacity);
	Iy.resize(capacity);

	for (int y = 1; y < height - 1; y++) {
		for (int x = 1; x < width - 1; x++) {
			const int idx = y * width + x;
			Ix[idx] = 0.5 * ((int)imgData[idx + 1] - (int)imgData[idx - 1]);
			Iy[idx] = 0.5 * ((int)imgData[idx + width] - (int)imgData[idx - width]);
		}
	}
}

void mycv::calcOpticalFlowPyrLK(Mat prevImg, Mat nextImg,
	vector<Point2f> prevPts, vector<Point2f>& nextPts,
	vector<uchar>& status, vector<float>& err,
	Size winSize, int maxLevel,
	TermCriteria criteria, int flags, double minEigThreshold) {

	maxLevel = calcMaximumLevel(prevImg, winSize, maxLevel);

	vector<Mat> prevImgs;
	vector<Mat> nextImgs;

	vector<vector<double>> Ix;
	vector<vector<double>> Iy;

	prevImgs.push_back(resizeForPyramid(prevImg, maxLevel));
	nextImgs.push_back(resizeForPyramid(nextImg, maxLevel));

	for (int i = 1; i <= maxLevel; i++) {
		prevImgs.push_back(makeNextLevelImage(prevImgs[i - 1]));
		nextImgs.push_back(makeNextLevelImage(nextImgs[i - 1]));
	}

	for (int i = 0; i <= maxLevel; i++) {
		Ix.push_back(vector<double>());
		Iy.push_back(vector<double>());
		calcDerivatives(prevImgs[i], Ix[i], Iy[i]);
	}

	for (int p = 0; p < prevPts.size(); p++) {
		const Point2f from = prevPts[p];
		Point2f guess(0, 0);

		for (int L = maxLevel; L >= 0; L--) {
			const Point2f lFrom = from / (1 << L);
			const Mat img = prevImgs[L];
			Point2f optFlowVector(0, 0);
			calcOpticalFlowLK(img, nextImgs[L], Ix[L], Iy[L], lFrom, guess, optFlowVector, winSize, criteria);

			guess += optFlowVector;

			/*
				추정된 점이 영상 밖으로 나가게 될 경우
				이전 level에서 잘못 계산되었을 경우를 고려하여
				guess를 초기화 한 후에 다시 optical flow를 계산
				마지막 level에서 같은 현상 발생 시
				해당 점은 추적할 수 없는것으로 판단
			*/
			if (L != 0) {
				Point2f to = lFrom + guess;
				if (0 <= to.x && to.x < img.cols && 0 <= to.y && to.y < img.rows) {
					guess *= 2;
				}
				else {
					guess = Point2f(0, 0);
					if (L == maxLevel) {
						continue;
					}

					optFlowVector = Point2f(0, 0);
					calcOpticalFlowLK(img, nextImgs[L], Ix[L], Iy[L], lFrom, guess, optFlowVector, winSize, criteria);

					guess += optFlowVector;
					to = lFrom + guess;

					if (0 <= to.x && to.x < img.cols && 0 <= to.y && to.y < img.rows) {
						guess *= 2;
					}
					else {
						guess = Point2f(0, 0);
					}
				}
			}
		}
		
		const Mat img = prevImgs[0];
		Point2f to = from + guess;
		bool isInsideFrame = 0 <= to.x && to.x < img.cols && 0 <= to.y && to.y < img.rows;

		if (!isInsideFrame) {
			guess = Point2f(0, 0);
			Point2f optFlowVector(0, 0);
			calcOpticalFlowLK(img, nextImgs[0], Ix[0], Iy[0], from, guess, optFlowVector, winSize, criteria);
			to = from + guess + optFlowVector;
			isInsideFrame = 0 <= to.x && to.x < img.cols && 0 <= to.y && to.y < img.rows;
		}

		nextPts.push_back(to);
		status.push_back(isInsideFrame);
	}
}

/*
	영상의 크기 변화 없이 Iterative Lucas-Kanade Optical Flow 알고리즘으로
	원하는 픽셀(from)의 이동정보(flow vector)를 추정
*/
void mycv::calcOpticalFlowLK(Mat prevImg, Mat nextImg,
	const vector<double>& Ix, const vector<double>& Iy,
	const Point2f from, const Point2f guess, Point2f& optFlowVector,
	Size winSize, TermCriteria criteria) {

	const int width = prevImg.cols;
	const int height = prevImg.rows;
	const int frameSize = width * height;
	const unsigned char* prevData = prevImg.data;
	const unsigned char* nextData = nextImg.data;

	/*
		window 크기 내에서 영상의 gradient 행렬로
		벡터 A = [ Ix, Iy ]에 대하여 A.transpose() * A로 계산
	*/
	double G[2][2] = {
		{0, 0},
		{0, 0}
	};

	for (int wy = -winSize.height; wy <= winSize.height; wy++) {
		for (int wx = -winSize.width; wx <= winSize.width; wx++) {
			const int y = from.y + wy;
			const int x = from.x + wx;

			if (x < 0 || width <= x || y < 0 || height <= y) {
				continue;
			}

			const int idx = y * width + x;
			G[0][0] += Ix[idx] * Ix[idx];
			G[0][1] += Ix[idx] * Iy[idx];
			G[1][0] += Ix[idx] * Iy[idx];
			G[1][1] += Iy[idx] * Iy[idx];
		}
	}

	const double det = G[0][0] * G[1][1] - G[0][1] * G[1][0];
	const double InvG[2][2] = {
		{G[1][1] / det, -G[0][1] / det},
	};

	/*
		다음 영상과의 비교를 통해 점의 이동을 추정
		Iterative한 방법으로 이동량이 0에 수렴하게 되면 종료된다.
		수렴하지 않는 경우에 maxCount까지 iteration이 수행되고 종료.
	*/
	for (int i = 0; i < criteria.maxCount; i++) {
		Point2f mismatch(0, 0);

		for (int wy = -winSize.height; wy <= winSize.height; wy++) {
			for (int wx = -winSize.width; wx <= winSize.width; wx++) {
				const int fy = from.y + wy;
				const int fx = from.x + wx;
				const int ty = from.y + wy + guess.y + optFlowVector.y;
				const int tx = from.x + wx + guess.x + optFlowVector.x;

				if (fx < 0 || width <= fx || fy < 0 || height <= fy
					|| tx < 0 || width <= tx || ty < 0 || height <= ty) {
					continue;
				}

				const int diff = (int)prevData[fy * width + fx] - (int)nextData[ty * width + tx];
				mismatch.y += diff * Iy[fy * width + fx];
				mismatch.x += diff * Ix[fy * width + fx];
			}
		}

		const Point2f gap(InvG[0][0] * mismatch.x + InvG[0][1] * mismatch.y, InvG[1][0] * mismatch.x + InvG[0][1] * mismatch.y);
		optFlowVector += gap;

		const double norm = sqrt(gap.x * gap.x + gap.y * gap.y);
		if (norm < criteria.epsilon) {
			break;
		}
	}
}
