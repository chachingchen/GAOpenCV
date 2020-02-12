#include "opencv2/opencv.hpp"
#include <cstdio>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <ctime>
#include <iostream>

using namespace cv;
using namespace std;

void blob_detector_train();
void blob_detector_test();


#pragma warning(disable:4996)


int counts;
Mat im;
Mat im_with_keypoints;
int area;
size_t x;
int i;
int k;
char filename[100];

//blob_detector_train: 驗證訓練集訓練出的最佳參數是否正確
void blob_detector_train(){

	for (i = 1; i <= 218; i++)
			{
				//載入影像
				sprintf(filename, "D:/tests/test1/training/test%d.jpg", i);
//		sprintf(filename, "D:/tests/ng/test%d.jpg", i);

				im = imread(filename, IMREAD_GRAYSCALE);

				if (!im.data)
				{
					break;
				}

		//影像前處理resize()
		if (im.cols >= 1024 && im.rows >= 1024){
			cv::resize(im, im, Size(1024/1.5, 1024/1.5));
		}else if(im.cols < 1024 && im.rows < 1024){
			cv::resize(im, im, Size(im.cols, im.rows));
		}

		//影像前處理:二值化，平滑，膨脹

		GaussianBlur(im, im, Size(5,5) ,0 ,0);
		Mat erodeStruct = getStructuringElement(1,Size(3,3));//MORPH_CROSS、MORPH_RECT
		dilate(im, im, erodeStruct);



		//斑點檢測
			SimpleBlobDetector::Params params;
			params.minThreshold = 10;
			params.maxThreshold = 255;


			//面積
			params.filterByArea = true;
			params.minArea = 66.3;//
			params.maxArea = 1034.5;//


			//斑點圓度
			params.filterByCircularity = true;
			params.minCircularity = 0.2499;
			params.maxCircularity = 1;


			//斑點凹凸度
			params.filterByConvexity = true;
			params.minConvexity = 0.0688;
			params.maxConvexity = 1;


			// 越接近0為直線，接近1為圓圈。??性率范??[0.20 ,maxInertiaRatio)?，?中?blob
			params.filterByInertia = true;
			params.minInertiaRatio = 0.1458;
			params.maxInertiaRatio = 1;

			// Storage for blobs
			vector<KeyPoint> keypoints;

			// Set up detector with params
			Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);

			// Detect blobs
			detector->detect( im, keypoints);

			x=keypoints.size();

			// the size of the circle corresponds to the size of blob

			drawKeypoints( im, keypoints, im_with_keypoints, Scalar(0,0,255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );

		std::ostringstream name;
		name << "test" << i << ".jpg";

			if(x == 0){
				cv::imwrite("D:/tests/negative/"+name.str(),im_with_keypoints);

			}else{
				cv::imwrite("D:/tests/positive/"+name.str(),im_with_keypoints);

			}
	}
}

//blob_detector_test: 匯入測試集驗證最佳化參數
void blob_detector_test(){

	for (i = 1; i <= 93; i++)
	{


		sprintf(filename, "D:/tests/test1/testing/test%d.jpg", i);
//		sprintf(filename, "D:/tests/b/test%d.jpg", i);

		im = imread(filename, IMREAD_GRAYSCALE);


		if (!im.data)
			{
				break;
			}


		if (im.cols >= 1024 && im.rows >= 1024){
			cv::resize(im, im, Size(1024/1.5, 1024/1.5));
		}else if(im.cols < 1024 && im.rows < 1024){
			cv::resize(im, im, Size(im.cols, im.rows));
		}


		GaussianBlur(im, im, Size(5	,5) ,0 ,0);
		Mat erodeStruct = getStructuringElement(1,Size(3,3));//MORPH_CROSS、MORPH_RECT
		dilate(im, im, erodeStruct);

		//斑點檢測
			SimpleBlobDetector::Params params;
			params.minThreshold = 10;
			params.maxThreshold = 255;


			params.filterByArea = true;
			params.minArea = 66.3;//
			params.maxArea = 1034.5;//


			params.filterByCircularity = true;
			params.minCircularity = 0.2499;
			params.maxCircularity = 1;



			params.filterByConvexity = true;
			params.minConvexity = 0.0688;
			params.maxConvexity = 1;


			params.filterByInertia = true;
			params.minInertiaRatio = 0.1458;
			params.maxInertiaRatio = 1;

			// Storage for blobs
			vector<KeyPoint> keypoints;

			// Set up detector with params
			Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);

			// Detect blobs
			detector->detect( im, keypoints);

			x=keypoints.size();

			// the size of the circle corresponds to the size of blob

			drawKeypoints( im, keypoints, im_with_keypoints, Scalar(0,0,255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );

		std::ostringstream name;
		name << "test" << i << ".jpg";
			//分類異常照、正常照。分類正確加1分
			if(x > 0){

					cv::imwrite("D:/tests/positive/"+name.str(),im_with_keypoints);

			}else if(x == 0){

					cv::imwrite("D:/tests/negative/"+name.str(),im_with_keypoints);

				}

			}


}


int main()
{


	std::clock_t start;
	double duration;

	start = std::clock();
	srand(time(NULL));


//	blob_detector_train();
	blob_detector_test();


	duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
	std::cout<<"time: "<< duration <<'\n';
		return 0;

}


