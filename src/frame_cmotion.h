#ifndef FRAME_CMOTION_H
#define FRAME_CMOTION_H

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include "filter.h"
// BP old //
// #include "algTEST/alg_src/BP.h"
// #include "algTEST/alg_src/image.h"
// BP new //
 #include "BP/Optimization.h"
// #include "BP/Optimization.cpp"


#include <time.h>
#include <math.h>
#include <algorithm>
#define MAX_DEPTH 256
#define WIN_SIZE 3
//For How many should trust before:

#define PREROBUST 50 //original 50
#define FDF_CONSTANT 0.15 //orginally 50 <- hahaha
#define FDF_CONSTANT_ASW 0.15 //best 0.1
#define FDF_GAMMA 0.001 //best 0.01
//Tsukuba
/*
#define RANSAC_THRESHOLD 0.05
#define RANSAC_CONFIDENCE 0.995
#define FOCAL_LENGTH 615
#define STEREO_BASELINE 10
*/
//Kitti
#define RANSAC_THRESHOLD 0.5
#define RANSAC_CONFIDENCE 0.95
#define FOCAL_LENGTH 35
#define STEREO_BASELINE 340

using namespace std;
using namespace cv;

Mat get_optical(IplImage *imageL, IplImage *imageL_pre){
    Mat matL = cvarrToMat(imageL);
    Mat matL_pre = cvarrToMat(imageL_pre);	
    //Mat matD = cvarrToMat(imageD);
    Mat flow;
    UMat flowUmat;
	
    cvtColor(matL, matL, cv::COLOR_BGR2GRAY);
    cvtColor(matL_pre, matL_pre, cv::COLOR_BGR2GRAY);
    calcOpticalFlowFarneback(matL_pre, matL, flowUmat, 0.4, 1, 12, 2, 8, 1.2, 0);
    flowUmat.copyTo(flow);
    return flow;
    //y->row x->column
    //const Point2f flowatxy = flow.at<Point2f>(y, x) * 10;
    //flowatxy.x,flowatxy.y
}

void testing_affine(IplImage *imageL, IplImage *imageD, IplImage *imageL_pre, IplImage *imageD_pre, IplImage *imageD_refine, int file_iter){
	if(imageL_pre != 0 && imageD_pre!=0){
		int method = 2; // 1 for flow, 2 for flow+Edge, 3 for Feature
		int lowThreshold = 120; //threshold for Canny Edge

		double ransacThreshold = RANSAC_THRESHOLD; // Ransac Threshold in estimateAffine3D
		double ransacConfidence = RANSAC_CONFIDENCE; // Ransac Confidence in estimateAffine3D
		float f = FOCAL_LENGTH;
		float B = STEREO_BASELINE;

		Mat matL = cvarrToMat(imageL);
		Mat matD = cvarrToMat(imageD);
		Mat matD_pre = cvarrToMat(imageD_pre);
		Mat matL_pre = cvarrToMat(imageL_pre);
		Mat gmat_L, gmatL_pre, flow;
		UMat flowUmat;
		//Get optical flow
    	cvtColor(matL, gmat_L, cv::COLOR_BGR2GRAY);
    	cvtColor(matL_pre, gmatL_pre, cv::COLOR_BGR2GRAY);

		
		vector<Point3f> first, second, total;
		Mat out, inlier;
		// namedWindow("matL", WINDOW_AUTOSIZE);
		// imshow("matL", matL);
        // waitKey(0);
		int nRows = matL.rows;
		int nCols = matL.cols;
		if(method==1){
			calcOpticalFlowFarneback(gmatL_pre, gmat_L, flowUmat, 0.4, 1, 12, 2, 8, 1.2, 0);
			flowUmat.copyTo(flow);
			for(int i=0; i<nRows; i++){
				for(int j=0; j<nCols; j++){
					Vec3b srcPixel = matL.at<Vec3b>(i, j);
					int p_d = (matD_pre.at<Vec3b>(i,j))[0];
					const Point2f flowatxy = flow.at<Point2f>(i, j)*1;
					int flow_i = i+ flowatxy.y;
					int flow_j = j+ flowatxy.x;
					if(flow_i<0)
						flow_i=0;
					if(flow_j<0)
						flow_j=0;
					int d = (matD.at<Vec3b>(flow_i,flow_j))[0];
					float p_Z = (float)(255-p_d)/(f*B);
					float p_X = (float)j/(f*p_Z);
					float p_Y = (float)i/(f*p_Z);
					float Z = (float)(255-d)/(f*B);
					float X = flow_j/(f*Z);
					float Y = flow_i/(f*Z);
					total.push_back(Point3f(p_X,p_Y,p_Z));
					first.push_back(Point3f(p_X,p_Y,p_Z));
					second.push_back(Point3f(X,Y,Z));
					//cerr<<(int)srcPixel[0]<<" "<<(int)srcPixel[1]<<" "<<(int)srcPixel[2]<<endl;
				}
			}
		}
		else if(method==2){
			Mat detected_edges;
			Canny( matL_pre, detected_edges, lowThreshold, lowThreshold*3, 3);
			// Mat original;
			// detected_edges.copyTo(original); 
			// namedWindow( "Display window", WINDOW_AUTOSIZE );
			// imshow("Display window", original);
			// waitKey(0);
			calcOpticalFlowFarneback(gmatL_pre, gmat_L, flowUmat, 0.4, 1, 12, 2, 8, 1.2, 0);
			flowUmat.copyTo(flow);
			for(int i=0; i<nRows; i++){
				for(int j=0; j<nCols; j++){
					//detect if it is canny edge
					
					Vec3b srcPixel = matL.at<Vec3b>(i, j);
					int p_d = (matD_pre.at<Vec3b>(i,j))[0];
					const Point2f flowatxy = flow.at<Point2f>(i, j)*1;
					int flow_i = i+ flowatxy.y;
					int flow_j = j+ flowatxy.x;
					if(flow_i<0)
						flow_i=0;
					if(flow_j<0)
						flow_j=0;
					int d = (matD.at<Vec3b>(flow_i,flow_j))[0];
					float p_Z = (float)(255-p_d)/(f*B);
					float p_X = (float)j/(f*p_Z);
					float p_Y = (float)i/(f*p_Z);
					float Z = (float)(255-d)/(f*B);
					float X = flow_j/(f*Z);
					float Y = flow_i/(f*Z);
					total.push_back(Point3f(p_X,p_Y,p_Z));
					if((int)detected_edges.at<uchar>(i,j)==255){
						first.push_back(Point3f(p_X,p_Y,p_Z));
						second.push_back(Point3f(X,Y,Z));
						//arrowedLine(original, Point(j, i), Point(cvRound(flow_j), cvRound(flow_i)), Scalar(180,180,180));
					}
				}
			}
			// imshow("Display window", original);
			// waitKey(0); 
		}
		else{
			Ptr<FeatureDetector> detector = ORB::create();
    		vector<KeyPoint> keypoints_L, keypoints_pre;
    		detector->detect(matL, keypoints_L);
    		detector->detect(matL_pre, keypoints_pre);

			Ptr<DescriptorExtractor> extractor = ORB::create();
			Mat descriptors_L, descriptors_pre;
   			extractor->compute(matL, keypoints_L, descriptors_L);
    		extractor->compute(matL_pre, keypoints_pre, descriptors_pre);
			if(descriptors_pre.type()!=CV_32F) {
				descriptors_pre.convertTo(descriptors_pre, CV_32F);
			}
			if(descriptors_L.type()!=CV_32F) {
				descriptors_L.convertTo(descriptors_L, CV_32F);
			}
			Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("FlannBased");
			vector<DMatch> matches;
			matcher->match(descriptors_pre, descriptors_L, matches);
			double max_dist = 0; double min_dist = 100;
			//-- Quick calculation of max and min distances between keypoints
			for (int i = 0; i < matches.size(); i++)
			{
				double dist = matches[i].distance;
				if (dist < min_dist) min_dist = dist;
				if (dist > max_dist) max_dist = dist;
			}
			vector<DMatch> good_matches;
			for (int i = 0; i < matches.size(); i++)
			{
				if (matches[i].distance < 200)
				{
					good_matches.push_back(matches[i]);
				}
			}
			for(size_t ii = 0; ii < good_matches.size(); ii++)
			{
				//cerr<<good_matches[ii].queryIdx<<","<<good_matches[ii].trainIdx<<endl;
				Point2f point1 = keypoints_pre[good_matches[ii].queryIdx].pt;
				Point2f point2 = keypoints_L[good_matches[ii].trainIdx].pt;

				int i = point1.y;
				int j = point1.x;
				int flow_i = point2.y;
				int flow_j = point2.x;

				int p_d = (matD_pre.at<Vec3b>(i,j))[0];
				int d = (matD.at<Vec3b>(flow_i,flow_j))[0];

				float p_Z = (float)(255-p_d)/(f*B);
				float p_X = (float)j/(f*p_Z);
				float p_Y = (float)i/(f*p_Z);
				float Z = (float)(255-d)/(f*B);
				float X = flow_j/(f*Z);
				float Y = flow_i/(f*Z);
				first.push_back(Point3f(p_X,p_Y,p_Z));
				second.push_back(Point3f(X,Y,Z));
				// cerr<<"dis:"<<good_matches[i].distance;
				//cerr<<"("<<point1<<")->("<<point2<<")\n";	
				// cerr<<"("<<Point3f(p_X,p_Y,p_Z)<<")->";
				// cerr<<"("<<Point3f(X,Y,Z)<<")\n";
				// do something with the best points...

			}
			for(int i=0; i<nRows; i++){
				for(int j=0; j<nCols; j++){
					//detect if it is canny edge
					Vec3b srcPixel = matL.at<Vec3b>(i, j);
					int p_d = (matD_pre.at<Vec3b>(i,j))[0];
					float p_Z = (float)(255-p_d)/(f*B);
					float p_X = (float)j/(f*p_Z);
					float p_Y = (float)i/(f*p_Z);
					total.push_back(Point3f(p_X,p_Y,p_Z));
				}
			}
			Mat img_matches;

			// drawMatches(matL_pre, keypoints_pre, matL, keypoints_L, good_matches, img_matches, Scalar::all(-1), Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
			// namedWindow("match", WINDOW_AUTOSIZE);
			// imshow("match", img_matches);
        	// waitKey(0);
		}
		//cerr<<"\nsize:"<<first.size();
		estimateAffine3D(first, second, out, inlier, ransacThreshold, ransacConfidence);
		out.convertTo(out, CV_32F);
		cerr<<"\n"<<out<<endl;
		//Testing affine
		//namedWindow("affine", WINDOW_AUTOSIZE);
		IplImage* imageTest;
		cvSet(imageD_refine, CV_RGB(0,0,0));
		int firstSize = nRows*nCols;
		for(int i=0; i<firstSize; i++){
			float xx = total[i].x;
			float yy = total[i].y;
			float zz = total[i].z;
			Mat PointA = (Mat_<float>(4, 1) << xx,yy,zz,1);
			Mat testResult = out*PointA;
			float tx = testResult.at<float>(0,0);
			float ty = testResult.at<float>(1,0);
			float tz = testResult.at<float>(2,0);
			int u = ty*f*tz; //row
			int v = tx*f*tz; //column
			int d = 255-tz*f*B;
			if(u<0) u=0;
			if(v<0) v=0;	
			if(u>=nRows) u=nRows-1;
			if(v>=nCols) v=nCols-1;
			if(d<0) d=0;
			if(d>=255) d=255;
			imageD_refine->imageData[u*imageD_refine->widthStep+v*3]=d;
			imageD_refine->imageData[u*imageD_refine->widthStep+v*3+1]=d;
			imageD_refine->imageData[u*imageD_refine->widthStep+v*3+2]=d;
		}
		// cvShowImage("Output:", imageD);
		// cvWaitKey(0);
		//cvShowImage("Output:", imageD_refine);
		// cvWaitKey(0);
	}
    else{
        imageD_refine=NULL;
    }
}

void true_affine(IplImage *imageL, IplImage *imageR, IplImage *imageD, IplImage *imageL_pre, IplImage *imageD_pre, IplImage *imageD_refine, int depth, int BSIZE=3, bool isLeft=true, int file_iter=1, int clean=10, bool doCensus=false){
	if(imageL_pre != 0 && imageD_pre!=0){
		int method = 3; // 1 for flow, 2 for flow+Edge, 3 for Feature
		int lowThreshold = 120; //threshold for Canny Edge
		//int useFlow = 4; //detect if nearby have Canny Edge
		

		double ransacThreshold = RANSAC_THRESHOLD; // Ransac Threshold in estimateAffine3D
		double ransacConfidence = RANSAC_CONFIDENCE; // Ransac Confidence in estimateAffine3D
		float f = FOCAL_LENGTH;
		float B = STEREO_BASELINE;

		Mat matL = cvarrToMat(imageL);
		Mat matD = cvarrToMat(imageD);
		Mat matD_pre = cvarrToMat(imageD_pre);
		Mat matL_pre = cvarrToMat(imageL_pre);
		Mat gmat_L, gmatL_pre, flow;
		UMat flowUmat;
		//Get optical flow
    	cvtColor(matL, gmat_L, cv::COLOR_BGR2GRAY);
    	cvtColor(matL_pre, gmatL_pre, cv::COLOR_BGR2GRAY);

		
		vector<Point3f> first, second, total;
		Mat out, inlier;
		// namedWindow("matL", WINDOW_AUTOSIZE);
		// imshow("matL", matL);
        // waitKey(0);
		int nRows = matL.rows;
		int nCols = matL.cols;
		if(method==1){
			calcOpticalFlowFarneback(gmatL_pre, gmat_L, flowUmat, 0.4, 1, 12, 2, 8, 1.2, 0);
			flowUmat.copyTo(flow);
			for(int i=0; i<nRows; i++){
				for(int j=0; j<nCols; j++){
					Vec3b srcPixel = matL.at<Vec3b>(i, j);
					int p_d = (matD_pre.at<Vec3b>(i,j))[0];
					const Point2f flowatxy = flow.at<Point2f>(i, j)*1;
					int flow_i = i+ flowatxy.y;
					int flow_j = j+ flowatxy.x;
					if(flow_i<0)
						flow_i=0;
					if(flow_j<0)
						flow_j=0;
					int d = (matD.at<Vec3b>(flow_i,flow_j))[0];
					float p_Z = (float)(255-p_d)/(f*B);
					float p_X = (float)j/(f*p_Z);
					float p_Y = (float)i/(f*p_Z);
					float Z = (float)(255-d)/(f*B);
					float X = flow_j/(f*Z);
					float Y = flow_i/(f*Z);
					total.push_back(Point3f(p_X,p_Y,p_Z));
					first.push_back(Point3f(p_X,p_Y,p_Z));
					second.push_back(Point3f(X,Y,Z));
					//cerr<<(int)srcPixel[0]<<" "<<(int)srcPixel[1]<<" "<<(int)srcPixel[2]<<endl;
				}
			}
		}
		else if(method==2){
			Mat detected_edges;
			Canny( matL_pre, detected_edges, lowThreshold, lowThreshold*3, 3);
			// Mat original;
			// detected_edges.copyTo(original); 
			// namedWindow( "Display window", WINDOW_AUTOSIZE );
			// imshow("Display window", original);
			// waitKey(0);
			calcOpticalFlowFarneback(gmatL_pre, gmat_L, flowUmat, 0.4, 1, 12, 2, 8, 1.2, 0);
			flowUmat.copyTo(flow);
			for(int i=0; i<nRows; i++){
				for(int j=0; j<nCols; j++){
					//detect if it is canny edge
					Vec3b srcPixel = matL.at<Vec3b>(i, j);
					int p_d = (matD_pre.at<Vec3b>(i,j))[0];
					const Point2f flowatxy = flow.at<Point2f>(i, j)*1;
					int flow_i = i+ flowatxy.y;
					int flow_j = j+ flowatxy.x;
					if(flow_i<0)
						flow_i=0;
					if(flow_j<0)
						flow_j=0;
					int d = (matD.at<Vec3b>(flow_i,flow_j))[0];
					float p_Z = (float)(255-p_d)/(f*B);
					float p_X = (float)j/(f*p_Z);
					float p_Y = (float)i/(f*p_Z);
					float Z = (float)(255-d)/(f*B);
					float X = flow_j/(f*Z);
					float Y = flow_i/(f*Z);
					total.push_back(Point3f(p_X,p_Y,p_Z));
					if((int)detected_edges.at<uchar>(i,j)==255){
						first.push_back(Point3f(p_X,p_Y,p_Z));
						second.push_back(Point3f(X,Y,Z));
						//arrowedLine(original, Point(j, i), Point(cvRound(flow_j), cvRound(flow_i)), Scalar(180,180,180));
					}
				}
			}
			// imshow("Display window", original);
			// waitKey(0); 
		}
		else{
			Ptr<FeatureDetector> detector = ORB::create();
    		vector<KeyPoint> keypoints_L, keypoints_pre;
    		detector->detect(matL, keypoints_L);
    		detector->detect(matL_pre, keypoints_pre);

			Ptr<DescriptorExtractor> extractor = ORB::create();
			Mat descriptors_L, descriptors_pre;
   			extractor->compute(matL, keypoints_L, descriptors_L);
    		extractor->compute(matL_pre, keypoints_pre, descriptors_pre);
			if(descriptors_pre.type()!=CV_32F) {
				descriptors_pre.convertTo(descriptors_pre, CV_32F);
			}
			if(descriptors_L.type()!=CV_32F) {
				descriptors_L.convertTo(descriptors_L, CV_32F);
			}
			Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("FlannBased");
			vector<DMatch> matches;
			matcher->match(descriptors_pre, descriptors_L, matches);
			double max_dist = 0; double min_dist = 100;
			//-- Quick calculation of max and min distances between keypoints
			for (int i = 0; i < matches.size(); i++)
			{
				double dist = matches[i].distance;
				if (dist < min_dist) min_dist = dist;
				if (dist > max_dist) max_dist = dist;
			}
			vector<DMatch> good_matches;
			for (int i = 0; i < matches.size(); i++)
			{
				if (matches[i].distance < 200)
				{
					good_matches.push_back(matches[i]);
				}
			}
			for(size_t ii = 0; ii < good_matches.size(); ii++)
			{
				//cerr<<good_matches[ii].queryIdx<<","<<good_matches[ii].trainIdx<<endl;
				Point2f point1 = keypoints_pre[good_matches[ii].queryIdx].pt;
				Point2f point2 = keypoints_L[good_matches[ii].trainIdx].pt;

				int i = point1.y;
				int j = point1.x;
				int flow_i = point2.y;
				int flow_j = point2.x;

				int p_d = (matD_pre.at<Vec3b>(i,j))[0];
				int d = (matD.at<Vec3b>(flow_i,flow_j))[0];

				float p_Z = (float)(255-p_d)/(f*B);
				float p_X = (float)j/(f*p_Z);
				float p_Y = (float)i/(f*p_Z);
				float Z = (float)(255-d)/(f*B);
				float X = flow_j/(f*Z);
				float Y = flow_i/(f*Z);
				first.push_back(Point3f(p_X,p_Y,p_Z));
				second.push_back(Point3f(X,Y,Z));
				// cerr<<"dis:"<<good_matches[i].distance;
				//cerr<<"("<<point1<<")->("<<point2<<")\n";	
				// cerr<<"("<<Point3f(p_X,p_Y,p_Z)<<")->";
				// cerr<<"("<<Point3f(X,Y,Z)<<")\n";
				// do something with the best points...

			}
			for(int i=0; i<nRows; i++){
				for(int j=0; j<nCols; j++){
					//detect if it is canny edge
					Vec3b srcPixel = matL.at<Vec3b>(i, j);
					int p_d = (matD_pre.at<Vec3b>(i,j))[0];
					float p_Z = (float)(255-p_d)/(f*B);
					float p_X = (float)j/(f*p_Z);
					float p_Y = (float)i/(f*p_Z);
					total.push_back(Point3f(p_X,p_Y,p_Z));
				}
			}
			Mat img_matches;

			// drawMatches(matL_pre, keypoints_pre, matL, keypoints_L, good_matches, img_matches, Scalar::all(-1), Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
			// namedWindow("match", WINDOW_AUTOSIZE);
			// imshow("match", img_matches);
        	// waitKey(0);
		}
		//cerr<<"\nsize:"<<first.size();
		estimateAffine3D(first, second, out, inlier, ransacThreshold, ransacConfidence);
		out.convertTo(out, CV_32F);
		cerr<<"\n"<<out<<endl;
		//Testing affine
		//namedWindow("affine", WINDOW_AUTOSIZE);
		cvSet(imageD_refine, CV_RGB(0,0,0));

		//True Matching start here!!!!!!!
		int LR_minus=1;
		bool usingPreImg = true;
		if (!isLeft){
			IplImage *temp = imageL;
			imageL = imageR;
			imageR = temp;
			LR_minus=-1;
		}
		float alpha= 0.8;
		if(!doCensus){
			alpha = 0;
		}
		float gamma = 10;
		int preRobust = PREROBUST;
		int colorBound = 10;
		cvSet(imageD_refine, cvScalar(0,0,0));
		int mindis;
		float minNum;
		int dif;
		int totalIterator=0;
		for(int i=0;i<imageL->height;i++){
			for(int j=0;j<imageL->widthStep;j=j+3, totalIterator++){
				int wid = imageL->widthStep;
				int centerColor = imageL->imageData[i*wid+j];
				float preWeight = 0;
				int preDif = 0;
				int preDisp = 0;
				int preCount = 0;
				if(file_iter%clean!=0){
					float xx = total[totalIterator].x;
					float yy = total[totalIterator].y;
					float zz = total[totalIterator].z;
					Mat PointA = (Mat_<float>(4, 1) << xx,yy,zz,1);
					Mat testResult = out*PointA;
					float tx = testResult.at<float>(0,0);
					float ty = testResult.at<float>(1,0);
					float tz = testResult.at<float>(2,0);
					int u = ty*f*tz; //row
					int v = tx*f*tz; //column
					int d = 255-tz*f*B;
					if(u<0) u=0;
					if(v<0) v=0;	
					if(u>=nRows) u=nRows-1;
					if(v>=nCols) v=nCols-1;
					if(d<0) d=0;
					if(d>=255) d=255;

					int flow_i=u;
					int flow_j=v;

					for(int p=0;p<BSIZE;++p){
						if(((flow_i-BSIZE/2)+p)>=0 && ((flow_i-BSIZE/2)+p)<imageL->height){
							for(int q=0;q<BSIZE;++q){
								if(((flow_j-BSIZE/2)+q)>=0 && ((flow_j-BSIZE/2)+q)<imageL->widthStep/3){
									int ii=(flow_i-BSIZE/2+p);
									int jj=(flow_j-(BSIZE/2)+q)*3;
									preDif+=abs((imageL->imageData[ii*wid+jj])-(imageL_pre->imageData[ii*wid+jj]));
									preCount++;
								}
							}
						}
					}
					preWeight = FDF(preDif,preCount,FDF_CONSTANT);
					preDisp = imageD_pre->imageData[flow_i*wid+flow_j*3];
					preDisp = (int)((float)preDisp*depth/255);
				}
				// Instead of looking for the same point color in previous color, we add some optical flow in it!
				


				minNum=100000;
				mindis=depth;
				for(int k=j,c=0;k>0,c<depth;k=k-(LR_minus*3),c++){
					float mome_w = 0;
					float ww_wall = 0;
					for(int p=0;p<BSIZE;++p){
						if(((i-BSIZE/2)+p)>0){
							for(int q=0;q<BSIZE;++q){
								if(((k-BSIZE/2)+q-1)>0){
									int centerNewColor = imageR->imageData[i*wid+k];
									int ii=(i-BSIZE/2+p);
									int jj=(j-(BSIZE/2)*3+3*q);
									int kk=(k-(BSIZE/2)*3+3*q);
									dif=abs((imageL->imageData[ii*wid+jj])-(imageR->imageData[ii*wid+kk]));
									//TODO HERE//
									int LGW=(imageL->imageData[ii*wid+jj+3])-(imageL->imageData[ii*wid+jj-3]);
									int RGW=(imageR->imageData[ii*wid+kk+3])-(imageR->imageData[ii*wid+kk-3]);
									
									float censusAll = 0;
									int validPoint = 0;
									int Lcenter = imageL->imageData[ii*wid+jj];
									int Rcenter = imageR->imageData[ii*wid+kk];
									if(doCensus){
										for (int c_i=-1; c_i<=1; c_i++){
											for(int c_j=-3; c_j<=3; c_j+=3){
												//cerr<<kk+c_i<<","<<c_j<<endl;
												if(ii+c_i>=0 && kk+c_j>=0 && ii+c_i<(imageL->height) && kk+c_j<(wid)){
													validPoint++;
													bool Lcompare = false;
													bool Rcompare = true;
													Lcompare = (imageL->imageData[(ii+c_i)*wid+(jj+c_j)]>=Lcenter);
													Rcompare = (imageR->imageData[(ii+c_i)*wid+(kk+c_j)]>=Rcenter);
													if(Lcompare!=Rcompare){
														censusAll++;
													}
												}
											}
										}
										if(validPoint==0)
											censusAll=10000;
										else
											censusAll/=validPoint;
									}
									//TODO END//
									float ww_w = weiFunc(centerColor,(imageR->imageData[ii*wid+kk]),gamma);
									ww_wall+=ww_w;
									float rho_w = (1-alpha)*min(dif,colorBound) + alpha*(censusAll);
									mome_w += ww_w*rho_w;
									mome_w += preWeight*(min(preRobust,abs(c-preDisp)));
								}
								else{
									mome_w += 1000*BSIZE;
								}
							}
						}
						else{
							mome_w += 1000*BSIZE;
						}
					}
					mome_w=mome_w/ww_wall;
					if(mome_w<minNum){
						minNum=mome_w;
						mindis=c;
					}
				}
				int newColor = (255/depth)*mindis;
				imageD_refine->imageData[i*imageD->widthStep+j]=newColor;
				imageD_refine->imageData[i*imageD->widthStep+j+1]=newColor;
				imageD_refine->imageData[i*imageD->widthStep+j+2]=newColor;
			}
		}
	}
    else{
        imageD_refine=NULL;
    }
}

void testing_optical(IplImage *imageL, IplImage *imageD, IplImage *imageL_pre, int file_iter, int speed = 10){
    if (imageL_pre != NULL && imageL_pre != 0){
        namedWindow("prew", WINDOW_AUTOSIZE);
        Mat matL = cvarrToMat(imageL);
        Mat matL_pre = cvarrToMat(imageL_pre);	
        Mat matD = cvarrToMat(imageD);
        Mat matTemp;
        Mat matTempTemp;
        Mat flow;
        UMat flowUmat;
        Mat original;
		matL.copyTo(original);
        cvtColor(matL, matL, cv::COLOR_BGR2GRAY);
        cvtColor(matL_pre, matL_pre, cv::COLOR_BGR2GRAY);
        cvtColor(matD, matTemp, cv::COLOR_BGR2GRAY);
        imshow("prew", original);
        waitKey(0);

        calcOpticalFlowFarneback(matL_pre, matL, flowUmat, 0.4, 1, 12, 2, 8, 1.2, 0);
        flowUmat.copyTo(flow);
		speed=5;	
        for (int y = 0; y < original.rows; y += 10) {
            for (int x = 0; x < original.cols; x += 10)
            {
                // get the flow from y, x position * 10 for better visibility
                const Point2f flowatxy = flow.at<Point2f>(y, x) * speed;
                // draw line at flow direction
                arrowedLine(original, Point(x, y), Point(cvRound(x + flowatxy.x), cvRound(y + flowatxy.y)), Scalar(0,0,255));
                // draw initial point
                circle(original, Point(x, y), 0, Scalar(0, 0, 0), -1);
            }
        }
        imshow("prew", original);
		//imwrite("output/Tsukuba/left/test_optical/"+num2str(file_iter)+".png", original);
        waitKey(0);
	}
    else{
        
    }

}

//Frame different function//
//NEW VERSION WITH REFINE OUTPUT!
//When using big frame, the previous image comparison will use a BSIZE square to calculate the "simularity"
//TODO:
void disp_stereo_color_big_frame_optical(IplImage *imageL,IplImage *imageR,IplImage *imageD,IplImage *imageL_pre,IplImage *imageD_pre, IplImage *imageD_refine, int depth,bool window=false,int bsize=3,bool isLeft=true, int iter=1, int clean=10, int speed=1, int method=1, int nsize=5){
    
	
	int LR_minus=1;
    // Flow multiplier
	bool usingPreImg = true;
    Mat flow;
	if (!isLeft){
		IplImage *temp = imageL;
		imageL = imageR;
		imageR = temp;
		LR_minus=-1;
	}
	if (imageD_pre == NULL ||imageD_pre==0){
		usingPreImg = false;
	}
    else{
        flow = get_optical(imageL, imageL_pre);
    }
	int BSIZE = bsize;
	int NSIZE = nsize;
	float alpha= 0.2;
	float gamma = 10;
	int preRobust = PREROBUST;
	//Testing Two Method Avg!!!
	double preWeightSum=0;
	double preWeightCount=0;
	
	int colorBound = 10;
	int gradBound = 2;
	cvSet(imageD, cvScalar(0,0,0));
	int dif;
	int hei = imageL->height;
	int height = imageL->height;
	int wid = imageL->widthStep;
	int width = imageL->width;
  	for(int i=0;i<hei;i++){
		for(int j=0;j<wid;j=j+3){
			int centerColor = imageL->imageData[i*wid+j];
			float preWeight = 0;
			int preDif = 0;
			int preDisp = 0;
			int preCount = 0;
			float preDifSum = 0;
			// Instead of looking for the same point color in previous color, we add some optical flow in it!
			if(usingPreImg){
                int realJ = (int)(j/3);
                const Point2f flowatxy = flow.at<Point2f>(i, realJ)*speed;
                int flow_i = i- flowatxy.y;
                int flow_j = realJ- flowatxy.x;
                if(flow_i<0)
                    flow_i=0;
                if(flow_j<0)
                    flow_j=0;
				int preCenterColor=(imageL_pre->imageData[flow_i*wid+flow_j*3]);
				float weight_wall = 0;
				float momentum = 0;
				for(int p=0;p<NSIZE;++p){
					for(int q=0;q<NSIZE;++q){
						int pre_i=(flow_i-NSIZE/2+p);
						int pre_j=(flow_j-NSIZE/2+q);
						int current_i= i-NSIZE/2+p;
						int current_j= realJ-NSIZE/2+q;
						if(pre_i>=0 && pre_i<height && current_i>=0 && current_i<height){
							if(pre_j>=0 && pre_j<width && current_j>=0 && current_j<width){
								float preDif=abs((imageL->imageData[(current_i*width+current_j)*3])-(imageL_pre->imageData[(pre_i*width+pre_j)*3]));
								preDifSum+=preDif;
								float weight = weiFunc(preCenterColor,(imageL->imageData[(current_i*width+current_j)*3]),gamma);
								preCount++;
								momentum += preDif*weight;
								weight_wall += weight;
							}
						}
					}
				}
				//ADD previous
				if(method==1){
					if(preCount==0)
						preWeight = 0;
					else{
						preWeight = FDF(preDifSum,preCount,FDF_CONSTANT,FDF_GAMMA);
						preWeightCount++;
					}
				}
				// ASW previous
				else{
					if(weight_wall==0)
						preWeight = 0;
					else{
						preWeight = FDF(momentum,weight_wall,FDF_CONSTANT_ASW,FDF_GAMMA);
						preWeightCount++;
					}
				}

				preDisp = (uchar)imageD_pre->imageData[flow_i*wid+flow_j*3];
				preDisp = (int)(((float)(preDisp*depth)/255.0)+0.5);
				//cerr<<preWeight<<endl;
				preWeightSum+=preWeight;
			}
			if (iter%clean==0){
				preWeight*=0.1;
			}
			float minNum=100000;
			float minNum_orig=100000;
			int mindis=depth;
			int mindis_orig=depth;
			for(int k=j,c=0;k>0,c<depth;k=k-(LR_minus*3),c++){
				float mome_w = 0;  //With affect of optical flow
				float mome_w_orig = 0; //origin
				float ww_wall = 0;
				for(int p=0;p<BSIZE;++p){
					if(((i-BSIZE/2)+p)>0 && ((i-BSIZE/2)+p)< hei){
						for(int q=0;q<BSIZE;++q){
							if((k-(BSIZE/2)*3+3*q)>0 && (k-(BSIZE/2)*3+3*q)<wid){
								int centerNewColor = imageR->imageData[i*wid+k];
								int ii=(i-BSIZE/2+p);
								int jj=(j-(BSIZE/2)*3+3*q);
								int kk=(k-(BSIZE/2)*3+3*q);
								float dist = sqrt((ii-i)*(ii-i)+(jj-j)*(jj-j)/9.0);
								dif=abs((imageL->imageData[ii*wid+jj])-(imageR->imageData[ii*wid+kk]));
								int LGW=(imageL->imageData[ii*wid+jj+3])-(imageL->imageData[ii*wid+jj-3]);
								int RGW=(imageR->imageData[ii*wid+kk+3])-(imageR->imageData[ii*wid+kk-3]);
								float ww_w = weiFunc(centerNewColor,(imageR->imageData[ii*wid+kk]),gamma,dist,BSIZE/2);
								float ww_o = weiFunc(centerColor, (imageL->imageData[ii*wid+jj]), gamma, dist,BSIZE/2);
								ww_wall+=ww_w*ww_o;
								float rho_w = (1-alpha)*min(dif,colorBound) + alpha*min((abs(LGW-RGW)),gradBound);
								mome_w += ww_w*ww_o*rho_w;
								mome_w_orig += ww_w*ww_o*rho_w;
								//mome_w += preWeight*(min(preRobust,abs(c-preDisp))); //Previos Frame Magic Goes here.
							}
							else{
								mome_w += 1000*BSIZE;
								mome_w_orig += 1000*BSIZE;
							}
						}
					}
					else{
					}
				}
				mome_w=mome_w/ww_wall;
				mome_w_orig/=ww_wall;
				if((float)preWeight*(min(preRobust,abs(c-preDisp)))!=0&&false){
					cerr<<"m:"<<mome_w<<endl;
					cerr<<"p:"<<preWeight*(float)(min(preRobust,abs(c-preDisp)))<<endl;
					cerr<<endl;
				}
				mome_w += preWeight*(min(preRobust,abs(c-preDisp)));// newer version constant set 20
				if(mome_w<minNum){
					minNum=mome_w;
					mindis=c;
				}
				if(mome_w_orig<minNum_orig){
					minNum_orig=mome_w_orig;
					mindis_orig=c;
				}
			}
			int newColor = (int)((255.0/depth)*mindis+0.5);
			int newColor_orig = (int)((255.0/depth)*mindis_orig+0.5);
			imageD->imageData[i*wid+j]=newColor_orig;
	      	imageD->imageData[i*wid+j+1]=newColor_orig;
     		imageD->imageData[i*wid+j+2]=newColor_orig;
			imageD_refine->imageData[i*wid+j]=newColor;
	      	imageD_refine->imageData[i*wid+j+1]=newColor;
     		imageD_refine->imageData[i*wid+j+2]=newColor;
			//cerr<<(uchar)imageD_refine->imageData[i*imageD->widthStep+j+2]<<endl;
    	}
	}
	//cerr<<"preWeight avg is "<<preWeightSum/preWeightCount<<endl;
	/*
	if(imageD_pre!=NULL)
		cvReleaseImage(&imageD_pre);
	if(imageL_pre!=NULL)
		cvReleaseImage(&imageL_pre);
	imageD_pre = cvCloneImage(imageD);
	imageL_pre = cvCloneImage(imageL);
	*/
}

void new_bp_frame_optical(IplImage *Left_Img, IplImage *Right_Img, IplImage *disp_Img, IplImage *imageL_pre,IplImage *imageD_pre, IplImage *imageD_refine, int depth,bool window=false,int bsize=3,bool isLeft=true, int iter=1, int clean=10, int speed=1, int method=1, int nsize=5){
	//BP(imageL,imageR,imageD,depth);
	

	int ndisp=depth;
	int width = Left_Img->width;
	int height = Left_Img->height;
	int widthstep = Left_Img->widthStep; // more, nChannels times the width
	int widthstep_Disp = disp_Img->widthStep;
	int nchan_Disp = disp_Img->nChannels;

	int disp_scale = 256 / ndisp;

	float *Cost_Buf;
	float *Mes_L_Buf;
	float *Mes_R_Buf;
	float *Mes_U_Buf;
	float *Mes_D_Buf;

	float  weight_A;
	float  weight_B;
	float  weight_C;
	float  weight_D;

	int Buf_size;

	IplImage * Img_L_Gary = cvCreateImage(cvGetSize(Left_Img), Left_Img->depth, 1);
	IplImage * Img_R_Gary = cvCreateImage(cvGetSize(Right_Img), Right_Img->depth, 1);

	IplImage * _disp_Img_F = cvCreateImage(cvGetSize(disp_Img), disp_Img->depth, 1);

	cvCvtColor(Left_Img, Img_L_Gary, CV_RGB2GRAY);
	cvCvtColor(Right_Img, Img_R_Gary, CV_RGB2GRAY);
	Mat flow;
	bool usingPreImg=true;
	//judge if it is the first frame
	if (imageD_pre == NULL ||imageD_pre==0){
		usingPreImg = false;
	}
    else{
        flow = get_optical(Left_Img, imageL_pre);
    }
	
	float *Cost_Pixel;
	float *Mes_L_Pixel;
	float *Mes_R_Pixel;
	float *Mes_U_Pixel;
	float *Mes_D_Pixel;
	float *Mes_Result_Pixel;

	Cost_Pixel  = new float[ndisp];
	Mes_L_Pixel = new float[ndisp];
	Mes_R_Pixel = new float[ndisp];
	Mes_U_Pixel = new float[ndisp];
	Mes_D_Pixel = new float[ndisp];
	Mes_Result_Pixel = new float[ndisp];
	
	Buf_size = ndisp*width*height;

	Cost_Buf  = new float[Buf_size];
	Mes_L_Buf = new float[Buf_size];
	Mes_R_Buf = new float[Buf_size];
	Mes_U_Buf = new float[Buf_size];
	Mes_D_Buf = new float[Buf_size];
	//


	//====================================AD_Cost
	//============Cost Refill=============
	//====================================
	int BSIZE = bsize;
	int NSIZE = nsize;
	float alpha= 0.2;
	float gamma = 10;
	int preRobust = PREROBUST;
	//Testing Two Method Avg!!!
	double preWeightSum=0;
	double preWeightCount=0;
	// I change i and j to merge with my code
	for (int i = 0;i < height;++i) {
		for (int j = 0;j < widthstep;j=j+3) {
			int centerColor = Left_Img->imageData[i*width+j];
			float preWeight = 0;
			int preDif = 0;
			int preDisp = 0;
			int preCount = 0;
			float preDifSum = 0;
			// Instead of looking for the same point color in previous color, we add some optical flow in it!
			if(usingPreImg){
				int realJ = (int)(j/3);
				const Point2f flowatxy = flow.at<Point2f>(i, realJ)*speed;
				int flow_i = i- flowatxy.y;
				int flow_j = realJ- flowatxy.x;
				if(flow_i<0)
					flow_i=0;
				if(flow_j<0)
					flow_j=0;
				if(flow_i>height)
					flow_i=height-1;
				if(flow_j>width)
					flow_j=width-1;
				int preCenterColor=(imageL_pre->imageData[(flow_i*width+flow_j)*3]);
				float weight_wall = 0;
				float momentum = 0;
				for(int p=0;p<NSIZE;++p){
					for(int q=0;q<NSIZE;++q){
						int pre_i=(flow_i-NSIZE/2+p);
						int pre_j=(flow_j-NSIZE/2+q);
						int current_i= i-NSIZE/2+p;
						int current_j= realJ-NSIZE/2+q;

						if(pre_i>=0 && pre_i<height && current_i>=0 && current_i<height){
							if(pre_j>=0 && pre_j<width && current_j>=0 && current_j<width){
								float preDif=abs((Left_Img->imageData[(current_i*width+current_j)*3])-(imageL_pre->imageData[(pre_i*width+pre_j)*3]));
								preDifSum+=preDif;
								float weight = weiFunc(preCenterColor,(Left_Img->imageData[(current_i*width+current_j)*3]),gamma);
								preCount++;
								momentum += preDif*weight;
								weight_wall += weight;
							}
						}
					}
				}
				//ADD previous
				if(method==1){
					if(preCount==0)
						preWeight = 0;
					else{
						preWeight = FDF(preDifSum,preCount,FDF_CONSTANT,FDF_GAMMA);
						preWeightCount++;
					}
				}
				// ASW previous
				else{
					if(weight_wall==0)
						preWeight = 0;
					else{
						preWeight = FDF(momentum,weight_wall,FDF_CONSTANT_ASW,FDF_GAMMA);
						preWeightCount++;
					}
				}
				preDisp = (uchar)imageD_pre->imageData[flow_i*widthstep+flow_j*3];
				preDisp = (int)(((float)(preDisp*depth)/255.0)+0.5);
				preWeightSum+=preWeight;
			}
			// preWeight=0; //for testing
			// if(usingPreImg)
			// 	cerr<<preWeight<<endl;
			for (int d = 0;d < ndisp;++d) {
				int Buf_addr = (i*width + j/3)*ndisp + d;
				Cost_Buf[Buf_addr] = ASW_Aggre(Img_L_Gary, Img_R_Gary, j/3, i, 3, d) + preWeight*(min(preRobust,abs(d-preDisp)));
			}
		}
	}
	//=======================From Left to Right===================
	for (int j = 0;j < height;++j) {
		for (int i = 0;i < width-1;++i) {
			for (int d = 0;d < ndisp;++d) {
				int addr_buf = (j*width+i)*ndisp + d;
				Cost_Pixel[d]  = Cost_Buf[addr_buf];
				Mes_L_Pixel[d] = Mes_L_Buf[addr_buf];
				
				Mes_U_Pixel[d] = Mes_U_Buf[addr_buf];
				Mes_D_Pixel[d] = Mes_D_Buf[addr_buf];
			}
			weight_A = 1;
			weight_B = 1;
			weight_C = 1;
			BP_Update(Mes_L_Pixel, Mes_U_Pixel, Mes_D_Pixel,Cost_Pixel, Mes_Result_Pixel, weight_A, weight_B, weight_C,8,ndisp);

			for (int d = 0;d < ndisp;++d) {
				int addr_buf = (j*width + i+1)*ndisp + d;

				Mes_L_Buf[addr_buf]= Mes_Result_Pixel[d];
			}
		}
	}
	//=======================From Right to Left===================
	for (int j = 0;j < height;++j) {
		for (int i = width-1;i >=1;--i) {
			for (int d = 0;d < ndisp;++d) {
				int addr_buf = (j*width + i)*ndisp + d;
				Cost_Pixel[d] = Cost_Buf[addr_buf];
				Mes_R_Pixel[d] = Mes_R_Buf[addr_buf];

				Mes_U_Pixel[d] = Mes_U_Buf[addr_buf];
				Mes_D_Pixel[d] = Mes_D_Buf[addr_buf];
			}
			weight_A = 1;
			weight_B = 1;
			weight_C = 1;
			BP_Update(Mes_R_Pixel, Mes_U_Pixel, Mes_D_Pixel, Cost_Pixel, Mes_Result_Pixel, weight_A, weight_B, weight_C, 8, ndisp);

			for (int d = 0;d < ndisp;++d) {
				int addr_buf = (j*width + i-1)*ndisp + d;

				Mes_R_Buf[addr_buf] = Mes_Result_Pixel[d];
			}

		}
	}

	//=======================From Up to Down===================
	
	for (int i = 0;i < width;++i) {
		for (int j = 0;j < height-1;++j) {
			for (int d = 0;d < ndisp;++d) {
				int addr_buf = (j*width + i)*ndisp + d;
				Cost_Pixel[d] = Cost_Buf[addr_buf];
				Mes_U_Pixel[d] = Mes_U_Buf[addr_buf];

				Mes_L_Pixel[d] = Mes_L_Buf[addr_buf];
				Mes_R_Pixel[d] = Mes_R_Buf[addr_buf];
			}
			weight_A = 1;
			weight_B = 1;
			weight_C = 1;
			BP_Update(Mes_R_Pixel, Mes_U_Pixel, Mes_D_Pixel, Cost_Pixel, Mes_Result_Pixel, weight_A, weight_B, weight_C, 8, ndisp);

			for (int d = 0;d < ndisp;++d) {
				int addr_buf = ((j+1)*width + i)*ndisp + d;

				Mes_U_Buf[addr_buf] = Mes_Result_Pixel[d];
			}

		}
	}

	for (int i = 0;i < width;++i) {
		for (int j = height - 1;j >= 1;--j) {
			for (int d = 0;d < ndisp;++d) {
				int addr_buf = (j*width + i)*ndisp + d;
				Cost_Pixel[d] = Cost_Buf[addr_buf];
				Mes_D_Pixel[d] = Mes_D_Buf[addr_buf];

				Mes_L_Pixel[d] = Mes_L_Buf[addr_buf];
				Mes_R_Pixel[d] = Mes_R_Buf[addr_buf];
			}
			weight_A = 1;
			weight_B = 1;
			weight_C = 1;
			BP_Update(Mes_R_Pixel, Mes_U_Pixel, Mes_D_Pixel, Cost_Pixel, Mes_Result_Pixel, weight_A, weight_B, weight_C, 8, ndisp);

			for (int d = 0;d < ndisp;++d) {
				int addr_buf = ((j - 1)*width + i)*ndisp + d;

				Mes_D_Buf[addr_buf] = Mes_Result_Pixel[d];
			}

		}
	}

	for (int i = 0;i < width;++i) {
		for (int j = height - 1;j >= 1;--j) {
			for (int d = 0;d < ndisp;++d) {
				int addr_buf = (j*width + i)*ndisp + d;
				Cost_Pixel[d] = Cost_Buf[addr_buf];
				Mes_D_Pixel[d] = Mes_D_Buf[addr_buf];
				Mes_U_Pixel[d] = Mes_U_Buf[addr_buf];
				Mes_L_Pixel[d] = Mes_L_Buf[addr_buf];
				Mes_R_Pixel[d] = Mes_R_Buf[addr_buf];
			}
			int addr_disp_Img = AddrGet(i, j, widthstep_Disp, nchan_Disp);
			weight_A =1;
			weight_B =1;
			weight_C =1;
			weight_D =1;
			
			disp_Img->imageData[addr_disp_Img] =disp_scale*BP_Disp_Deter(Mes_L_Pixel, Mes_R_Pixel, Mes_U_Pixel, Mes_D_Pixel, Cost_Pixel, weight_A, weight_B,weight_C, weight_D, ndisp);
			disp_Img->imageData[addr_disp_Img+1] = disp_Img->imageData[addr_disp_Img];
			disp_Img->imageData[addr_disp_Img+2] = disp_Img->imageData[addr_disp_Img];
		}
	}
	//cvShowImage("Output", disp_Img);
	//cvWaitKey(0); // wait
}
		


void disp_stereo_census_big_frame_optical(IplImage *imageL,IplImage *imageR,IplImage *imageD,IplImage *imageL_pre,IplImage *imageD_pre,int depth,bool window=false,int bsize=3, bool isLeft=true, int iter=1, int clean=10, int speed=5, bool doCensus=true){
    
    int LR_minus=1;
    // Flow multiplier
	bool usingPreImg = true;
    Mat flow;
	if (!isLeft){
		IplImage *temp = imageL;
		imageL = imageR;
		imageR = temp;
		LR_minus=-1;
	}
	if (imageD_pre == NULL || iter%clean==0){
		usingPreImg = false;
	}
    else{
        flow = get_optical(imageL, imageL_pre);
    }
	int BSIZE = bsize;
	float alpha= 0.8;

	if(!doCensus){
		alpha = 0;
	}
	float gamma = 10;
	int preRobust = PREROBUST;

	int colorBound = 10;
	cvSet(imageD, cvScalar(0,0,0));
	int mindis;
	float minNum;
	int dif;
  	for(int i=0;i<imageL->height;i++){
		for(int j=0;j<imageL->widthStep;j=j+3){
			int wid = imageL->widthStep;
			int centerColor = imageL->imageData[i*wid+j];
			int preWeight = 0;
			int preDif = 0;
			int preDisp = 0;
			int preCount = 0;
			// Instead of looking for the same point color in previous color, we add some optical flow in it!
			if(usingPreImg){
                int realJ = (int)(j/3);
                const Point2f flowatxy = flow.at<Point2f>(i, realJ)*speed;
                int flow_i = i- flowatxy.y;
                int flow_j = realJ- flowatxy.x;
                if(flow_i<0)
                    flow_i=0;
                if(flow_j<0)
                    flow_j=0;
                if(flow_i>=imageL->height)
                    flow_i=imageL->height-1;
                if(flow_j>=imageL->widthStep/3)
                    flow_j=imageL->widthStep/3-1;
				for(int p=0;p<BSIZE;++p){
					if(((flow_i-BSIZE/2)+p)>=0 && ((flow_i-BSIZE/2)+p)<imageL->height){
						for(int q=0;q<BSIZE;++q){
							if(((flow_j-BSIZE/2)+q)>=0 && ((flow_j-BSIZE/2)+q)<imageL->widthStep/3){
								int ii=(flow_i-BSIZE/2+p);
								int jj=(flow_j-(BSIZE/2)+q)*3;
								preDif+=abs((imageL->imageData[ii*wid+jj])-(imageL_pre->imageData[ii*wid+jj]));
								preCount++;
							}
						}
					}
				}
				preWeight = FDF(preDif,preCount,FDF_CONSTANT);
				preDisp = imageD_pre->imageData[flow_i*wid+flow_j*3];
				preDisp = (int)((float)preDisp*depth/255);
			}
			minNum=100000;
			mindis=depth;
			for(int k=j,c=0;k>0,c<depth;k=k-(LR_minus*3),c++){
				float mome_w = 0;
				float ww_wall = 0;
				for(int p=0;p<BSIZE;++p){
					if(((i-BSIZE/2)+p)>0){
						for(int q=0;q<BSIZE;++q){
							if(((k-BSIZE/2)+q-1)>0){
								int centerNewColor = imageR->imageData[i*wid+k];
								int ii=(i-BSIZE/2+p);
								int jj=(j-(BSIZE/2)*3+3*q);
								int kk=(k-(BSIZE/2)*3+3*q);
								dif=abs((imageL->imageData[ii*wid+jj])-(imageR->imageData[ii*wid+kk]));
								//TODO HERE//
								int LGW=(imageL->imageData[ii*wid+jj+3])-(imageL->imageData[ii*wid+jj-3]);
								int RGW=(imageR->imageData[ii*wid+kk+3])-(imageR->imageData[ii*wid+kk-3]);
								
								float censusAll = 0;
								int validPoint = 0;
								int Lcenter = imageL->imageData[ii*wid+jj];
								int Rcenter = imageR->imageData[ii*wid+kk];
								if(doCensus){
									for (int c_i=-1; c_i<=1; c_i++){
										for(int c_j=-3; c_j<=3; c_j+=3){
											//cerr<<kk+c_i<<","<<c_j<<endl;
											if(ii+c_i>=0 && kk+c_j>=0 && ii+c_i<(imageL->height) && kk+c_j<(wid)){
												validPoint++;
												bool Lcompare = false;
												bool Rcompare = true;
												Lcompare = (imageL->imageData[(ii+c_i)*wid+(jj+c_j)]>=Lcenter);
												Rcompare = (imageR->imageData[(ii+c_i)*wid+(kk+c_j)]>=Rcenter);
												if(Lcompare!=Rcompare){
													censusAll++;
												}
											}
										}
									}
									if(validPoint==0)
										censusAll=10000;
									else
										censusAll/=validPoint;
								}
								//TODO END//
								float ww_w = weiFunc(centerColor,(imageR->imageData[ii*wid+kk]),gamma);
								ww_wall+=ww_w;
								float rho_w = (1-alpha)*min(dif,colorBound) + alpha*(censusAll);
								mome_w += ww_w*rho_w;
								mome_w += preWeight*(min(preRobust,abs(c-preDisp)));
							}
							else{
								mome_w += 1000*BSIZE;
							}
						}
					}
					else{
						mome_w += 1000*BSIZE;
					}
				}
				mome_w=mome_w/ww_wall;
				if(mome_w<minNum){
					minNum=mome_w;
					mindis=c;
				}
			}
			int newColor = (255/depth)*mindis;
			imageD->imageData[i*imageD->widthStep+j]=newColor;
	      	imageD->imageData[i*imageD->widthStep+j+1]=newColor;
     		imageD->imageData[i*imageD->widthStep+j+2]=newColor;
    	}
	}
	// Comment because test affine will use it
	// if(imageD_pre!=NULL)
	// 	cvReleaseImage(&imageD_pre);
	// if(imageL_pre!=NULL)
	// 	cvReleaseImage(&imageL_pre);
	// imageD_pre = cvCloneImage(imageD);
	// imageL_pre = cvCloneImage(imageL);
}
/*
void disp_stereo_bp_big_frame_optical(IplImage *imageL,IplImage *imageR,IplImage *imageD,IplImage *imageL_pre,IplImage *imageD_pre,int depth,bool window=false,int bsize=3,bool isLeft=true, int iter=1, int clean=10, int speed=5){
	BP::parameters param(BP::MIDDLEBURY);
    // if (no_interp) {
	// //param = Elas::parameters(Elas::ROBOTICS);
	// // don't use full 'robotics' setting, just the parameter to fill gaps
    //     param.ipol_gap_width = 3;
    // }
    param.postprocess_only_left = false;
    param.disp_max = 255;
    BP bp(param);
	
	image<uchar> *I1,*I2;
	int wid = imageL->widthStep;
	int hei = imageL->height;
	I1 = new image<uchar>(wid/3, hei);
	I2 = new image<uchar>(wid/3, hei);
	for(int i=0; i<hei;++i){
		for(int j=0; j<wid; j=j+3){
			I1->data[i*wid/3+j/3]=(unsigned char)imageL->imageData[i*wid+j];
			I2->data[i*wid/3+j/3]=(unsigned char)imageR->imageData[i*wid+j];
		}
	}
	int32_t width  = I1->width();
    int32_t height = I1->height();

    // allocate memory for disparity images
    const int32_t dims[3] = {width,height,width}; // bytes per line = width
    float* D1_data = new float[width*height]();
    float* D2_data = new float[width*height]();

	bp.process(I1->data,I2->data,D1_data,D2_data, dims, imageL_pre, imageD_pre);
	delete I1;
    delete I2;
	for(int i=0; i<hei;++i){
		for(int j=0; j<wid; j=j+3){
			imageD->imageData[i*wid+j]=(int)D1_data[i*wid/3+j/3];
			imageD->imageData[i*wid+j+1]=(int)D1_data[i*wid/3+j/3];
			imageD->imageData[i*wid+j+2]=(int)D1_data[i*wid/3+j/3];
		}
	}
	delete [] D1_data;
	delete [] D2_data;
}
*/


#endif
