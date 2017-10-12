
/***********************************************************
* Domain transform filtering implementation
************************************************************
* This code is an implementation of the paper [Gastal and Oliveira 2011].
* This paper proposes an edge-preserving filter,
* which is effectively parallelizable. This filter transforms
* the domain of the filter function, and performs linear filtering
* to the transfomed domain.
*
* usage: DomainTransformFiltering.exe [input_image] ([sigma_s] [sigma_r] [maxiter])
* (last three arguments are optional)
*
* This code is programmed by 'tatsy'. You can use this
* code for any purpose :-)
* If you are satisfied with the program and kind enough of
* cheering me up, please contact me from my github account
* "https://github.com/tatsy/". Thanks!
************************************************************/

#include <iostream>
#include <vector>
#include <ctime>
#include <omp.h>
#include "DFT.h"
using namespace std;

//#include "opencv2/opencv.hpp"



// Recursive filter for vertical direction
void recursiveFilterVertical(float* out, float* dct, double sigma_H,const int32_t* dims) {
	int width  = dims[0];
        int height = dims[1];
	double a = exp(-sqrt(2.0) / sigma_H);
	//printf("dim in Vertical %d\n", dim);
	//cv::Mat V;
	//dct.convertTo(V, CV_64FC1);
	for (int x = 0; x<width; x++) {
		for (int y = 0; y<height - 1; y++) {
			dct[y*width+x] = pow(a, dct[y*width+x]);
		}
	}

	// if openmp is available, compute in parallel
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (int x = 0; x<width; x++) {
		for (int y = 1; y<height; y++) {
			double p = dct[(y-1)*width+x];
			//for (int c = 0; c<dim; c++) {
				double val1 = out[y*width+x];
				double val2 = out[(y-1)*width+x];
				out[y*width+x] = val1 + p * (val2 - val1);
			//}
		}

		for (int y = height - 2; y >= 0; y--) {
			double p = dct[(y)*width+x];
			//for (int c = 0; c<dim; c++) {
				double val1 = out[y*width+x];
				double val2 = out[(y+1)*width+x];
				out[y*width+x] = val1 + p * (val2 - val1);
			//}
		}
	}
}


// Recursive filter for horizontal direction
void recursiveFilterHorizontal(float* out,float* dct, double sigma_H,const int32_t* dims) {
	int width  = dims[0];
        int height = dims[1];
	//int dim = out.channels();
	double a = exp(-sqrt(2.0) / sigma_H);
	//printf("dim in Horizontal %d\n", dim);

	for (int x = 0; x<width - 1; x++) {
		for (int y = 0; y<height; y++) {
			dct[y*width+x] = pow(a, dct[y*width+x]);
		}
	}

	// if openmp is available, compute in parallel
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (int y = 0; y<height; y++) {
		for (int x = 1; x<width; x++) {
			double p = dct[y*width+x-1];
			//for (int c = 0; c<dim; c++) {
				double val1 = out[y*width+x];
				double val2 = out[y*width+x-1];
				out[y*width+x] = val1 + p * (val2 - val1);
			//}
		}

		for (int x = width - 2; x >= 0; x--) {
			double p = dct[y*width+x];
			//for (int c = 0; c<dim; c++) {
				double val1 = out[y*width+x];
				double val2 = out[y*width+x+1];
				out[y*width+x]= val1 + p * (val2 - val1);
			//}
		}
	}
}


// Domain transform filtering
void domainTransformFilter(float* Depth ,float* Depth_F, uint8_t* Left, double sigma_s, double sigma_r, int maxiter,const int32_t* dims) {
	//assert(img.depth() == CV_64F && joint.depth() == CV_64F);
	//assert(img.depth() == CV_8U && joint.depth() == CV_8U);

	//int width = img.cols;
	//int height = img.rows;
	//int dim = img.channels();

        int width  = dims[0];
        int height = dims[1];
        int bpl    = width + 15-(width-1)%16;

	// compute derivatives of transformed domain "dct"
	// and a = exp(-sqrt(2) / sigma_H) to the power of "dct"
	
        float *dctx;
        dctx  = (float*)_mm_malloc(width*height*sizeof(float),64);

        float *dcty;
        dcty  = (float*)_mm_malloc(width*height*sizeof(float),64);

	//cv::Mat dctx = cv::Mat(height, width - 1, CV_64FC1);
	//cv::Mat dcty = cv::Mat(height - 1, width, CV_64FC1);
	//cv::Mat dctx = cv::Mat(height, width - 1, CV_8UC1);
	//cv::Mat dcty = cv::Mat(height - 1, width, CV_8UC1);

	
	double ratio = sigma_s / sigma_r;

	for (int y = 0; y<height; y++) {
		for (int x = 0; x<width - 1; x++) {
			double accum = 0.0;
			//for (int c = 0; c<dim; c++) {
				int Guided_addr = Get_addr(bpl,x,y) ;
				int Guided_addr_next = Get_addr(bpl,x+1,y) ;
				accum += abs((double)Left[Guided_addr_next]/(double)255  - (double)Left[Guided_addr]/(double)255);
			//}
			dctx[y*width+x] = 1.0 + ratio * accum;
		}
	}
	//printf("Finish dctx\n");
	for (int x = 0; x<width; x++) {
		for (int y = 0; y<height - 1; y++) {
			double accum = 0.0;
			//for (int c = 0; c<dim; c++) {
				int Guided_addr = Get_addr(bpl,x,y) ;
				int Guided_addr_next = Get_addr(bpl,x,y+1) ;
				accum += abs((double)Left[Guided_addr_next]/(double)255 - (double)Left[Guided_addr]/(double)255);
			//}
			dcty[y*width+x] = 1.0 + ratio * accum;
		}
	}
	//printf("Finish dcty\n");
	// Apply recursive folter maxiter times
	//img.convertTo(out, CV_MAKETYPE(CV_64F, dim));
	//img.convertTo(out, CV_MAKETYPE(CV_64F, dim));
	for (int y = 0; y<height; y++) {
		for (int x = 0; x<width - 1; x++) {
			Depth_F[y*width+x]  = Depth[y*width+x];
		}
	}
	for (int i = 0; i<maxiter; i++) {
		printf("%d-th iteration\n", i);
		double sigma_H = sigma_s * sqrt(3.0) * pow(2.0, maxiter - i - 1) / sqrt(pow(4.0, maxiter) - 1.0);
		recursiveFilterHorizontal(Depth_F, dctx, sigma_H,dims);
		//printf("Finish X Filter\n");
		recursiveFilterVertical(Depth_F, dcty, sigma_H,dims);
		//printf("Finish Y Filter\n");
		//recursiveFilterHorizontal(out, dctx, sigma_H);
		//recursiveFilterVertical(out, dcty, sigma_H);
	}
	//printf("DIM of out is %d\n", out.dims);
	//printf("Original DIM of out is %d\n", dim);
}


/*
// Main function
int main(int argc, char** argv) {
	// Check arguments
	if (argc <= 1) {
		cout << "usage: DomainTransformFiltering.exe [input_image] ([sigma_s] [sigma_r] [maxiter])" << endl;
		return -1;
	}

	// Load image
	cv::Mat img = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);
	if (img.empty()) {
		cout << "Failed to load image "" << argv[1] << """ << endl;
		return -1;
	}

	// change depth
	img.convertTo(img, CV_64FC3, 1.0 / 255.0);

	// Parameter set
	const double sigma_s = argc <= 2 ? 25.0 : atof(argv[2]);
	const double sigma_r = argc <= 3 ? 0.1 : atof(argv[3]);
	const int    maxiter = argc <= 4 ? 10 : atoi(argv[4]);

	cout << "[ Parameters ]" << endl;
	cout << "  * sigma_s = " << sigma_s << endl;
	cout << "  * sigma_r = " << sigma_r << endl;
	cout << "  * maxiter = " << maxiter << endl;
	cout << endl;

	// Call domain transform filter
	CLOCK_START
		cv::Mat out;
	domainTransformFilter(img, out, img, sigma_s, sigma_r, maxiter);
	CLOCK_END

		// Show results
		cv::imshow("Input", img);
	cv::imshow("Output", out);
	cv::waitKey(0);
	cv::destroyAllWindows();
}
*/
