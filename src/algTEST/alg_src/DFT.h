#ifndef DFT__H__

#define DFT_H__

#include <iostream>
#include <vector>
#include <ctime>
#include <omp.h>
#include "BP.h"
#include <math.h>
using namespace std;

//#include "opencv2/opencv.hpp"

//long t, elapsed;
//#define CLOCK_START t = clock();
//#define CLOCK_END cout << "[ Time ]" << endl << "  " << ((clock() - t) / 1000.0) << " sec" << endl;


void recursiveFilterVertical(float* out, float*dct, double sigma_H,const int32_t* dims);

void recursiveFilterHorizontal(float* out, float* dct, double sigma_H,const int32_t* dims);

void domainTransformFilter(float* Depth, float* Depth_F, uint8_t* Left, double sigma_s, double sigma_r, int maxiter,const int32_t* dims);

inline int Get_addr(int width,int x, int y){return y*width+x;}



#endif
