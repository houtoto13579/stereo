#ifndef BP_MOTION_H
#define BP_MOTION_H

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include "filter.h"
#include "BP/Optimization.h"

#define BP_PREROBUST 50 //original 50
#define BP_FDF_CONSTANT 0.15 
#define BP_FDF_CONSTANT_ASW 0.1//best 0.1 
#define BP_FDF_GAMMA 0.001 //best 0.01

using namespace std;
using namespace cv;



float g_temporal(int preDisp, int curDisp, float var=12){
    if(preDisp==-1){
        return 1;
    }
    float ans;
    float pi = 3.14;
    float d = preDisp;
    float x = curDisp;
    ans = 1-(1/(sqrt(2*pi*var)))*exp(-(x-d)*(x-d)/(2*var));
    return ans;
}

void new_bp(IplImage *Left_Img, IplImage *Right_Img, IplImage *disp_Img, IplImage *imageL_pre,IplImage *imageD_pre, IplImage *imageD_refine, int depth,bool window=false,int bsize=3,bool isLeft=true, int iter=1, int clean=10, int speed=1, int method=1, int nsize=5){
	BP(Left_Img,Right_Img,disp_Img,depth,bsize);
}

void new_bp_frame_optical(IplImage *Left_Img, IplImage *Right_Img, IplImage *disp_Img, IplImage *imageL_pre,IplImage *imageD_pre, IplImage *imageD_refine, int depth,bool window=false,int bsize=3,bool isLeft=true, int iter=1, int clean=10, int speed=1, int method=1, int nsize=5){
	//====================================AD_Cost
	//============Cost Refill=============
	//====================================
	
	int lamda = 2;
	int Iteration = 2;
	int f=1;
	int width = Left_Img->width;
	int height = Left_Img->height;
	int widthstep = Left_Img->widthStep;
	
	int widthstep_Disp = disp_Img->widthStep;
	int nchan_Disp = disp_Img->nChannels;

	int ndisp=depth;
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
	float *Temporal_Cost_Pixel;
	float *Mes_L_Pixel;
	float *Mes_R_Pixel;
	float *Mes_U_Pixel;
	float *Mes_D_Pixel;
	float *Mes_Result_Pixel;

	Cost_Pixel  = new float[ndisp];
	Temporal_Cost_Pixel = new float[ndisp];
	Mes_L_Pixel = new float[ndisp];
	Mes_R_Pixel = new float[ndisp];
	Mes_U_Pixel = new float[ndisp];
	Mes_D_Pixel = new float[ndisp];
	Mes_Result_Pixel = new float[ndisp];

	#ifdef Tile_BP



	#endif

	#ifndef Tile_BP
	
	Buf_size = ndisp*width*height;

	Cost_Buf  = new float[Buf_size];
	Mes_L_Buf = new float[Buf_size];
	Mes_R_Buf = new float[Buf_size];
	Mes_U_Buf = new float[Buf_size];
	Mes_D_Buf = new float[Buf_size];
	//====================================
	//============Cost Refill=============
	//====================================
	int BSIZE = bsize;
	int NSIZE = nsize;
	float alpha= 0.2;
	float gamma = 10;
	int preRobust = BP_PREROBUST;
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
						preWeight = FDF(preDifSum,preCount,BP_FDF_CONSTANT,BP_FDF_GAMMA);
						preWeightCount++;
					}
				}
				// ASW previous
				else{
					if(weight_wall==0)
						preWeight = 0;
					else{
						preWeight = FDF(momentum,weight_wall,BP_FDF_CONSTANT_ASW,BP_FDF_GAMMA);
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
				if (j/3 < d)
					Cost_Buf[Buf_addr] = 999;
				else
					Cost_Buf[Buf_addr] = ASW_Aggre(Img_L_Gary, Img_R_Gary, j/3, i, bsize, d)+ preWeight*(min(preRobust,abs(d-preDisp)));
			}
		}
	}
	//cerr<<"\n"<<Cost_Buf[0]<<endl;
	for (int iter = 0;iter < Iteration;++iter) {
		//=======================From Left to Right===================
		for (int j = 0;j < height;++j) {
			for (int i = 0;i < width - 1;++i) {
				for (int d = 0;d < ndisp;++d) {
					int addr_buf = (j*width + i)*ndisp + d;
					Cost_Pixel[d] = Cost_Buf[addr_buf];
					Mes_L_Pixel[d] = Mes_L_Buf[addr_buf];

					Mes_U_Pixel[d] = Mes_U_Buf[addr_buf];
					Mes_D_Pixel[d] = Mes_D_Buf[addr_buf];
				}
				weight_A = 1;
				weight_B = 1;
				weight_C = 1;
				int addr_l = AddrGet(i - 1, j, Img_L_Gary->widthStep, Img_L_Gary->nChannels);
				int addr_u = AddrGet(i, j - 1, Img_L_Gary->widthStep, Img_L_Gary->nChannels);
				int addr_d = AddrGet(i, j + 1, Img_L_Gary->widthStep, Img_L_Gary->nChannels);
				int addr_p = AddrGet(i, j, Img_L_Gary->widthStep, Img_L_Gary->nChannels);
				unsigned char color_p;
				unsigned char color_q;
				if (i == 0) {
					weight_A = 0;
				}
				else {
					color_p = (unsigned char)Img_L_Gary->imageData[addr_p];
					color_q = (unsigned char)Img_L_Gary->imageData[addr_l];
					weight_A = WeiGet(color_p, color_q, 5);
				}
				if (j == 0) {
					weight_B = 0;
				}
				else {
					color_p = (unsigned char)Img_L_Gary->imageData[addr_p];
					color_q = (unsigned char)Img_L_Gary->imageData[addr_u];
					weight_B = WeiGet(color_p, color_q, 5);
				}
				if (j == height - 1) {
					weight_C = 0;
				}
				else {
					color_p = (unsigned char)Img_L_Gary->imageData[addr_p];
					color_q = (unsigned char)Img_L_Gary->imageData[addr_d];
					weight_C = WeiGet(color_p, color_q, 5);
				}

				BP_Update(Mes_L_Pixel, Mes_U_Pixel, Mes_D_Pixel, Cost_Pixel, Mes_Result_Pixel, weight_A, weight_B, weight_C, lamda, ndisp);

				for (int d = 0;d < ndisp;++d) {
					int addr_buf = (j*width + i + 1)*ndisp + d;

					Mes_L_Buf[addr_buf] = Mes_Result_Pixel[d];
				}
			}
		}
		//=======================From Right to Left===================
		for (int j = 0;j < height;++j) {
			for (int i = width - 1;i >= 1;--i) {
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

				int addr_r = AddrGet(i + 1, j, Img_L_Gary->widthStep, Img_L_Gary->nChannels);
				int addr_u = AddrGet(i, j - 1, Img_L_Gary->widthStep, Img_L_Gary->nChannels);
				int addr_d = AddrGet(i, j + 1, Img_L_Gary->widthStep, Img_L_Gary->nChannels);
				int addr_p = AddrGet(i, j, Img_L_Gary->widthStep, Img_L_Gary->nChannels);
				unsigned char color_p;
				unsigned char color_q;
				if (i == width - 1) {
					weight_A = 0;
				}
				else {
					color_p = (unsigned char)Img_L_Gary->imageData[addr_p];
					color_q = (unsigned char)Img_L_Gary->imageData[addr_r];
					weight_A = WeiGet(color_p, color_q, 5);
				}
				if (j == 0) {
					weight_B = 0;
				}
				else {
					color_p = (unsigned char)Img_L_Gary->imageData[addr_p];
					color_q = (unsigned char)Img_L_Gary->imageData[addr_u];
					weight_B = WeiGet(color_p, color_q, 5);
				}
				if (j == height - 1) {
					weight_C = 0;
				}
				else {
					color_p = (unsigned char)Img_L_Gary->imageData[addr_p];
					color_q = (unsigned char)Img_L_Gary->imageData[addr_d];
					weight_C = WeiGet(color_p, color_q, 5);
				}
				BP_Update(Mes_R_Pixel, Mes_U_Pixel, Mes_D_Pixel, Cost_Pixel, Mes_Result_Pixel, weight_A, weight_B, weight_C, lamda, ndisp);

				for (int d = 0;d < ndisp;++d) {
					int addr_buf = (j*width + i - 1)*ndisp + d;

					Mes_R_Buf[addr_buf] = Mes_Result_Pixel[d];
				}

			}
		}

		//=======================From Up to Down===================

		for (int i = 0;i < width;++i) {
			for (int j = 0;j < height - 1;++j) {
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

				int addr_u = AddrGet(i, j - 1, Img_L_Gary->widthStep, Img_L_Gary->nChannels);
				int addr_l = AddrGet(i - 1, j, Img_L_Gary->widthStep, Img_L_Gary->nChannels);
				int addr_r = AddrGet(i + 1, j, Img_L_Gary->widthStep, Img_L_Gary->nChannels);
				int addr_p = AddrGet(i, j, Img_L_Gary->widthStep, Img_L_Gary->nChannels);
				unsigned char color_p;
				unsigned char color_q;
				if (j == 0) {
					weight_A = 0;
				}
				else {
					color_p = (unsigned char)Img_L_Gary->imageData[addr_p];
					color_q = (unsigned char)Img_L_Gary->imageData[addr_u];
					weight_A = WeiGet(color_p, color_q, 5);
				}
				if (i == 0) {
					weight_B = 0;
				}
				else {
					color_p = (unsigned char)Img_L_Gary->imageData[addr_p];
					color_q = (unsigned char)Img_L_Gary->imageData[addr_l];
					weight_B = WeiGet(color_p, color_q, 5);
				}
				if (i == width - 1) {
					weight_C = 0;
				}
				else {
					color_p = (unsigned char)Img_L_Gary->imageData[addr_p];
					color_q = (unsigned char)Img_L_Gary->imageData[addr_r];
					weight_C = WeiGet(color_p, color_q, 5);
				}

				BP_Update(Mes_U_Pixel, Mes_L_Pixel, Mes_R_Pixel, Cost_Pixel, Mes_Result_Pixel, weight_A, weight_B, weight_C, lamda, ndisp);

				for (int d = 0;d < ndisp;++d) {
					int addr_buf = ((j + 1)*width + i)*ndisp + d;

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

				int addr_d = AddrGet(i, j + 1, Img_L_Gary->widthStep, Img_L_Gary->nChannels);
				int addr_l = AddrGet(i - 1, j, Img_L_Gary->widthStep, Img_L_Gary->nChannels);
				int addr_r = AddrGet(i + 1, j, Img_L_Gary->widthStep, Img_L_Gary->nChannels);
				int addr_p = AddrGet(i, j, Img_L_Gary->widthStep, Img_L_Gary->nChannels);
				unsigned char color_p;
				unsigned char color_q;
				if (j == height - 1) {
					weight_A = 0;
				}
				else {
					color_p = (unsigned char)Img_L_Gary->imageData[addr_p];
					color_q = (unsigned char)Img_L_Gary->imageData[addr_d];
					weight_A = WeiGet(color_p, color_q, 5);
				}
				if (i == 0) {
					weight_B = 0;
				}
				else {
					color_p = (unsigned char)Img_L_Gary->imageData[addr_p];
					color_q = (unsigned char)Img_L_Gary->imageData[addr_l];
					weight_B = WeiGet(color_p, color_q, 5);
				}
				if (i == width - 1) {
					weight_C = 0;
				}
				else {
					color_p = (unsigned char)Img_L_Gary->imageData[addr_p];
					color_q = (unsigned char)Img_L_Gary->imageData[addr_r];
					weight_C = WeiGet(color_p, color_q, 5);
				}

				BP_Update(Mes_D_Pixel, Mes_L_Pixel, Mes_R_Pixel, Cost_Pixel, Mes_Result_Pixel, weight_A, weight_B, weight_C, lamda, ndisp);

				for (int d = 0;d < ndisp;++d) {
					int addr_buf = ((j - 1)*width + i)*ndisp + d;

					Mes_D_Buf[addr_buf] = Mes_Result_Pixel[d];
				}

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

			int addr_u = AddrGet(i, j - 1, Img_L_Gary->widthStep, Img_L_Gary->nChannels);
			int addr_d = AddrGet(i , j+1, Img_L_Gary->widthStep, Img_L_Gary->nChannels);
			int addr_l = AddrGet(i - 1, j, Img_L_Gary->widthStep, Img_L_Gary->nChannels);
			int addr_r = AddrGet(i + 1, j, Img_L_Gary->widthStep, Img_L_Gary->nChannels);
			int addr_p = AddrGet(i, j, Img_L_Gary->widthStep, Img_L_Gary->nChannels);
			unsigned char color_p;
			unsigned char color_q;
			if (j == 0) {
				weight_C = 0;
			}
			else {
				color_p = Img_L_Gary->imageData[addr_p];
				color_q = Img_L_Gary->imageData[addr_u];
				weight_C = WeiGet(color_p, color_q, 5);
			}
			if (j == height-1) {
				weight_D = 0;
			}
			else {
				color_p = Img_L_Gary->imageData[addr_p];
				color_q = Img_L_Gary->imageData[addr_d];
				weight_D = WeiGet(color_p, color_q, 5);
			}
			if (i == 0) {
				weight_A = 0;
			}
			else {
				color_p = Img_L_Gary->imageData[addr_p];
				color_q = Img_L_Gary->imageData[addr_l];
				weight_A = WeiGet(color_p, color_q, 5);
			}
			if (i == width - 1) {
				weight_B = 0;
			}
			else {
				color_p = Img_L_Gary->imageData[addr_p];
				color_q = Img_L_Gary->imageData[addr_r];
				weight_B = WeiGet(color_p, color_q, 5);
			}

			disp_Img->imageData[addr_disp_Img] =disp_scale*BP_Disp_Deter(Mes_L_Pixel, Mes_R_Pixel, Mes_U_Pixel, Mes_D_Pixel, Cost_Pixel, weight_A, weight_B,weight_C, weight_D, ndisp);
			disp_Img->imageData[addr_disp_Img+1] = disp_Img->imageData[addr_disp_Img];
			disp_Img->imageData[addr_disp_Img+2] = disp_Img->imageData[addr_disp_Img];
		}
	}
	delete[]Cost_Buf;
	delete[]Mes_L_Buf;
	delete[]Mes_R_Buf;
	delete[]Mes_U_Buf;
	delete[]Mes_D_Buf;
	#endif
}

void new_bp_temporal(IplImage *Left_Img, IplImage *Right_Img, IplImage *disp_Img, IplImage *imageL_pre,IplImage *imageD_pre, IplImage *imageD_refine, int depth,bool window=false,int bsize=3,bool isLeft=true, int iter=1, int clean=10, int speed=1, int method=1, int nsize=5){
	//BP(imageL,imageR,imageD,depth);
	int lamda = 2;
	int Iteration = 2;
	int f=1;
	int width = Left_Img->width;
	int height = Left_Img->height;
	int widthstep = Left_Img->widthStep;
	
	int widthstep_Disp = disp_Img->widthStep;
	int nchan_Disp = disp_Img->nChannels;

	int ndisp=depth;
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
	float *Temporal_Cost_Pixel;
	float *Mes_L_Pixel;
	float *Mes_R_Pixel;
	float *Mes_U_Pixel;
	float *Mes_D_Pixel;
	float *Mes_Result_Pixel;

	Cost_Pixel  = new float[ndisp];
	Temporal_Cost_Pixel = new float[ndisp];
	Mes_L_Pixel = new float[ndisp];
	Mes_R_Pixel = new float[ndisp];
	Mes_U_Pixel = new float[ndisp];
	Mes_D_Pixel = new float[ndisp];
	Mes_Result_Pixel = new float[ndisp];

	#ifdef Tile_BP



	#endif

	#ifndef Tile_BP
	
	Buf_size = ndisp*width*height;

	Cost_Buf  = new float[Buf_size];
	Mes_L_Buf = new float[Buf_size];
	Mes_R_Buf = new float[Buf_size];
	Mes_U_Buf = new float[Buf_size];
	Mes_D_Buf = new float[Buf_size];
	//====================================
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
			int preDisp = -1;
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
				preDisp = (uchar)imageD_pre->imageData[flow_i*widthstep+flow_j*3];
				preDisp = (int)(((float)(preDisp*depth)/255.0)+0.5);
			}
			for (int d = 0;d < ndisp;++d) {
				int Buf_addr = (i*width + j/3)*ndisp + d;
				
				if (j/3 < d)
					Cost_Buf[Buf_addr] = 999;
				else
					Cost_Buf[Buf_addr] = ASW_Aggre(Img_L_Gary, Img_R_Gary, j/3, i, bsize, d)*g_temporal(preDisp,d);
			}
		}
	}
	//cerr<<"\n"<<Cost_Buf[0]<<endl;
	for (int iter = 0;iter < Iteration;++iter) {
		//=======================From Left to Right===================
		for (int j = 0;j < height;++j) {
			for (int i = 0;i < width - 1;++i) {
				for (int d = 0;d < ndisp;++d) {
					int addr_buf = (j*width + i)*ndisp + d;
					Cost_Pixel[d] = Cost_Buf[addr_buf];
					Mes_L_Pixel[d] = Mes_L_Buf[addr_buf];

					Mes_U_Pixel[d] = Mes_U_Buf[addr_buf];
					Mes_D_Pixel[d] = Mes_D_Buf[addr_buf];
				}
				weight_A = 1;
				weight_B = 1;
				weight_C = 1;
				int addr_l = AddrGet(i - 1, j, Img_L_Gary->widthStep, Img_L_Gary->nChannels);
				int addr_u = AddrGet(i, j - 1, Img_L_Gary->widthStep, Img_L_Gary->nChannels);
				int addr_d = AddrGet(i, j + 1, Img_L_Gary->widthStep, Img_L_Gary->nChannels);
				int addr_p = AddrGet(i, j, Img_L_Gary->widthStep, Img_L_Gary->nChannels);
				unsigned char color_p;
				unsigned char color_q;
				if (i == 0) {
					weight_A = 0;
				}
				else {
					color_p = (unsigned char)Img_L_Gary->imageData[addr_p];
					color_q = (unsigned char)Img_L_Gary->imageData[addr_l];
					weight_A = WeiGet(color_p, color_q, 5);
				}
				if (j == 0) {
					weight_B = 0;
				}
				else {
					color_p = (unsigned char)Img_L_Gary->imageData[addr_p];
					color_q = (unsigned char)Img_L_Gary->imageData[addr_u];
					weight_B = WeiGet(color_p, color_q, 5);
				}
				if (j == height - 1) {
					weight_C = 0;
				}
				else {
					color_p = (unsigned char)Img_L_Gary->imageData[addr_p];
					color_q = (unsigned char)Img_L_Gary->imageData[addr_d];
					weight_C = WeiGet(color_p, color_q, 5);
				}

				BP_Update(Mes_L_Pixel, Mes_U_Pixel, Mes_D_Pixel, Cost_Pixel, Mes_Result_Pixel, weight_A, weight_B, weight_C, lamda, ndisp);

				for (int d = 0;d < ndisp;++d) {
					int addr_buf = (j*width + i + 1)*ndisp + d;

					Mes_L_Buf[addr_buf] = Mes_Result_Pixel[d];
				}
			}
		}
		//=======================From Right to Left===================
		for (int j = 0;j < height;++j) {
			for (int i = width - 1;i >= 1;--i) {
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

				int addr_r = AddrGet(i + 1, j, Img_L_Gary->widthStep, Img_L_Gary->nChannels);
				int addr_u = AddrGet(i, j - 1, Img_L_Gary->widthStep, Img_L_Gary->nChannels);
				int addr_d = AddrGet(i, j + 1, Img_L_Gary->widthStep, Img_L_Gary->nChannels);
				int addr_p = AddrGet(i, j, Img_L_Gary->widthStep, Img_L_Gary->nChannels);
				unsigned char color_p;
				unsigned char color_q;
				if (i == width - 1) {
					weight_A = 0;
				}
				else {
					color_p = (unsigned char)Img_L_Gary->imageData[addr_p];
					color_q = (unsigned char)Img_L_Gary->imageData[addr_r];
					weight_A = WeiGet(color_p, color_q, 5);
				}
				if (j == 0) {
					weight_B = 0;
				}
				else {
					color_p = (unsigned char)Img_L_Gary->imageData[addr_p];
					color_q = (unsigned char)Img_L_Gary->imageData[addr_u];
					weight_B = WeiGet(color_p, color_q, 5);
				}
				if (j == height - 1) {
					weight_C = 0;
				}
				else {
					color_p = (unsigned char)Img_L_Gary->imageData[addr_p];
					color_q = (unsigned char)Img_L_Gary->imageData[addr_d];
					weight_C = WeiGet(color_p, color_q, 5);
				}
				BP_Update(Mes_R_Pixel, Mes_U_Pixel, Mes_D_Pixel, Cost_Pixel, Mes_Result_Pixel, weight_A, weight_B, weight_C, lamda, ndisp);

				for (int d = 0;d < ndisp;++d) {
					int addr_buf = (j*width + i - 1)*ndisp + d;

					Mes_R_Buf[addr_buf] = Mes_Result_Pixel[d];
				}

			}
		}

		//=======================From Up to Down===================

		for (int i = 0;i < width;++i) {
			for (int j = 0;j < height - 1;++j) {
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

				int addr_u = AddrGet(i, j - 1, Img_L_Gary->widthStep, Img_L_Gary->nChannels);
				int addr_l = AddrGet(i - 1, j, Img_L_Gary->widthStep, Img_L_Gary->nChannels);
				int addr_r = AddrGet(i + 1, j, Img_L_Gary->widthStep, Img_L_Gary->nChannels);
				int addr_p = AddrGet(i, j, Img_L_Gary->widthStep, Img_L_Gary->nChannels);
				unsigned char color_p;
				unsigned char color_q;
				if (j == 0) {
					weight_A = 0;
				}
				else {
					color_p = (unsigned char)Img_L_Gary->imageData[addr_p];
					color_q = (unsigned char)Img_L_Gary->imageData[addr_u];
					weight_A = WeiGet(color_p, color_q, 5);
				}
				if (i == 0) {
					weight_B = 0;
				}
				else {
					color_p = (unsigned char)Img_L_Gary->imageData[addr_p];
					color_q = (unsigned char)Img_L_Gary->imageData[addr_l];
					weight_B = WeiGet(color_p, color_q, 5);
				}
				if (i == width - 1) {
					weight_C = 0;
				}
				else {
					color_p = (unsigned char)Img_L_Gary->imageData[addr_p];
					color_q = (unsigned char)Img_L_Gary->imageData[addr_r];
					weight_C = WeiGet(color_p, color_q, 5);
				}

				BP_Update(Mes_U_Pixel, Mes_L_Pixel, Mes_R_Pixel, Cost_Pixel, Mes_Result_Pixel, weight_A, weight_B, weight_C, lamda, ndisp);

				for (int d = 0;d < ndisp;++d) {
					int addr_buf = ((j + 1)*width + i)*ndisp + d;

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

				int addr_d = AddrGet(i, j + 1, Img_L_Gary->widthStep, Img_L_Gary->nChannels);
				int addr_l = AddrGet(i - 1, j, Img_L_Gary->widthStep, Img_L_Gary->nChannels);
				int addr_r = AddrGet(i + 1, j, Img_L_Gary->widthStep, Img_L_Gary->nChannels);
				int addr_p = AddrGet(i, j, Img_L_Gary->widthStep, Img_L_Gary->nChannels);
				unsigned char color_p;
				unsigned char color_q;
				if (j == height - 1) {
					weight_A = 0;
				}
				else {
					color_p = (unsigned char)Img_L_Gary->imageData[addr_p];
					color_q = (unsigned char)Img_L_Gary->imageData[addr_d];
					weight_A = WeiGet(color_p, color_q, 5);
				}
				if (i == 0) {
					weight_B = 0;
				}
				else {
					color_p = (unsigned char)Img_L_Gary->imageData[addr_p];
					color_q = (unsigned char)Img_L_Gary->imageData[addr_l];
					weight_B = WeiGet(color_p, color_q, 5);
				}
				if (i == width - 1) {
					weight_C = 0;
				}
				else {
					color_p = (unsigned char)Img_L_Gary->imageData[addr_p];
					color_q = (unsigned char)Img_L_Gary->imageData[addr_r];
					weight_C = WeiGet(color_p, color_q, 5);
				}

				BP_Update(Mes_D_Pixel, Mes_L_Pixel, Mes_R_Pixel, Cost_Pixel, Mes_Result_Pixel, weight_A, weight_B, weight_C, lamda, ndisp);

				for (int d = 0;d < ndisp;++d) {
					int addr_buf = ((j - 1)*width + i)*ndisp + d;

					Mes_D_Buf[addr_buf] = Mes_Result_Pixel[d];
				}

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

			int addr_u = AddrGet(i, j - 1, Img_L_Gary->widthStep, Img_L_Gary->nChannels);
			int addr_d = AddrGet(i , j+1, Img_L_Gary->widthStep, Img_L_Gary->nChannels);
			int addr_l = AddrGet(i - 1, j, Img_L_Gary->widthStep, Img_L_Gary->nChannels);
			int addr_r = AddrGet(i + 1, j, Img_L_Gary->widthStep, Img_L_Gary->nChannels);
			int addr_p = AddrGet(i, j, Img_L_Gary->widthStep, Img_L_Gary->nChannels);
			unsigned char color_p;
			unsigned char color_q;
			if (j == 0) {
				weight_C = 0;
			}
			else {
				color_p = Img_L_Gary->imageData[addr_p];
				color_q = Img_L_Gary->imageData[addr_u];
				weight_C = WeiGet(color_p, color_q, 5);
			}
			if (j == height-1) {
				weight_D = 0;
			}
			else {
				color_p = Img_L_Gary->imageData[addr_p];
				color_q = Img_L_Gary->imageData[addr_d];
				weight_D = WeiGet(color_p, color_q, 5);
			}
			if (i == 0) {
				weight_A = 0;
			}
			else {
				color_p = Img_L_Gary->imageData[addr_p];
				color_q = Img_L_Gary->imageData[addr_l];
				weight_A = WeiGet(color_p, color_q, 5);
			}
			if (i == width - 1) {
				weight_B = 0;
			}
			else {
				color_p = Img_L_Gary->imageData[addr_p];
				color_q = Img_L_Gary->imageData[addr_r];
				weight_B = WeiGet(color_p, color_q, 5);
			}

			disp_Img->imageData[addr_disp_Img] =disp_scale*BP_Disp_Deter(Mes_L_Pixel, Mes_R_Pixel, Mes_U_Pixel, Mes_D_Pixel, Cost_Pixel, weight_A, weight_B,weight_C, weight_D, ndisp);
			disp_Img->imageData[addr_disp_Img+1] = disp_Img->imageData[addr_disp_Img];
			disp_Img->imageData[addr_disp_Img+2] = disp_Img->imageData[addr_disp_Img];
		}
	}
	delete[]Cost_Buf;
	delete[]Mes_L_Buf;
	delete[]Mes_R_Buf;
	delete[]Mes_U_Buf;
	delete[]Mes_D_Buf;
	#endif
}

#endif