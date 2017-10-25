#include "Optimization.h"
using namespace std;

//#define Tile_BP
#ifdef Tile_BP
 #define Tile_size 32
#endif

void BP(IplImage* Left_Img, IplImage* Right_Img, IplImage* disp_Img,int ndisp) {

	int width = Left_Img->width;
	int height = Left_Img->height;
	int widthstep = Left_Img->widthStep;

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
	for (int j = 0;j < height;++j) {
		for (int i = 0;i < width;++i) {
			for (int d = 0;d < ndisp;++d) {
				int Buf_addr = (j*width + i)*ndisp + d;
				Cost_Buf[Buf_addr] = ASW_Aggre(Img_L_Gary, Img_R_Gary, i, j, 3, d);
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

#endif


}

void BP_Update(float *Mes_A, float *Mes_B, float *Mes_C, float *Cost, float *Mes_Out, float weight_A, float weight_B, float weight_C, float lamda,int ndisp) {

	int min_d_q, min_d_p;
	float energy_min_q, energy_min_p, energy_temp;
	int T = ndisp >> 3;
	for (int d_p = 0;d_p < ndisp;++d_p) {
		for (int d_q = 0;d_q < ndisp;++d_q) {
			int diff_pq = abs(d_p - d_q);
			if (diff_pq > T)
				diff_pq = T;
			energy_temp = Mes_A[d_q] * weight_A + Mes_B[d_q] * weight_B + Mes_C[d_q] * weight_C + Cost[d_q] + diff_pq*lamda;
			if (d_q == 0 || energy_min_q > energy_temp) {
				min_d_q = d_q;
				energy_min_q = energy_temp;
			}
		}
		Mes_Out[d_p] = energy_min_q;

		if (d_p == 0 || energy_min_q < energy_min_p) {
			energy_min_p = energy_min_q;
		}
	}
	for (int d_p = 0;d_p < ndisp;++d_p) {
		Mes_Out[d_p] = Mes_Out[d_p] - energy_min_p;
	}

}

int BP_Disp_Deter(float *Mes_A, float *Mes_B, float *Mes_C, float *Mes_D, float *Cost, float weight_A, float weight_B, float weight_C, float weight_D, int ndisp) {
	int min_d;
	float energy_min, energy_temp;
	for (int d = 0;d < ndisp;++d) {
		energy_temp = Mes_A[d] *weight_A + Mes_B[d] *weight_B + Mes_C[d] *weight_C + Mes_D[d] * weight_D + Cost[d];
		if (d == 0 || energy_temp < energy_min) {
			energy_min = energy_temp;
			min_d = d;
		}
	}
	return min_d;
}