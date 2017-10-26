#include "Optimization.h"

//#define Tile_BP
#ifdef Tile_BP
 #define Tile_size 32
#endif

int lamda = 2;
int Iteration = 2;

extern IplImage * Img_Disp_Pre[5];
extern IplImage * Img_L_Pre[5];
extern IplImage * Img_R_Pre[5];
extern int f;

void BP(IplImage* Left_Img, IplImage* Right_Img, IplImage* disp_Img,int ndisp, int bsize) {
	int f = 1;
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
				if (f == 1) {
					if (i < d)
						Cost_Buf[Buf_addr] = 999;
					else
						Cost_Buf[Buf_addr] = ASW_Aggre(Img_L_Gary, Img_R_Gary, i, j, bsize, d);
				}
				else{
					if (i < d)
						Cost_Buf[Buf_addr] = 999;
					else

						Cost_Buf[Buf_addr] = ASW_Aggre(Img_L_Gary, Img_R_Gary, i, j, bsize, d)+ Temporal_Cost(i, j, d, Left_Img);
				}
			}
		}
	}
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
	#endif

	delete[]Cost_Buf;
	delete[]Mes_L_Buf;
	delete[]Mes_R_Buf;
	delete[]Mes_U_Buf;
	delete[]Mes_D_Buf;

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


inline float Temporal_Cost(int index_x, int index_y, int d, IplImage* _Img) {

	float weight;
	float temporal_cost;

	int addr = AddrGet(index_x,index_y, Img_Disp_Pre[0]->widthStep, Img_Disp_Pre[0]->nChannels);
	weight = Temproal_Sim(index_x, index_y, 3, _Img);
	int Pre_d = (unsigned char)Img_Disp_Pre[0]->imageData[addr];
	Pre_d = Pre_d >> 2;
	temporal_cost = weight * abs(d - Pre_d);

	return temporal_cost;

}
inline float Temproal_Sim(int index_x, int index_y, int radius, IplImage* _Img) {

	float similarity;
	float Diff_sum=0;
	float omega_sum=0;
	int addr_p = AddrGet(index_x, index_y, _Img->widthStep, _Img->nChannels);
	for (int Y = index_y - radius;Y <= index_y + radius;++Y) {
		for (int X = index_x - radius;X <= index_x + radius;++X) {
			if (X < 0 || X >= _Img->width || Y < 0 || Y >= _Img->height)
				continue;
			int addr_q = AddrGet(X, Y, _Img->widthStep, _Img->nChannels);

			float Diff_pixel = 0;
			for (int c = 0;c < _Img->nChannels;++c){
				Diff_pixel += abs((unsigned char)_Img->imageData[addr_q + c] - (unsigned char)Img_L_Pre[0]->imageData[addr_q + c]);
			}
			Diff_pixel = Diff_pixel / (float)_Img->nChannels;

			float Diff_pq = 0;
			for (int c = 0;c < _Img->nChannels;++c) {
				Diff_pq += abs((unsigned char)_Img->imageData[addr_q + c] - (unsigned char)_Img->imageData[addr_p + c]);
			}
			Diff_pq = Diff_pq/ (float)_Img->nChannels;
			float omega = exp(-Diff_pq / (float)5.0f);
			Diff_sum += Diff_pixel*omega;
			omega_sum += omega;
		}
	}

	similarity = Diff_sum / omega_sum;

	similarity = exp(-similarity / 5.0f);

	return similarity;
}


void ICCV2010(IplImage* Left_Img, IplImage* Right_Img, IplImage* disp_Img, int ndisp) {
	int width = Left_Img->width;
	int height = Left_Img->height;
	int widthstep = Left_Img->widthStep;

	int widthstep_Disp = disp_Img->widthStep;
	int nchan_Disp = disp_Img->nChannels;
	int Buf_size = width*height*ndisp;
	float *Cost_Buf;
	Cost_Buf = new float[Buf_size];

	int disp_scale = 256 / ndisp;
	for (int j = 0;j < height;++j) {
		for (int i = 0;i < width;++i) {
			for (int d = 0;d < ndisp;++d) {
				int Buf_addr = (j*width + i)*ndisp + d;
				if (i < d)
					Cost_Buf[Buf_addr] = 999;
				else {
					Cost_Buf[Buf_addr] = ASW_Aggre(Left_Img, Right_Img, i, j, 17, d);
					
					for (int k = 1;k <= 4;++k) {
						if (f - k > 0) {
							float weight = exp(-(k*k) >> 3);
							Cost_Buf[Buf_addr] += weight*ASW_Aggre(Img_L_Pre[k], Img_R_Pre[k], i, j, 17, d);
						}
						else
							continue;
					}
					
				}

			}
		}
	}


	for (int j = 0;j < height;++j) {
		for (int i = 0;i < width;++i) {
			int addr_disp_Img = AddrGet(i, j, widthstep_Disp,nchan_Disp);
			int min_d = 0;
			float min_cost = 999;
			float temp_cost;

			for (int d = 0;d < ndisp;++d) {
				int Buf_addr = (j*width + i)*ndisp + d;
				temp_cost = Cost_Buf[Buf_addr];
				if (d == 0 || temp_cost < min_cost) {
					min_d = d;
					min_cost = temp_cost;
				}
			}

			disp_Img->imageData[addr_disp_Img] = disp_scale*min_d;

		}
	}
}