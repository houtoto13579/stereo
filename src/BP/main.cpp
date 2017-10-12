#include "main.h"

using namespace std;
using namespace cv;

void Evaluate(IplImage * _Img_GT, IplImage *_disp_Img, int ndisp, float thres) {
	int height = _Img_GT->height;
	int width = _Img_GT->width;
	int nchan_GT = _Img_GT->nChannels;
	int widthstep_GT = _Img_GT->widthStep;

	int nchan_disp = _disp_Img->nChannels;
	int widthstep_disp = _disp_Img->widthStep;

	int error_num = 0;
	float error_rate;
	float scale = 256 / ndisp;
	for (int j = 0;j < height;++j) {
		for (int i = 0;i < width;++i) {
			int addr_GT = AddrGet(i, j, widthstep_GT, nchan_GT);
			int addr_disp = AddrGet(i, j, widthstep_disp, nchan_disp);
			unsigned char data_GT = (unsigned char)_Img_GT->imageData[addr_GT];
			unsigned char data_disp = (unsigned char)_disp_Img->imageData[addr_disp];
			if (abs(data_GT - data_disp) > scale*thres)
				error_num++;
		}
	}
	error_rate = error_num / (float)(width*height);
	printf("Error rate is %f\n", error_rate);
}

void SpatiotemporalStereo(IplImage * _Img_L, IplImage * _Img_R,IplImage *_disp_Img,int ndisp) {

	int height = _Img_L->height;
	int width = _Img_L->width;

	float disp_scale = 256 /(float)ndisp;

	IplImage * Img_L_Gary = cvCreateImage(cvGetSize(_Img_L), _Img_L->depth, 1);
	IplImage * Img_R_Gary = cvCreateImage(cvGetSize(_Img_R), _Img_R->depth, 1);

	IplImage * _disp_Img_F = cvCreateImage(cvGetSize(_disp_Img), _Img_L->depth, 1);

	cvCvtColor(_Img_L, Img_L_Gary, CV_RGB2GRAY);
	cvCvtColor(_Img_R, Img_R_Gary, CV_RGB2GRAY);

	//cvCopy(_Img_L, Img_L_, NULL);
	//cvCopy(_Img_R, Img_R_, NULL);
	/*
	for (int j = 0;j < height;++j) {
		//printf("j = %d\n", j);
		for (int i = 0;i < width;++i) {
			int min_d=0;
			float min_cost=999;
			float temp_cost;
			for (int d = 0;d < ndisp;++d) {
				if (i < d) {
					temp_cost = 999;
					continue;
				}

				//temp_cost = AD_Cost(_Img_L, _Img_R, i, j, d);
				//temp_cost =  Box_Aggre(_Img_L, _Img_R,i,j, 3,  d);
				temp_cost = ASW_Aggre(Img_L_Gary, Img_R_Gary, i, j, 3, d);
				if (temp_cost<min_cost) {
					min_cost = temp_cost;
					min_d = d;
				}
			}

			int addr_disp = AddrGet(i, j, _disp_Img->widthStep, 1);
			_disp_Img->imageData[addr_disp] = (unsigned char)min_d*disp_scale;
		}
	}
	*/
	BP(_Img_L, _Img_R, _disp_Img, ndisp);
	WMode_F(_disp_Img, Img_L_Gary, _disp_Img_F, 3, 256);
	cvCopy(_disp_Img_F, _disp_Img, NULL);

	//printf("End Cost\n");
	//cvShowImage("Disp_Image", _disp_Img);
	//cvWaitKey(0);

	//cvDestroyWindow("Disp_Image");


}

// int main(int argc, char **argv)
// {

// 	printf("SpatioTemporal Stereo\n");

// 	char ImgName_L[50];
// 	IplImage * Img_L = NULL;
// 	char ImgName_R[50];
// 	IplImage * Img_R = NULL;
// 	IplImage * Img_GT = NULL;
// 	char ImgName_Disp[50];
// 	char ImgName_GT[50];

// 	for (int f = 1;f <= 41;++f){
// 		printf("Frame #%d\n", f);
// 		sprintf(ImgName_L, "book/L%04d.png", f);
// 		sprintf(ImgName_R, "book/R%04d.png", f);
// 		sprintf(ImgName_Disp, "D%04d.png", f);

// 		if (cvLoadImage(ImgName_L, CV_LOAD_IMAGE_COLOR) == NULL)
// 			printf("Error Loading Inputs\n");
// 		Img_L = cvLoadImage(ImgName_L, CV_LOAD_IMAGE_COLOR);
// 		Img_R = cvLoadImage(ImgName_R, CV_LOAD_IMAGE_COLOR);

// 		IplImage * disp_Img = cvCreateImage(cvGetSize(Img_L), IPL_DEPTH_8U, 1);


// 		//cvShowImage("Image", Img_L);
// 		//cvWaitKey(0);
// 		//cvDestroyWindow("Image");

// 		SpatiotemporalStereo(Img_L, Img_R, disp_Img, 64);
// 		cvSaveImage(ImgName_Disp, disp_Img);

// 		sprintf(ImgName_GT, "book/TL%04d.png", f);
// 		Img_GT = cvLoadImage(ImgName_GT, CV_LOAD_IMAGE_COLOR);
		
// 		Evaluate(Img_GT, disp_Img, 64, 1.0);

// 		cvReleaseImage(&disp_Img);
// 		cvReleaseImage(&Img_L);
// 		cvReleaseImage(&Img_R);
// 	}
	
// 	scanf(" ");
// 	return 0;
// }

