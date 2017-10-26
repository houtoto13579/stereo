#include "main.h"

using namespace std;
using namespace cv;

IplImage * Img_Disp_Pre[5];
IplImage * Img_L_Pre[5];
IplImage * Img_R_Pre[5];
int       f;

void Occlusion_Mask(IplImage * _Img_GT, IplImage * _Img_GT_R, IplImage * _Img_GT_Mask,int ndisp) {

	int height = _Img_GT->height;
	int width = _Img_GT->width;
	int widthStep = _Img_GT->widthStep;
	int nchan = _Img_GT->nChannels;

	for (int j = 0;j < height;++j) {
		for (int i = width - 1;i >= 0;--i) {
			//cout << "i=" << i << "j=" << j << endl;
			int addr_l = AddrGet(i, j, widthStep, nchan);
			int disp_l = (unsigned char)_Img_GT->imageData[addr_l];
			disp_l = disp_l *  ndisp/256;
			if (i < disp_l) {
				for (int c = 0;c < nchan;++c) {
					_Img_GT_Mask->imageData[addr_l+c] = (unsigned char)0;
				}
				continue;
			}
			int addr_r = AddrGet(i- disp_l, j, widthStep, nchan);
			int disp_r = (unsigned char)_Img_GT_R->imageData[addr_r];
			disp_r = disp_r * ndisp/256;

			if (disp_l != disp_r) {
				for (int c = 0;c < nchan;++c) {
					_Img_GT_Mask->imageData[addr_l + c] = (unsigned char)0;
				}
			}

		}
	}

}

float Evaluate(IplImage * _Img_GT, IplImage *_disp_Img, IplImage *_disp_Error_Img, int ndisp, float thres) {
	int height = _Img_GT->height;
	int width = _Img_GT->width;
	int nchan_GT = _Img_GT->nChannels;
	int widthstep_GT = _Img_GT->widthStep;

	int nchan_disp = _disp_Img->nChannels;
	int widthstep_disp = _disp_Img->widthStep;

	int total_num = width*height;
	int error_num = 0;
	float error_rate;
	float scale = 256 / ndisp;
	for (int j = 0;j < height;++j) {
		for (int i = 0;i < width;++i) {
			int addr_Error = AddrGet(i, j, _disp_Error_Img->widthStep, _disp_Error_Img->nChannels);
			int addr_GT = AddrGet(i, j, widthstep_GT, nchan_GT);
			int addr_disp = AddrGet(i, j, widthstep_disp, nchan_disp);
			unsigned char data_GT = (unsigned char)_Img_GT->imageData[addr_GT];
			if (data_GT == 0) {
				for (int c = 0;c < _disp_Error_Img->nChannels;++c) {
					_disp_Error_Img->imageData[addr_Error + c] = (unsigned char)0;
				}
				total_num--;
				continue;
			}
			unsigned char data_disp = (unsigned char)_disp_Img->imageData[addr_disp];
			if (abs(data_GT - data_disp) > scale*thres) {
				for (int c = 0;c < _disp_Error_Img->nChannels;++c) {
					if (c != _disp_Error_Img->nChannels - 1)
						_disp_Error_Img->imageData[addr_Error + c] = (unsigned char)0;
					else
						_disp_Error_Img->imageData[addr_Error + c] = (unsigned char)255;
				}
				error_num++;

			}
			else {
				for (int c = 0;c < _disp_Error_Img->nChannels;++c) {
					_disp_Error_Img->imageData[addr_Error + c] = (unsigned char)data_disp;
				}
			}
		}
	}
	error_rate = error_num / (float)(total_num);
	printf("Error rate is %f\n", error_rate);
	return error_rate;
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
	//BP(_Img_L, _Img_R, _disp_Img, ndisp);
	//ICCV2010(_Img_L, _Img_R, _disp_Img, ndisp);
	//WMode_F(_disp_Img, Img_L_Gary, _disp_Img_F, 3, 256);
	//cvCopy(_disp_Img_F, _disp_Img, NULL);

	//printf("End Cost\n");
	//cvShowImage("Disp_Image", _disp_Img);
	//cvWaitKey(0);

	//cvDestroyWindow("Disp_Image");
	cvReleaseImage(&Img_L_Gary);
	cvReleaseImage(&Img_R_Gary);
	cvReleaseImage(&_disp_Img_F);

}
/*
int main(int argc, char **argv)
{



	printf("SpatioTemporal Stereo\n");

	char ImgName_L[50];
	IplImage * Img_L = NULL;
	char ImgName_R[50];
	IplImage * Img_R = NULL;
	IplImage * Img_GT = NULL;
	IplImage * Img_GT_R = NULL;
	IplImage * Img_GT_Mask = NULL;
	char ImgName_Disp[50];
	char ImgName_Error_Disp[50];
	char ImgName_GT[50];
	char ImgName_GT_R[50];
	char ImgName_GT_Mask[50];

	char *dataset = "temple";
	int total_frame = 40;
	char filename[30];
	fstream fp;
	sprintf(filename, "%s.txt", dataset);
	fp.open(filename, ios::out);//�}���ɮ�
	if (!fp) {//�p�G�}���ɮץ��ѡAfp��0�F���\�Afp���D0
		cout << "Fail to open file: " << filename << endl;
	}

	float total_error = 0;

	for (f = 1;f < 1+ total_frame;++f){
		printf("Frame #%d\n", f);
		sprintf(ImgName_L, "%s/L%04d.png", dataset, f);
		sprintf(ImgName_R, "%s/R%04d.png", dataset, f);
		sprintf(ImgName_Disp, "D%04d.png", f);
		sprintf(ImgName_GT, "%s/TL%04d.png", dataset, f);
		sprintf(ImgName_GT_R, "%s/TR%04d.png", dataset, f);
		sprintf(ImgName_GT_Mask, "TL_MASK%04d.png", f);

		if (cvLoadImage(ImgName_L, CV_LOAD_IMAGE_COLOR) == NULL)
			printf("Error Loading Inputs\n");
		Img_L = cvLoadImage(ImgName_L, CV_LOAD_IMAGE_COLOR);
		Img_R = cvLoadImage(ImgName_R, CV_LOAD_IMAGE_COLOR);
		Img_GT = cvLoadImage(ImgName_GT, CV_LOAD_IMAGE_COLOR);
		Img_GT_R = cvLoadImage(ImgName_GT_R, CV_LOAD_IMAGE_COLOR);
		Img_GT_Mask = cvLoadImage(ImgName_GT, CV_LOAD_IMAGE_COLOR);
		IplImage * disp_Img = cvCreateImage(cvGetSize(Img_L), IPL_DEPTH_8U, 1);
		IplImage * disp_Error_Img = cvCreateImage(cvGetSize(Img_L), IPL_DEPTH_8U, 3);

		if (f == 1) {
			for (int n = 0;n < 5;++n) {
				Img_Disp_Pre[n] = cvCreateImage(cvGetSize(disp_Img), IPL_DEPTH_8U, 1);
				Img_L_Pre[n] = cvCreateImage(cvGetSize(Img_L), IPL_DEPTH_8U, 3);
				Img_R_Pre[n] = cvCreateImage(cvGetSize(Img_R), IPL_DEPTH_8U, 3);
			}
		}



		//cvShowImage("Image", Img_L);
		//cvWaitKey(0);
		//cvDestroyWindow("Image");

		SpatiotemporalStereo(Img_L, Img_R, disp_Img, 64);
		cvSaveImage(ImgName_Disp, disp_Img);
		Occlusion_Mask(Img_GT, Img_GT_R,Img_GT_Mask,64);
		cvSaveImage(ImgName_GT_Mask, Img_GT_Mask);

		float error_rate = Evaluate(Img_GT_Mask, disp_Img, disp_Error_Img, 64, 1.0);
		total_error += error_rate;
		fp << "Frame " << f << "Error rate:" << error_rate << endl;
		sprintf(ImgName_Error_Disp, "DL_Error%04d.png", f);
		cvSaveImage(ImgName_Error_Disp, disp_Error_Img);

		//Img_Disp_Pre = NULL;
		//Img_L_Pre = NULL;
		for (int n = 4;n >= 1;--n) {
			if (f > n) {
				cvCopy(Img_Disp_Pre[n], Img_Disp_Pre[(n - 1)]);
				cvCopy(Img_L_Pre[n], Img_L_Pre[n - 1]);
				cvCopy(Img_R_Pre[n], Img_R_Pre[n - 1]);
			}
			else
				continue;
		}
		cvCopy(disp_Img, Img_Disp_Pre[0]);
		cvCopy(Img_L, Img_L_Pre[0]);
		cvCopy(Img_R, Img_R_Pre[0]);

		cvReleaseImage(&disp_Img);
		cvReleaseImage(&Img_L);
		cvReleaseImage(&Img_R);
	}
	total_error = total_error / (float)total_frame;
	cout << "Total Error is" << total_error << endl;
	fp << "Total Error is " << total_error << endl;

	fp.close();//�����ɮ�
	
	scanf(" ");
	return 0;
}

*/