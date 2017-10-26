#include "MatchCost.h"

int AD_Cost(IplImage * _Img_L, IplImage * _Img_R,int x,int y,int d) {

	int Addr_Img_L;
	int Addr_Img_R;
	int nChans = _Img_L->nChannels;
	int WidthStep = _Img_L->widthStep;

	int Cost=0;

	//IplImage * Img_L_ = cvCreateImage(cvGetSize(_Img_L), _Img_L->depth, _Img_L->nChannels);
	//IplImage * Img_R_ = cvCreateImage(cvGetSize(_Img_R), _Img_R->depth, _Img_L->nChannels);

	//cvCopy(_Img_L, Img_L_, NULL);
	//cvCopy(_Img_R, Img_R_, NULL);

	int x_ = x;
	int y_ = y;
	int d_ = d;

	int x_r_ = x_ - d;
	if (x_r_ < 0)
		return 255;

	Addr_Img_L = AddrGet(x_,y_, WidthStep, nChans);
	Addr_Img_R = AddrGet(x_r_, y_, WidthStep, nChans);

	for (int C = 0;C < nChans;C++) {
		Cost = +abs(_Img_L->imageData[Addr_Img_L]- _Img_R->imageData[Addr_Img_R]);
		Addr_Img_L++;
		Addr_Img_R++;
	}
	Cost = Cost / nChans;

	return Cost;
	//cvReleaseImage(&Img_L_);
	//cvReleaseImage(&Img_R_);
}

int Census_Cost(IplImage * _Img_L, IplImage * _Img_R, int x, int y, int d,int radius) {
	int width = _Img_L->width;
	int height = _Img_L->height;
	int WidthStep = _Img_L->widthStep;
	int nChan = _Img_L->nChannels;

	int cost = 0;

	int ImgAddr_p_l=AddrGet(x,y,WidthStep, nChan);
	int ImgAddr_p_r = AddrGet(x-d, y, WidthStep, nChan);

	int Data_p_l = (unsigned char)_Img_L->imageData[ImgAddr_p_l];
	int Data_p_r = (unsigned char)_Img_R->imageData[ImgAddr_p_r];

	for (int j = y - radius;y >= y + radius;++j) {
		for (int i = x - radius;i <= x + radius;++i) {
			if (j < 0 || j >= height || i < 0 || i >= width || i-d < 0 || i-d >= width)
				continue;
			int ImgAddr_q_l = AddrGet(i, j, WidthStep, nChan);
			int ImgAddr_q_r = AddrGet(i - d, j, WidthStep, nChan);

			int Data_q_l = (unsigned char)_Img_L->imageData[ImgAddr_q_l];
			int Data_q_r = (unsigned char)_Img_R->imageData[ImgAddr_q_r];

			cost += (Data_q_l > Data_p_l) ^ (Data_q_r > Data_p_r);

		}
	}
	
	return cost;


}