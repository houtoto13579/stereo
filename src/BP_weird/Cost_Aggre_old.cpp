#include "Cost_Aggre.h"
float alpha = 0.25;

float Box_Aggre(IplImage * _Img_L, IplImage * _Img_R, int x, int y, int radius, int d) {

	int width = _Img_L->width;
	int height = _Img_L->height;
	int num = 0;
	float cost = 0;
	for (int j = y - radius;j <= y + radius;++j) {
		for (int i = x - radius;i <= x + radius;++i) {
			if (j < 0 || j >= height || i < 0 || i >= width)
				continue;
			num++;
			cost+= AD_Cost(_Img_L, _Img_R, i, j, d);

		}
	}

	cost = cost / (float)(num+0.01);
	return cost;

}

float ASW_Aggre(IplImage * _Img_L, IplImage * _Img_R, int x, int y, int radius, int d) {

	int width = _Img_L->width;
	int height = _Img_L->height;
	int WidthStep = _Img_L->widthStep;
	int nChan     = _Img_L->nChannels;
	float num = 0;
	float cost = 0;
	for (int j = y - radius;j <= y + radius;++j) {
		for (int i = x - radius;i <= x + radius;++i) {
			if (j < 0 || j >= height || i < 0 || i >= width || i-d < 0 || i-d >= width)
				continue;
			int addr_L = AddrGet(i, j, WidthStep, nChan);
			int addr_R = AddrGet(i-d, j, WidthStep, nChan);
			int Data_L = (unsigned char)_Img_L->imageData[addr_L];
			int Data_R = (unsigned char)_Img_L->imageData[addr_R];
			float wieght = WeiGet(Data_L, Data_R, 10.0f);
			num+= wieght;
			cost += (alpha*AD_Cost(_Img_L, _Img_R, i, j, d)+ (1-alpha)*Census_Cost(_Img_L, _Img_R, i, j, d,3))*wieght;

		}
	}

	cost = cost / (float)(num + 0.01);
	return cost;

}