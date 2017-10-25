#include "Post_Proc.h"


void WMode_F(IplImage *_Disp_Img_Pre, IplImage *_Img, IplImage *_Disp_Img_Filter,int radius,int ndisp) {

	int width = _Disp_Img_Pre->width;
	int height = _Disp_Img_Pre->height;

	int WidthStep_disp = _Disp_Img_Pre->widthStep;
	int WidthStep_Img = _Img->widthStep;

	int nChan_disp = _Disp_Img_Pre->nChannels;
	int nChan_Img = _Img->nChannels;

	float *Buffer = new float[ndisp];



	for (int j = 0;j < height;++j) {
		for (int i = 0;i < width;++i) {

			for (int k = 0;k < ndisp;++k) {
				Buffer[k] = 0;
			}
			int addr_Img = AddrGet(i, j, WidthStep_Img, nChan_Img);
			int Data_p = (unsigned char)_Img->imageData[addr_Img];

			for (int w_y = j - radius;w_y <= j + radius;++w_y) {
				for (int w_x = i - radius;w_x <= i + radius;++w_x) {
					if (w_x < 0 || w_y < 0 || w_x >= width || w_y >= height)
						continue;
					int addr_disp = AddrGet(w_x, w_y, WidthStep_disp, nChan_disp);
					addr_Img = AddrGet(w_x, w_y, WidthStep_Img, nChan_Img);
					int Data_disp = (unsigned char)_Disp_Img_Pre->imageData[addr_disp];
					int Data_q = (unsigned char)_Img->imageData[addr_Img];
					Buffer[Data_disp] += WeiGet(Data_p, Data_q, 5.0);

				}
			}

			int addr_disp_p = AddrGet(i, j, WidthStep_disp, nChan_disp);
			float max_cost;
			float max_d;
			for (int k = 0;k < ndisp;++k) {
				if (k == 0 || Buffer[k] > max_cost) {
					max_d = k;
					max_cost = Buffer[k];
				}
			}
			_Disp_Img_Filter->imageData[addr_disp_p] = (unsigned char)max_d;

		}
	}


}