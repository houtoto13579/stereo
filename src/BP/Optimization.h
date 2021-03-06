#pragma once
#ifndef __Optimization_H__
#define __Optimization_H__

#include "main.h"

void BP(IplImage* Left_Img, IplImage* Right_Img, IplImage* disp_Img, int ndisp, int bsize=13);
void BP_Update(float *Mes_A, float *Mes_B, float *Mes_C, float *Cost, float *Mes_Out, float weight_A, float weight_B, float weight_C, float lamda, int ndisp);
int BP_Disp_Deter(float *Mes_A, float *Mes_B, float *Mes_C, float *Mes_D, float *Cost, float weight_A, float weight_B, float weight_C, float weight_D, int ndisp);
inline float Temproal_Sim(int index_x, int index_y, int radius, IplImage* _Img);
inline float Temporal_Cost(int index_x, int index_y, int d, IplImage* _Img);

void ICCV2010(IplImage* Left_Img, IplImage* Right_Img, IplImage* disp_Img, int ndisp);

#endif