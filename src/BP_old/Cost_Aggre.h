#pragma once
#ifndef __Cost_Aggre__H___
#define __Cost_Aggre__H___

#include "main.h"
#include "BasicFun.h"

float Box_Aggre(IplImage * _Img_L, IplImage * _Img_R, int x,int y,int radius,int d);
float ASW_Aggre(IplImage * _Img_L, IplImage * _Img_R, int x, int y, int radius, int d);

#endif