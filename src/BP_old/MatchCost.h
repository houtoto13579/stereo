#pragma once
#ifndef  ___MatchCost__H__
#define ___MatchCost__H__

#include "main.h"
#include "BasicFun.h"

int AD_Cost(IplImage * _Img_L, IplImage * _Img_R, int x, int y, int d);
int Census_Cost(IplImage * _Img_L, IplImage * _Img_R, int x, int y, int d, int radius);

#endif
