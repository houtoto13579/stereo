#pragma once
#ifndef __BASICFUN__H__
#define __BASICFUN__H__

#include "main.h"

inline int AddrGet(int x, int y, int WidthStep, int nChan){return (y*WidthStep + x*nChan);}

inline float WeiGet(int x, int y, float gamma) { return exp(-abs(x - y) / (float)gamma); }

inline int min_Ben(int a, int b) { return (a < b) ? a : b; }

#endif