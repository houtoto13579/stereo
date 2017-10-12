#ifndef __FILTER_H__
#define __FILTER_H__

#include <iostream>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <vector>
#include <emmintrin.h>
#include <math.h>


class Filter {
    public:
        inline int Get_addr(int width,int x, int y){return y*width+x;}
	void Weight_MOD(uint8_t* Feature,float* Disp,float* Disp_F,int win_radius,const int32_t* dims,int ndisp);
	inline float Get_Weight(uint8_t feature_p,uint8_t feature_q, int p_x,int p_y,int q_x,int q_y);
};

#endif
