#include "Filter.h"

void Filter::Weight_MOD(uint8_t* Feature,float* Disp,float* Disp_F,int win_radius,const int32_t* dims,int ndisp){

  int width  = dims[0];
  int height = dims[1];
  int bpl    = width + 15-(width-1)%16;


  float Hist[ndisp];
  memset( Hist, 0, ndisp*sizeof(float));
  //============================Begin for Pixel For loop
  for(int x=0;x<width;x++){
	for(int y=0;y<height;y++){

	    //==================Begin for Initialation
	    for(int d=0;d<ndisp;d++){
		Hist[d] = 0;
	    }

            //==================End for Initialation
 
  	    int addr_feature_p  = Get_addr(bpl,x,y) ;
      	    int addr_disp_p     = Get_addr(width,x,y);
	    int disp_p          = (int)Disp[addr_disp_p];
	    uint8_t feature_p   = Feature[addr_feature_p];
	    //==================Begin for Window
	    for(int w_x=x-win_radius;w_x<=x+win_radius;w_x++){
		for(int w_y=y-win_radius;w_y<=y+win_radius;w_y++){
	            //===================Boundary Check
		    if(w_x<0||w_x>=width||w_y<0||w_y>=height)
			continue;
  	    	    int addr_feature_q  = Get_addr(bpl,w_x,w_y) ;
      	    	    int addr_disp_q     = Get_addr(width,w_x,w_y);
	    	    int disp_q          = (int)Disp[addr_disp_q];
	    	    uint8_t feature_q   = Feature[addr_feature_q];
		    float weight;
		    weight = Get_Weight(feature_p,feature_q,x,y,w_x,w_y);
		    Hist[disp_q] +=weight;
		}
	    }
	    //==================End for Window
	    float max_hist;
	    int   max_disp;
	    //==================Begin for Max Selection
	    for(int d=0;d<ndisp;d++){
		if(d==0||Hist[d]>max_hist){
		    max_disp = d;
		    max_hist = Hist[d];
		}
	    }
	    Disp_F[addr_disp_p]=max_disp;
            //==================End for Max Selection 


	}
  }
  //============================End for Pixel For loop

}
inline float Filter::Get_Weight(uint8_t _feature_p,uint8_t _feature_q, int p_x,int p_y,int q_x,int q_y){

    int theta_loc=5;
    int theta_app=5;

    float loc_weight = exp(-(abs(p_x-q_x)+abs(p_y-q_y))/theta_loc);
    float app_weight = exp(-(abs(_feature_p-_feature_q))/theta_app);

    return loc_weight*app_weight;
}
