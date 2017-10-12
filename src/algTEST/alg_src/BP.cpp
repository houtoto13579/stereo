/*
Copyright 2011. All rights reserved.
Institute of Measurement and Control Systems
Karlsruhe Institute of Technology, Germany

This file is part of libelas.
Authors: Andreas Geiger

libelas is free software; you can redistribute it and/or modify it under the
terms of the GNU General Public License as published by the Free Software
Foundation; either version 3 of the License, or any later version.

libelas is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
libelas; if not, write to the Free Software Foundation, Inc., 51 Franklin
Street, Fifth Floor, Boston, MA 02110-1301, USA 
*/

#include "BP.h"
#include "JointWMF.h"
#include <math.h>
#include "DFT.h"
#include "slic.h"


#include "descriptor.h"
#include "triangle.h"
#include "matrix.h"
#include "Filter.h"

int  Weight_mode = 1;
int  Post_process = 1;
unsigned char BP_mode = 1;
bool Truncated_label = false;
int  Num_k = 15;
bool LR_mode = 0;
int  Outer_Iteration = 2;
int  Inner_Iteration = 2;

// propagate method
//1: direct
//2: other
int propagation_method = 1;
int pre_confident_window = 5;
int preRobust = 50;

//Frame different function//
float FDF_BP(int dif, int comm = 1){
	 return 5*pow(0.5,dif/comm);
}

using namespace std;

void BP::process (uint8_t* I1_,uint8_t* I2_,float* D1,float* D2,const int32_t* dims, IplImage* preL, IplImage* preD){
  
  //BP in_process;

  // get width, height and bytes per line
  width  = dims[0];
  height = dims[1];
  bpl    = width + 15-(width-1)%16;

  int ndisp = param.disp_max+1;
  
  // copy images to byte aligned memory
  I1 = (uint8_t*)_mm_malloc(bpl*height*sizeof(uint8_t),16);
  I2 = (uint8_t*)_mm_malloc(bpl*height*sizeof(uint8_t),16);
	memset(I1,0,bpl*height*sizeof(uint8_t));
	memset(I2,0,bpl*height*sizeof(uint8_t));
  if (bpl==dims[2]) {
    memcpy(I1,I1_,bpl*height*sizeof(uint8_t));
    memcpy(I2,I2_,bpl*height*sizeof(uint8_t));
  } else {
    for (int32_t v=0; v<height; v++) {
      memcpy(I1+v*bpl,I1_+v*dims[2],width*sizeof(uint8_t));
      memcpy(I2+v*bpl,I2_+v*dims[2],width*sizeof(uint8_t));
    }
  }

//printf("Image's Height is %d, height is %d dim3 os %d\n",width,height,dims[2]);

    //CvSize SIZE = cvSize(width,height);
    //IplImage *_image = cvCreateImage(SIZE,IPL_DEPTH_8U,3);
    //cvShowImage("result", _image);
    //cvWaitKey(0);
    //cvCvtColor(image, lab_image, CV_BGR2Lab);

unsigned char min_dis;
int           cost;
int           min_cost;



if(BP_mode !=1){
	for(int i = 0;i<width;i++){
		//printf("pixel i %d\n",i);
		for(int j= 0;j<height;j++){
			int Left_addr_p = Get_addr(bpl,i,j);
			int Dispar_addr = Get_addr(width,i,j);
			int preWeight=0;
			int preDisp=0;
			// Calculate previous Frame preWeight and preDisp
			if(preL==NULL || preD==NULL){
				preWeight=0;
			}
			else{
				int preDif=0;
				int preCount=0;
				if(propagation_method == 1){
					for(int p=i-pre_confident_window/2;p<=i+pre_confident_window/2;p++){
						for(int q=j-pre_confident_window/2;q<=j+pre_confident_window/2;q++){
							if(p>=0 && p<width && q>=0 && q<height){
								++preCount;
								preDif+=abs((I1[Get_addr(bpl,p,q)])-(preL->imageData[q*preL->widthStep+(p*3)]));
							}
						}
					}
					preWeight = FDF_BP(preDif, preCount);
				}
				preDisp = preD->imageData[j*preD->widthStep+(i*3)];
			}
			//
			for(int d= 0;d<param.disp_max;d++){
				if(i-d<0)
					continue;
				else{
					int Right_addr_p = Get_addr(bpl,(i-d),j);
					//int cost = abs(I1[Left_addr_p]-I2[Right_addr_p]);

					//========AD Cost
					cost = Box_Cost(I1,I2,i,j,d,2,dims);			
					//========Census Cost
					//cost = Census_Cost(I1,I2,i,j,d,5,dims);		
					//========ASW Cost
					//cost = ASW_Cost(I1,I2,i,j,d,5,dims);
					cost += preWeight*(min(preRobust,abs(d-preDisp)));
					if(d==0||cost<min_cost){
						min_cost = cost;
						min_dis = (unsigned char)d; 
					}
				}
			}
			D2[Dispar_addr] = (float)min_dis;
			D1[Dispar_addr] = (float)50;		
		}
	}
}
else{
        // printf("=================================================\n");
        // printf("Processing BP Process============================\n");
        // printf("=================================================\n");
	if(Post_process!=1){
	    
	    //BP_process(I1,I2,D1,D2,dims);
	    BP_process_Overlap(I1,I2,D1,D2,dims,false);	
	    //==========For Right Disparity====================
        if(LR_mode==1){
        // printf("=================================================\n");
        // printf("Processing BP Process For Right==================\n");
        // printf("=================================================\n");
            BP_process_Overlap(I2,I1,D2,D1,dims,true);	
	}
        }
	else{
            
	    //BP_process(I1,I2,D2,D1,dims);
	    BP_process_Overlap(I1,I2,D2,D1,dims,false);
	    if(LR_mode==1){
	    BP_process_Overlap(I2,I1,D1,D2,dims,true);
	    }
	}
}

// printf("Finish Optimization\n");

if(LR_mode==1){
    if(BP_mode==1){
			// printf("=================================================\n");
			// printf("==================LRC============================\n");

		if(Post_process==0){
			LRC(D1,D2, dims);
		}
		else{
			LRC(D2,D1, dims);
		}

		// printf("=================================================\n");
    }
}
Filter filter;

const double sigma_s = 60.0;
const double sigma_r = 0.4;
const int    maxiter = 1;

//=============Prepare for Joint_WMF======================
Mat Img_L_M(height,width,CV_8U);
Mat Img_D_M(height,width,CV_8U);
Mat Filtered_M;
JointWMF JointWMF_;
IplImage Filtered_I;



if(Post_process==1){
    // printf("=================================================\n");
    // printf("Processing Post Process==========================\n");
    // printf("=================================================\n");
    //domainTransformFilter(D2, D1, I1, sigma_s,sigma_r,maxiter,dims);
    filter.Weight_MOD( I1, D2, D1, 5,dims,ndisp);
    /*
for(int i=0;i<width;i++){
    for(int j=0;j<height;j++){
        int Dispar_addr = Get_addr(width,i,j) ; 
        int Left_addr_p = Get_addr(bpl,i,j) ;
	Img_L_M.at<uchar>(i,j) = I1[Left_addr_p];
	Img_D_M.at<uchar>(i,j) = D2[Dispar_addr];
    }
}




    //Filtered_M = JointWMF_.filter(Img_D_M,Img_L_M,10);

    
    for(int i=0;i<width;i++){
        for(int j=0;j<height;j++){
            int Dispar_addr = Get_addr(width,i,j) ; 
            int Left_addr_p = Get_addr(bpl,i,j) ;
	    D1[Dispar_addr] = Filtered_M.at<uchar>(i,j);
        }
    }
    */

}

Img_L_M.release();
Img_D_M.release();


//Mat Img_L_M(height,width,CV_8UC1,I1);
//Mat Img_D_M(height,width,CV_32F,D1);


/*

for(int i=0;i<width;i++){
    for(int j=0;j<height;j++){
        int Dispar_addr = Get_addr(width,i,j) ; 
        int Left_addr_p = Get_addr(bpl,i,j) ;
	Img_L_M.at<uchar>(i,j) = I1[Left_addr_p];
	Img_D_M.at<float>(i,j) = D1[Dispar_addr];
    }
}

//Mat Filtered_D = JointWMF::filter(Img_D_M,Img_L_M,10);

/*
for(int i=0;i<width;i++){
    for(int j=0;j<height;j++){
        int Dispar_addr = Get_addr(width,i,j) ; 
        int Left_addr_p = Get_addr(bpl,i,j) ;
	D1[Dispar_addr] = Img_D_M.at<float>(i,j);
    }
}

Img_L_M.release();
Img_D_M.release();
*/

#ifdef PROFILE
  timer.start("Support Matches");
#endif



#ifdef PROFILE
  timer.start("Delaunay Triangulation");
#endif


#ifdef PROFILE
  timer.start("Grid");
#endif

#ifdef PROFILE
  timer.plot();
#endif

  // release memory

  _mm_free(I1);
  _mm_free(I2);
}

int BP::Box_Cost(uint8_t* Left,uint8_t* Right,int x,int y,int d,int win_radius,const int32_t* dims){

  // get width, height and bytes per line
  width  = dims[0];
  height = dims[1];
  bpl    = width + 15-(width-1)%16;



int addr_left_p;
int addr_left_q;
int addr_right_p;
int addr_right_q;

int cost = 0;
  //         dims[0] = width of I1 and I2
  //         dims[1] = height of I1 and I2
  //         dims[2] = bytes per line (often equal to width, but allowed to differ)

if(x-d<0){
	cost       = 99999;
	return cost;
}

cost = 0;
addr_left_p  = Get_addr(bpl,x,y) ;
addr_right_p = Get_addr(bpl,x-d,y) ;
for(int i=x-win_radius;i<=x+win_radius;i++){
	
	for(int j=y-win_radius;j<=y+win_radius;j++){
		if(i<0||(i-d)<0||j<0||i>=width||j>=height)
			continue;
		addr_left_q   = Get_addr(bpl,i,j) ;
		addr_right_q  = Get_addr(bpl,i-d,j) ;

		cost += abs(Left[addr_left_q]-Right[addr_right_q]);

	}
}

return cost;

} 


int BP::Census_Cost(uint8_t* Left,uint8_t* Right,int x,int y,int d,int win_radius,const int32_t* dims){

  // get width, height and bytes per line
	width  = dims[0];
	height = dims[1];
	bpl    = width + 15-(width-1)%16;

	int addr_left_p;
	int addr_left_q;
	int addr_right_p;
	int addr_right_q;

	int cost = 0;
	//         dims[0] = width of I1 and I2
	//         dims[1] = height of I1 and I2
	//         dims[2] = bytes per line (often equal to width, but allowed to differ)

	if(x-d<0){
		cost       = 99999;
		return cost;
	}

	cost = 0;
	addr_left_p  = Get_addr(bpl,x,y) ;
	uint8_t Left_p = Left[addr_left_p]; 
	addr_right_p = Get_addr(bpl,x-d,y) ;
	uint8_t Right_p = Right[addr_right_p]; 
	for(int i=x-win_radius;i<=x+win_radius;i++){
		for(int j=y-win_radius;j<=y+win_radius;j++){
			if(i<0||(i-d)<0||(i-d)>=width||j<0||i>=width||j>=height)
				continue;
			addr_left_q   = Get_addr(bpl,i,j) ;
			addr_right_q  = Get_addr(bpl,i-d,j) ;
			uint8_t Left_q = Left[addr_left_q]; 
			uint8_t Right_q = Right[addr_right_q]; 

			cost += (Left_q>=Left_p)^(Right_q>=Right_p);

		}
	}

	return cost;

} 


int BP::ASW_Cost(uint8_t* Left,uint8_t* Right,int x,int y,int d,int win_radius,const int32_t* dims){

  // get width, height and bytes per line
	width  = dims[0];
	height = dims[1];
	bpl    = width + 15-(width-1)%16;

	int dist_theta = 10;
	int app_theta  = 10;

	int addr_left_p;
	int addr_left_q;
	int addr_right_p;
	int addr_right_q;

	int cost = 0;
		//         dims[0] = width of I1 and I2
		//         dims[1] = height of I1 and I2
		//         dims[2] = bytes per line (often equal to width, but allowed to differ)

	if(x-d<0){
		cost       = 99999;
		return cost;
	}

	float cost_up = 0;
	float cost_down = 0;
	addr_left_p  = Get_addr(bpl,x,y) ;
	//uint8_t Left_p = Left[addr_left_p]; 
	addr_right_p = Get_addr(bpl,x-d,y) ;
	//uint8_t Right_p = Right[addr_right_p]; 

	float weight_left=0;
	float weight_right=0;
	float weight_dist=0;



	for(int i=x-win_radius;i<=x+win_radius;i++){
		for(int j=y-win_radius;j<=y+win_radius;j++){
			if(i<0||(i-d)<0||j<0||i>=width||j>=height)
				continue;
			addr_left_q   = Get_addr(bpl,i,j) ;
			addr_right_q  = Get_addr(bpl,i-d,j) ;
			//uint8_t Left_q = Left[addr_left_q]; 
			//uint8_t Right_q = Right[addr_right_q]; 
			
			weight_dist = exp(-sqrt((i-x)*(i-x)+(j-y)*(j-y))/(float)app_theta);
			weight_left = exp(-abs(Left[addr_left_q]-Left[addr_left_p])/(float)app_theta);
			weight_right = exp(-abs(Right[addr_right_q]-Right[addr_right_p])/(float)app_theta);
			
			weight_left = weight_left * weight_dist;
			weight_right = weight_right * weight_dist;

			cost_up += weight_left*weight_right*(0.25*abs(Left[addr_left_q]-Right[addr_right_q])+0.75*Census_Cost(Left,Right,i,j,d,5,dims))	;
			cost_down += weight_left*weight_right;
		}
	}
	cost = cost_up/cost_down;
	return cost;
}  




void BP::BP_process_Overlap(uint8_t* Left,uint8_t* Right,float* D1_,float* D2_,const int32_t* dims,bool For_Right){

  // get width, height and bytes per line
  width  = dims[0];
  height = dims[1];
  bpl    = width + 15-(width-1)%16;
  int dis_range = param.disp_max+1;

  //printf("Overlap\n\nSize of Image width %d height %d disp %d\n",width,height, dis_range);
  
  unsigned char tile_size = 32; 
  bool residul_x  = (width%tile_size==0)?0:1;
  bool residul_y  = (height%tile_size==0)?0:1; 
  int tile_num_x = (width/tile_size + residul_x);
  int tile_num_y = (height/tile_size + residul_y);  
  int total_tile_num = tile_num_x*tile_num_y;
 
  int Message_size = tile_size*tile_size*dis_range;
  int Buffer_size  = total_tile_num*tile_size*dis_range;


  // copy images to byte aligned memory



  M_Cost  = (float*)_mm_malloc(Message_size*sizeof(float),64);
  M_Up    = (float*)_mm_malloc(Message_size*sizeof(float),64);
  M_Down  = (float*)_mm_malloc(Message_size*sizeof(float),64);
  M_Left  = (float*)_mm_malloc(Message_size*sizeof(float),64);
  M_Right = (float*)_mm_malloc(Message_size*sizeof(float),64);

  //printf("Finish Allocate Messgae\n");

  B_Up    = (float*)_mm_malloc(Buffer_size*sizeof(float),64);
  B_Down  = (float*)_mm_malloc(Buffer_size*sizeof(float),64);
  B_Left  = (float*)_mm_malloc(Buffer_size*sizeof(float),64);
  B_Right = (float*)_mm_malloc(Buffer_size*sizeof(float),64);

  //printf("Finish Allocate Buffer\n");

  Message_A = (float*)_mm_malloc(dis_range*sizeof(float),64);
  Message_B = (float*)_mm_malloc(dis_range*sizeof(float),64);
  Message_C = (float*)_mm_malloc(dis_range*sizeof(float),64);
  Message_D = (float*)_mm_malloc(dis_range*sizeof(float),64);
  Message_Cost = (float*)_mm_malloc(dis_range*sizeof(float),64);
  Message_Out = (float*)_mm_malloc(dis_range*sizeof(float),64);

  //printf("Finish Allocate Vector\n");

  //M_UP   = (uint8_t*)_mm_malloc(bpl*height*sizeof(float),16);
//===================Message of intra-tile
  memset (M_Cost ,0,Message_size*sizeof(float));
  memset (M_Up   ,0,Message_size*sizeof(float));
  memset (M_Down ,0,Message_size*sizeof(float));
  memset (M_Left ,0,Message_size*sizeof(float));
  memset (M_Right,0,Message_size*sizeof(float));
//===================Buffer of inter-tile
  memset (B_Up   ,0,Buffer_size*sizeof(float));
  memset (B_Down ,0,Buffer_size*sizeof(float));
  memset (B_Left ,0,Buffer_size*sizeof(float));
  memset (B_Right,0,Buffer_size*sizeof(float));

  memset (Message_A ,0,dis_range*sizeof(float));
  memset (Message_B ,0,dis_range*sizeof(float));
  memset (Message_C ,0,dis_range*sizeof(float));
  memset (Message_D ,0,dis_range*sizeof(float));
  memset (Message_Cost ,0,dis_range*sizeof(float));
  memset (Message_Out ,0,dis_range*sizeof(float));

  int lamda = 8;

  for(int outer_iter;outer_iter<Outer_Iteration;outer_iter++){
   
  for(int j=0;j<tile_num_y;j++){
    if(j%10==0)
    	//printf("Tile_y is %d\n",j);

    	for(int i=0;i<tile_num_x;i++){

	int tile_index = j*tile_num_x + i;
	//==========Intra Message Initial====================
	for(int m = 0;m<tile_size;m++){
	  for(int n=0;n<tile_size;n++){
	  int pixel_index = n*tile_size + m;
	  //========Up Message
	  if(n==0){
		for(int d=0;d<dis_range;d++){
		  M_Up[pixel_index*dis_range+d]  = B_Up[tile_index*tile_size*dis_range+m*dis_range+d]; 
		}		
	  }
	  else{
		for(int d=0;d<dis_range;d++){
		  M_Up[pixel_index*dis_range+d]  = 0; 
		}		
	  }
	  //========Down Message
	  if(n==tile_size-1){
		for(int d=0;d<dis_range;d++){
		  M_Down[pixel_index*dis_range+d]  = B_Down[tile_index*tile_size*dis_range+m*dis_range+d]; 
		}		
	  }
	  else{
		for(int d=0;d<dis_range;d++){
		  M_Down[pixel_index*dis_range+d]  = 0; 
		}		
	  }
	  //========Left Message
	  if(m==0){
		for(int d=0;d<dis_range;d++){
		  M_Left[pixel_index*dis_range+d]  = B_Left[tile_index*tile_size*dis_range+n*dis_range+d]; 
		}		
	  }
	  else{
		for(int d=0;d<dis_range;d++){
		  M_Left[pixel_index*dis_range+d]  = 0; 
		}		
	  }
	  //========Right Message
	  if(m==tile_size-1){
		for(int d=0;d<dis_range;d++){
		  M_Right[pixel_index*dis_range+d]  = B_Right[tile_index*tile_size*dis_range+n*dis_range+d]; 
		}		
	  }
	  else{
		
		for(int d=0;d<dis_range;d++){
		  M_Right[pixel_index*dis_range+d]  = 0; 
		}		
	  }
          //=========Cost
	  for(int d=0;d<dis_range;d++){
	    M_Cost[pixel_index*dis_range+d]  = 0;
	  }
        //===========End of For Message Initialization
	  }
	}
	//=====================
 	/*
        for(int k = 0;k<tile_size*tile_size;k++){
		for(int d=0;d<dis_range;d++){
		M_Up[k*dis_range+d]  = 0;
		M_Cost[k*dis_range+d]  = 0;
		M_Right[k*dis_range+d]  = 0;
		M_Left[k*dis_range+d]  = 0; 
		M_Down[k*dis_range+d]  = 0;
		} 
	}
        */
	//============================================
	//=================Cost ReFill================
	//============================================
	for(int m=0;m<tile_size;m++){
	  for(int n=0;n<tile_size;n++){
		int Left_addr_p = Get_addr(bpl,(i*tile_size+m),(j*tile_size+n)) ;
		int pixel_index = (n*tile_size+m)*dis_range;

		int Left_addr_pu;
		int Left_addr_pd;

                float Cost_0;
                float Cost_1;
		float Cost_2;
                if(n==0)
			Left_addr_pu = Left_addr_p;
		else
			Left_addr_pu = Get_addr(bpl,(i*tile_size+m),(j*tile_size+n-1)) ;
		if(n==tile_size-1)
			Left_addr_pd = Left_addr_p;
		else
			Left_addr_pd = Get_addr(bpl,(i*tile_size+m),(j*tile_size+n+1)) ;
//=========================================================================================
//=========================For Left Disparity=============================================
//=========================================================================================
//=========================================================================================	
		if(For_Right==0){
		for(int d = 0;d<dis_range;d++){
		  int Right_addr_p = Get_addr(bpl,(i*tile_size+m-d),(j*tile_size+n)) ;
		  int Right_addr_pu;
	          int Right_addr_pd;
                  if(n==0)
			Right_addr_pu = Right_addr_p;
		  else
			Right_addr_pu = Get_addr(bpl,(i*tile_size+m-d),(j*tile_size+n-1)) ;
		  if(n==tile_size-1)
			Right_addr_pd = Right_addr_p;
		  else
			Right_addr_pd = Get_addr(bpl,(i*tile_size+m-d),(j*tile_size+n+1)) ;

		  if(i*tile_size+m<d)
		    M_Cost[pixel_index+d] = 999;
		  else{
                    //AD Cost
                    //M_Cost[pixel_index+d] = abs(I1[Left_addr_p]-I2[Right_addr_p]);
                    //AD+Census Coist

                    Cost_0 =  abs(Left[Left_addr_p]-Right[Right_addr_p]);
                    Cost_0 =  min(Cost_0,90.0f);
		    Cost_1 =  abs((Left[Left_addr_pd]-Left[Left_addr_pu])-(Right[Right_addr_pd]-Right[Right_addr_pu]))/2;
                    //Cost_1 =  min(Cost_1,20.0f);
		    Cost_2 =  Census_Cost(Left,Right,(i*tile_size+m),(j*tile_size+n),d,5,dims); 

                    M_Cost[pixel_index+d] = 0.25*Cost_0+0.0*Cost_1+0.75*Cost_2;
//M_Cost[pixel_index+d] = 0.1*abs(I1[Left_addr_p]-I2[Right_addr_p])+0.15*abs((I1[Left_addr_pd]-I1[Left_addr_pu])-(I2[Right_addr_pd]-I2[Right_addr_pu]))/2-+0.75*Census_Cost(I1,I2,(i*tile_size+m),(j*tile_size+n),d,5,dims);
                    //Box Costs
		    //M_Cost[pixel_index+d] = Box_Cost(I1,I2,(i*tile_size+m),(j*tile_size+n),d,5,dims);	
		    //ASW
	            //M_Cost[pixel_index+d] = ASW_Cost(I1,I2,(i*tile_size+m),(j*tile_size+n),d,5,dims);
		  }
		}
	}
//=========================================================================================
//=========================For Right Disparity=============================================
//=========================================================================================
//=========================================================================================		
		else{
		for(int d = 0;d>-dis_range;d--){
		  int Right_addr_p = Get_addr(bpl,(i*tile_size+m-d),(j*tile_size+n)) ;
		  int Right_addr_pu;
	          int Right_addr_pd;
                  if(n==0)
			Right_addr_pu = Right_addr_p;
		  else
			Right_addr_pu = Get_addr(bpl,(i*tile_size+m-d),(j*tile_size+n-1)) ;
		  if(n==tile_size-1)
			Right_addr_pd = Right_addr_p;
		  else
			Right_addr_pd = Get_addr(bpl,(i*tile_size+m-d),(j*tile_size+n+1)) ;

		  if(i*tile_size+m<d)
		    M_Cost[pixel_index+abs(d)] = 999;
		  else{
                    //AD Cost
                    //M_Cost[pixel_index+d] = abs(I1[Left_addr_p]-I2[Right_addr_p]);
                    //AD+Census Coist

                    Cost_0 =  abs(Left[Left_addr_p]-Right[Right_addr_p]);
                    Cost_0 =  min(Cost_0,90.0f);
		    Cost_1 =  abs((Left[Left_addr_pd]-Left[Left_addr_pu])-(Right[Right_addr_pd]-Right[Right_addr_pu]))/2;
                    //Cost_1 =  min(Cost_1,20.0f);
		    Cost_2 =  Census_Cost(Left,Right,(i*tile_size+m),(j*tile_size+n),d,5,dims); 

                    M_Cost[pixel_index+abs(d)] = 0.0*Cost_0+0.25*Cost_1+0.75*Cost_2;
//M_Cost[pixel_index+d] = 0.1*abs(I1[Left_addr_p]-I2[Right_addr_p])+0.15*abs((I1[Left_addr_pd]-I1[Left_addr_pu])-(I2[Right_addr_pd]-I2[Right_addr_pu]))/2-+0.75*Census_Cost(I1,I2,(i*tile_size+m),(j*tile_size+n),d,5,dims);
                    //Box Costs
		    //M_Cost[pixel_index+d] = Box_Cost(I1,I2,(i*tile_size+m),(j*tile_size+n),d,5,dims);	
		    //ASW
	            //M_Cost[pixel_index+d] = ASW_Cost(I1,I2,(i*tile_size+m),(j*tile_size+n),d,5,dims);
		  }
		}
		}
//=========================================================================================
//=========================================================================================
//=========================================================================================
//=========================================================================================
	  }
	}

	//=================Inner Iteration
	for(int in_iter=0;in_iter<Inner_Iteration;in_iter++){
	
	//============================================
	//=================Left-To_Right==============
	//============================================
	  for(int m=0;m<tile_size;m++){
	    for(int n=0;n<tile_size;n++){
		int pixel_index = (n*tile_size+m)*dis_range;
		for(int d = 0;d<dis_range;d++){
		  Message_A[d] = M_Left[pixel_index+d];
		  Message_B[d] = M_Up[pixel_index+d];
		  Message_C[d] = M_Down[pixel_index+d];
		  Message_Cost[d] = M_Cost[pixel_index+d];
		}
 
                if(Weight_mode==1){
                  Weight_A = Message_Weight(Left,i*tile_size+m,j*tile_size+n,i*tile_size+m-1,j*tile_size+n,dims);
		  Weight_B = Message_Weight(Left,i*tile_size+m,j*tile_size+n,i*tile_size+m,j*tile_size+n-1,dims);
		  Weight_C = Message_Weight(Left,i*tile_size+m,j*tile_size+n,i*tile_size+m,j*tile_size+n+1,dims);
		}
		else{
		  Weight_A = 1;
		  Weight_B = 1;
		  Weight_C = 1;

		}


		//Message_Update(Message_A,Message_B,Message_C,Message_Cost,Message_Out,lamda,1,1,1);
		Message_Update(Message_A,Message_B,Message_C,Message_Cost,Message_Out,lamda,Weight_A,Weight_B,Weight_C);
                //==============Pass to next Tile if not boundary===========================
		if(m==tile_size-1){
	          if(i+1<tile_num_x){
		    for(int d = 0;d<dis_range;d++){
	    	      B_Left[(tile_index+1)*tile_size*dis_range+n*dis_range+d]=Message_Out[d];
		    }
		  }
		}//===============End Passing to Next Tile 
                //================Pass to the next pixel
		else{
		  for(int d = 0;d<dis_range;d++){
		    M_Left[pixel_index+dis_range+d] = Message_Out[d];

		  }
		}//===============End Passing to Next-Pixel
	    }
	  }//=====================End Left-To_Right
	
	//============================================
	//=================Right-To_Left==============
	//============================================
	  for(int m=tile_size-1;m>=0;m--){
	    for(int n=0;n<tile_size;n++){
		int pixel_index = (n*tile_size+m)*dis_range;
		for(int d = 0;d<dis_range;d++){
		  Message_A[d] = M_Right[pixel_index+d];
		  Message_B[d] = M_Up[pixel_index+d];
		  Message_C[d] = M_Down[pixel_index+d];
		  Message_Cost[d] = M_Cost[pixel_index+d];
                 
		}
                if(Weight_mode==1){
                  Weight_A = Message_Weight(Left,i*tile_size+m,j*tile_size+n,i*tile_size+m+1,j*tile_size+n,dims);
		  Weight_B = Message_Weight(Left,i*tile_size+m,j*tile_size+n,i*tile_size+m,j*tile_size+n-1,dims);
		  Weight_C = Message_Weight(Left,i*tile_size+m,j*tile_size+n,i*tile_size+m,j*tile_size+n+1,dims);
		}
		else{
		  Weight_A = 1;
		  Weight_B = 1;
		  Weight_C = 1;

		}
		//Message_Update(Message_A,Message_B,Message_C,Message_Cost,Message_Out,lamda,1,1,1);
		Message_Update(Message_A,Message_B,Message_C,Message_Cost,Message_Out,lamda,Weight_A,Weight_B,Weight_C);
                //==============Pass to next Tile if not boundary===========================
		if(m==0){
	          if(i-1>=0){
		    for(int d = 0;d<dis_range;d++){
	    	      B_Right[(tile_index-1)*tile_size*dis_range+n*dis_range+d]=Message_Out[d];
		    }
		  }
		}//===============End Passing to Next Tile 
                //==============Pass to the next pixel
		else{
		  for(int d = 0;d<dis_range;d++){
		    M_Right[pixel_index-dis_range+d] = Message_Out[d];

		  }
		}//===============End Passing to Next Pixel 
	    }
	  }//=====================End Right to Left
	
	//============================================
	//=================Up-To_Down==============
	//============================================
	    for(int n=0;n<tile_size;n++){	  
	      for(int m=0;m<tile_size;m++){
	    
		int pixel_index = (n*tile_size+m)*dis_range;
		for(int d = 0;d<dis_range;d++){
		  Message_A[d] = M_Up[pixel_index+d];
		  Message_B[d] = M_Left[pixel_index+d];
		  Message_C[d] = M_Right[pixel_index+d];
		  Message_Cost[d] = M_Cost[pixel_index+d];
		}
		if(Weight_mode==1){   
                  Weight_A = Message_Weight(Left,i*tile_size+m,j*tile_size+n,i*tile_size+m,j*tile_size+n-1,dims);
		  Weight_B = Message_Weight(Left,i*tile_size+m,j*tile_size+n,i*tile_size+m-1,j*tile_size+n,dims);
		  Weight_C = Message_Weight(Left,i*tile_size+m,j*tile_size+n,i*tile_size+m+1,j*tile_size+n,dims);
		}
		else{
		  Weight_A = 1;
		  Weight_B = 1;
		  Weight_C = 1;

		}

		//Message_Update(Message_A,Message_B,Message_C,Message_Cost,Message_Out,lamda,1,1,1);
		Message_Update(Message_A,Message_B,Message_C,Message_Cost,Message_Out,lamda,Weight_A,Weight_B,Weight_C);
                //==============Pass to next Tile if not boundary===========================
		if(n==tile_size-1){
	          if(j+1<tile_num_y){
		    for(int d = 0;d<dis_range;d++){
	    	      B_Up[(tile_index+tile_num_x)*tile_size*dis_range+m*dis_range+d]=Message_Out[d];
		    }
		  }
		}//===============End Passing to Next Tile 
                //==============Pass to the next pixel
		else{
		  for(int d = 0;d<dis_range;d++){
		    M_Up[pixel_index+dis_range*tile_size+d] = Message_Out[d];

		  }
		}//===============End Passing to Next Pixel 
	    }
	  }//=====================End Up to Down
	
	
	//============================================
	//=================Down-To-Up==============
	//============================================
        
	   for(int n=tile_size-1;n>=0;n--){	  
	     for(int m=0;m<tile_size;m++){
	    
		int pixel_index = (n*tile_size+m)*dis_range;
		for(int d = 0;d<dis_range;d++){
		  Message_A[d] = M_Down[pixel_index+d];
		  Message_B[d] = M_Left[pixel_index+d];
		  Message_C[d] = M_Right[pixel_index+d];
		  Message_Cost[d] = M_Cost[pixel_index+d];
		}
		if(Weight_mode==1){
                  Weight_A = Message_Weight(Left,i*tile_size+m,j*tile_size+n,i*tile_size+m,j*tile_size+n+1,dims);
		  Weight_B = Message_Weight(Left,i*tile_size+m,j*tile_size+n,i*tile_size+m-1,j*tile_size+n,dims);
		  Weight_C = Message_Weight(Left,i*tile_size+m,j*tile_size+n,i*tile_size+m+1,j*tile_size+n,dims);
		}
		else{
		  Weight_A = 1;
		  Weight_B = 1;
		  Weight_C = 1;

		}
		//Message_Update(Message_A,Message_B,Message_C,Message_Cost,Message_Out,lamda,1,1,1);
		Message_Update(Message_A,Message_B,Message_C,Message_Cost,Message_Out,lamda,Weight_A,Weight_B,Weight_C);
                //==============Pass to next Tile if not boundary===========================
		if(n==0){
	          if(j-1>=0){
		    for(int d = 0;d<dis_range;d++){
	    	      B_Down[(tile_index-tile_num_x)*tile_size*dis_range+m*dis_range+d]=Message_Out[d];
		    }
		  }
		}//===============End Passing to Next Tile 
                //==============Pass to the next pixel
		else{
		  for(int d = 0;d<dis_range;d++){
		    M_Down[pixel_index-dis_range*tile_size+d] = Message_Out[d];

		  }
		}//===============End Passing to Next Pixel 
	    }
	  }//=====================End Down to Up
	

 	//=================End of Inner Iteration
	}


	for(int m=0;m<tile_size;m++){



	  for(int n=0;n<tile_size;n++){
            int pixel_index = (n*tile_size+m)*dis_range;
	    for(int d = 0;d<dis_range;d++){

		 Message_A[d] = M_Left[pixel_index+d];
		 Message_B[d] = M_Right[pixel_index+d];
		 Message_C[d] = M_Up[pixel_index+d];
		 Message_D[d] = M_Down[pixel_index+d];
		 Message_Cost[d] = M_Cost[pixel_index+d];
	    }

		if(Weight_mode==1){
		  Weight_A = Message_Weight(Left,i*tile_size+m,j*tile_size+n,i*tile_size+m-1,j*tile_size+n,dims);
		  Weight_B = Message_Weight(Left,i*tile_size+m,j*tile_size+n,i*tile_size+m+1,j*tile_size+n,dims);
                  Weight_C = Message_Weight(Left,i*tile_size+m,j*tile_size+n,i*tile_size+m,j*tile_size+n-1,dims);
                  Weight_D = Message_Weight(Left,i*tile_size+m,j*tile_size+n,i*tile_size+m,j*tile_size+n+1,dims);
		}
		else{
		  Weight_A = 1;
		  Weight_B = 1;
		  Weight_C = 1;
		  Weight_D = 1;
		}
	    int Dispar_addr = Get_addr(width,i*tile_size+m,j*tile_size+n) ;
	    if((i*tile_size+m)>=width||(j*tile_size+n)>=height)
		continue;
            //printf("j %d pixel %d\n",j,j*tile_size+n);
//(i*tile_size+m),(j*tile_size+n)
            //int min_dis     = Message_Determine(Message_A,Message_B,Message_C,Message_D,Message_Cost,1,1,1,1);
            int min_dis     = Message_Determine(Message_A,Message_B,Message_C,Message_D,Message_Cost,Weight_A,Weight_B,Weight_C,Weight_D);
	    D1_[Dispar_addr] = (float)min_dis;
	    //D2_[Dispar_addr] = (float)50;
	  }
	}

  //=================End of For Tile Index i ,j	
    }
  }
  //=================End of Outer Iteration
  }


  free(M_Cost);
  free(M_Up);
  free(M_Down);
  free(M_Left);
  free(M_Right);

  free(B_Up);
  free(B_Down);
  free(B_Left);
  free(B_Right);

  free(Message_A);
  free(Message_B);
  free(Message_C);
  free(Message_D);
  free(Message_Cost);
  free(Message_Out);
}


void BP::Message_Update(float *Message_A,float *Message_B,float *Message_C,float *cost,float *Message_out,int lamda,float Weight_A,float Weight_B,float Weight_C){

  int dis_range = param.disp_max;
  int T = dis_range/8;
  int count = 0;
  float min_ref_message;
  int   min_ref_label;
  
  for(int i=0;i<=dis_range;i++){
    float temp_message = cost[i] + Weight_A*Message_A[i] + Weight_B*Message_B[i] + Weight_C*Message_C[i];
    if(i==0||temp_message<min_ref_message){
 	min_ref_label = i;
        min_ref_message = temp_message;
    }
  }
  
  min_ref_message = min_ref_message + lamda*T;
  float Total_min=999.0f;
  for(int i=0;i<=dis_range;i++){
    float min_message = 999.0f;
    
    int min_label;
    //for(int j =i-T;j<=i+T;j++){
    //	if(j<0||j>=dis_range)
    //		continue;
    for(int j=0;j<=dis_range;j++){
	int label_diff = min(abs(i-j),T);
	float temp_message =  cost[j] + Weight_A*Message_A[j] + Weight_B*Message_B[j] + Weight_C*Message_C[j] + lamda*label_diff;

	if(temp_message<min_message){
	  min_message = temp_message;
	}
    }

    if(min_message>min_ref_message)
       min_message = min_ref_message;

    Message_out[i] = min_message;

    if(i==0||Message_out[i]<Total_min){
	Total_min = Message_out[i];
	//count ++;
    }

  }
  //cout<<"Num of Label truncated"<<count/(dis_range+1)<<endl;

  float  *Buffer = new float[Num_k];
  for(int i=0;i<Num_k;i++){
	Buffer[i] = 999.0f;
  }

  for(int i=0;i<=dis_range;i++){
	Message_out[i] = Message_out[i]-Total_min;
	if(Truncated_label==true){
            if(Message_out[i]>=Buffer[Num_k-1])
	        continue;
	    for(int j=0;j<Num_k;j++){
	        if(Message_out[i]<=Buffer[j]){
		    for(int l=Num_k-1;l>j;l--){
		        Buffer[l] = Buffer[l-1];
	        	}		

		    Buffer[j] = Message_out[i];
		    break;
	        }
	    
	    }
	}
  }	
  int max;
  if(Truncated_label==true){
       for(int i=0;i<=dis_range;i++){
          if(Message_out[i]>max||i==0)
	      max = Message_out[i];
      }   


      for(int i=0;i<=dis_range;i++){
          if(Message_out[i]>Buffer[Num_k])
	      Message_out[i] = max;
      }
  }
  free(Buffer);
  


/*
  for(int i=0;i<=dis_range;i++){
	if(i%2==0||i==dis_range)
	    continue;
	else{
	    Message_out[i] = (Message_out[i-1]+Message_out[i+1])/2;
	}	
  }
*/

}


int BP::Message_Determine(float *Message_A,float *Message_B,float *Message_C,float *Message_D,float *cost,float Weight_A,float Weight_B,float Weight_C,float Weight_D){

  int dis_range = param.disp_max;
  float min_message;
  int   min_label;
  for(int i=0;i<=dis_range;i++){
    float temp_message = cost[i] + Weight_A*Message_A[i] + Weight_B*Message_B[i] + Weight_C*Message_C[i] + Weight_D*Message_D[i];
    if(i==0||temp_message<min_message){
 	min_label = i;
        min_message = temp_message;
    }
  }

  return min_label;

}

float BP::Message_Weight(uint8_t* Left,int ref_x,int ref_y,int tar_x,int tar_y,const int32_t* dims){

  // get width, height and bytes per line
  width  = dims[0];
  height = dims[1];
  bpl    = width + 15-(width-1)%16;
  int dis_range = param.disp_max+1;
  int diff_para = 5;
  int diff_thres = 10;
  if(tar_x<0||tar_y<0||tar_x>=width||tar_y>=height)
	return 0;

  int addr_left_p   = Get_addr(bpl,ref_x,ref_y) ;
  int addr_left_q   = Get_addr(bpl,tar_x,tar_y) ;
  int diff          = abs(Left[addr_left_p]-Left[addr_left_q]);
  diff              = (diff>diff_para*diff_thres)?diff_para*diff_thres:diff;
  float weight = pow(2,-(double)(diff/diff_para));
  
  return weight;

}

void BP::LRC(float* D1_,float* D2_,const int32_t* dims){
  // get width, height and bytes per line
  width  = dims[0];
  height = dims[1];
  bpl    = width + 15-(width-1)%16;
  int dis_range = param.disp_max+1;
  
  for(int i=0;i<width;i++){
    for(int j=0;j<height;j++){
      int Dispar_addr = Get_addr(width,i,j) ;
      float Value_D1 = D1_[Dispar_addr];
      int   Loca_D2   = Get_addr(width,i-int(Value_D1),j) ;
      float Value_D2 = D2_[Loca_D2];	
      if(Value_D1==Value_D2)
	continue;
      else{
	D1_[Dispar_addr] = 0;
      }
	
    }
  }
 
}
