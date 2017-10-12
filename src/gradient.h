#include<cmath>
using namespace std;
using namespace cv;

void gradient_x(Mat left, Mat right, Mat& dst, int disp, int block_size) {
	//declare variables
	int width = left.cols;
	int height = left.rows;
	int Min = INT_MAX; int argMin = 0;	
	int half = (block_size / 2) + 1;
	//NOTE: this method has a default block size == 1;
	//NOTE: only on the x_coordinate?
		
	//run the for loop
	for(int i=3; i<height; ++i) {
		for(int j=0; j<width; ++j) {
			for(int d=0; d<disp; ++d) {
				if((j - d)< 0) break;
				else if(i+half-1<height && j+half<width && i-half+1>0 && j-half-d>0) {

					float temp = 0;
					for(int x=j-half+1; x<j+half+1-1; ++x)//blocksize+1
						for(int y=i-half+1; y<i+half+1-1; ++y) {//subtract 1?
						
							uchar L1 = left.at<uchar>(Point(x-1, y));
							uchar L2 = left.at<uchar>(Point(x+1, y));
							uchar R1 = right.at<uchar>(Point(x-1-d, y));
							uchar R2 = right.at<uchar>(Point(x+1-d, y));
							int diff_l = L2 - L1;
							int diff_r = R2 - R1;
							temp += abs(diff_r - diff_l);
						}
					if (temp <= Min) {
						Min = temp;
						argMin = d;
					}
				}			
			}
			Min = INT_MAX;
			uchar a = argMin * (256/disp);
			dst.at<uchar>(Point(j, i)) = a;
		}
	}	
}

void gradient_y(Mat left, Mat right, Mat& dst, int disp, int block_size) {
	//declare variables
	int width = left.cols;
	int height = left.rows;
	int Min = INT_MAX; int argMin = 0;	
	int half = (block_size / 2) + 1;
	//NOTE: this method has a default block size == 1;
	//NOTE: only on the x_coordinate?
		
	//run the for loop
	for(int i=3; i<height; ++i) {
		for(int j=0; j<width; ++j) {
			for(int d=0; d<disp; ++d) {
				if((j - d)< 0) break;
				else if(i+half<height && j+half-1<width && i-half>0 && j-half-d+1>0) {

					float temp = 0;
					for(int x=j-half+1; x<j+half+1-1; ++x)//blocksize+1
						for(int y=i-half+1; y<i+half+1-1; ++y) {//subtract 1?
						
							uchar L1 = left.at<uchar>(Point(x, y-1));
							uchar L2 = left.at<uchar>(Point(x, y+1));
							uchar R1 = right.at<uchar>(Point(x-d, y-1));
							uchar R2 = right.at<uchar>(Point(x-d, y+1));
							int diff_l = L2 - L1;
							int diff_r = R2 - R1;
							temp += abs(diff_r - diff_l);
						}
					if (temp <= Min) {
						Min = temp;
						argMin = d;
					}
				}			
			}
			Min = INT_MAX;
			uchar a = argMin * (256/disp);
			dst.at<uchar>(Point(j, i)) = a;
		}
	}
}

void gradient_xy(Mat left, Mat right, Mat& dst, int disp, int block_size) {
	//declare variables
	int width = left.cols;
	int height = left.rows;
	int Min = INT_MAX; int argMin = 0;	
	int half = (block_size / 2) + 1;
	//NOTE: this method has a default block size == 1;
	//NOTE: only on the x_coordinate?
		
	//run the for loop
	for(int i=3; i<height; ++i) {
		for(int j=0; j<width; ++j) {
			for(int d=0; d<disp; ++d) {
				if((j - d)< 0) break;
				else if(i+half<height && j+half<width && i-half>0 && j-half-d>0) {

					float temp = 0;
					for(int x=j-half+1; x<j+half+1-1; ++x)//blocksize+1
						for(int y=i-half+1; y<i+half+1-1; ++y) {//subtract 1?
						
							uchar L1_xy = left.at<uchar>(Point(x, y-1));
							uchar L2_xy = left.at<uchar>(Point(x, y+1));
							uchar R1_xy = right.at<uchar>(Point(x-d, y-1));
							uchar R2_xy = right.at<uchar>(Point(x-d, y+1));
							int diff_l_xy = L2_xy - L1_xy;
							int diff_r_xy = R2_xy - R1_xy;
							temp += abs(diff_r_xy - diff_l_xy);
								
							L1_xy = left.at<uchar>(Point(x-1, y));
							L2_xy = left.at<uchar>(Point(x+1, y));
							R1_xy = right.at<uchar>(Point(x-1-d, y));
							R2_xy = right.at<uchar>(Point(x+1-d, y));
							diff_l_xy = L2_xy - L1_xy;
							diff_r_xy = R2_xy - R1_xy;
							temp += abs(diff_r_xy - diff_l_xy);
						}
					if (temp <= Min) {
						Min = temp;
						argMin = d;
					}
				}			
			}
			Min = INT_MAX;
			uchar a = argMin * (256/disp);
			dst.at<uchar>(Point(j, i)) = a;
		}
	}
}

void absolute_diff(Mat left, Mat right, Mat& dst, int disp, int block_size) {

	int width = left.cols;
	int height = left.rows;
	int Min = INT_MAX; int argMin = 0;
	int half_block = block_size / 2;

	for(int i=0; i<height; ++i) {
		for(int j=0; j<width; ++j) {
			for(int d=0; d<disp; ++d) {
				if ((j - d) < 0) break;
				else if (i+half_block<height && j+half_block<width && i-half_block>0 && j-half_block-d>0) {
					float temp = 0;
					for (int x=j-half_block; x<j+half_block+1;++x)
						for (int y=i-half_block; y<i+half_block+1;++y) {
							uchar color1 = left.at<uchar>(Point(x,y));
							uchar color2 = right.at<uchar>(Point(x-d,y));
								temp = temp + abs(color1 - color2);
						}
					if (temp <= Min) {
						Min = temp;	
						argMin = d;
					}
				}
			}
			Min = INT_MAX;
			uchar a = argMin*(256/disp);
			dst.at<uchar>(Point(j, i)) = a;
			//cout
		}
	}
}

void gradient_ad() {

}
