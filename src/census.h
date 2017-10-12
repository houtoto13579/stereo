using namespace std;
using namespace cv;

void census_33(Mat left, Mat right, Mat& dst, int disp, int block_size) {

	//declare variables
	int width = left.cols;
	int height = left.rows;
	int Min = INT_MAX;
	int argMin = 0;
	int half = (block_size / 2) + 1;
	//NOTE: like the gradient_xy function

	for(int i=0; i<height; ++i)
		for(int j=0; j<width; ++j) {
			for(int d=0; d<disp; ++d) {
				if((j - d) < 0) break;
				else if(i+half<height && j+half<width && i-half>0 && j-half-d>0) {
					float temp = 0;
					for(int x=j-half+1; x<j+half; ++x) // looping over box filter
						for(int y=i-half+1; y<i+half; ++y) {
							unsigned short int bin_L[9] = {}, bin_R[9] = {};
	
							for(int m=0; m<3; ++m) { // generating the binary code for a 3*3 window
								bin_L[m] = (left.at<uchar>(Point(x-1+m,y-1))>left.at<uchar>(Point(x,y))?1:0);
								bin_R[m] = (right.at<uchar>(Point(x-1-d+m,y-1))>right.at<uchar>(Point(x-d,y))?1:0);
								bin_L[m+3] = (left.at<uchar>(Point(x-1+m,y))>left.at<uchar>(Point(x,y))?1:0);
								bin_R[m+3] = (right.at<uchar>(Point(x-1-d+m,y))>right.at<uchar>(Point(x-d,y))?1:0);
								bin_L[m+6] = (left.at<uchar>(Point(x-1+m,y+1))>left.at<uchar>(Point(x,y))?1:0);
								bin_R[m+6] = (right.at<uchar>(Point(x-1-d+m,y+1))>right.at<uchar>(Point(x-d,y))?1:0);
							}
							for(int n=0; n<9; ++n) {
								bin_L[n] = (bin_L[n] != bin_R[n]?1:0);
								if(bin_L[n] == 1) temp++;
							}
						}
					if(temp < Min) {
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


