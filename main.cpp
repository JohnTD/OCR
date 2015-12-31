#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <typeinfo>
#include <math.h>

using namespace cv;
using namespace std;


int oTsu(int* src, int height, int width)
{
    double* histogram = new double[256]();

    for(int i = 0; i < height*width; ++i)
        histogram[src[i]]++;

    int size = height*width;

    int thresh = 0;

    long cnt0 = 0, cnt1 = 0;
    long sum0 = 0, sum1 = 0;
    double w0 = 0, w1 = 0;
    double u0 = 0, u1 = 0;
    double u = 0, G = 0;
    double maxG = 0;

    for(int i = 0; i < 256; ++i)
    {
        sum0 = 0, sum1 = 0;
        cnt0 = 0, cnt1 = 0;
        for(int j = 0; j < i; ++j)
        {
            cnt0 += histogram[j];
            sum0 += j * histogram[j];
        }
        u0 = (double)sum0 / cnt0;
        w0 = (double)cnt0 / size;
        
        for(int j = i; j < 256; ++j)
        {
            cnt1 += histogram[j];
            sum1 += j * histogram[j];
        }
        u1 = (double)sum1 / cnt1;
        w1 = (double)cnt1 / size;

        u = u0*w0 + u1*w1;
        G = w0 * w1 * (u0 - u1) * (u0 - u1);

        if(G > maxG)
        {
            maxG = G;
            thresh = i; 
        }
    }
    return thresh;
}

int main( int argc, char** argv )
{
    Mat img = imread(argv[1], 0);
    Mat test = imread(argv[1], 1);

    int height = img.rows;
    int width  = img.cols;

    int* src = new int[height*width]();

    for(int i = 0; i < height; ++i)
    {
        uchar* u = img.ptr<uchar>(i);
        for(int j = 0; j < width; ++j)
            src[i*width+j] = (int)*u++;
    }

    int thresh = oTsu(src, height, width);
    Canny(img, img, thresh*0.8, thresh*1.2);

    vector<vector<Point> > contours;
    Mat contourOutput = img.clone();

    findContours(contourOutput, contours, RETR_TREE, CV_CHAIN_APPROX_SIMPLE);

    RNG rng(12345);
    for(int i = 0; i < contours.size(); ++i)
    {
        Rect rect = boundingRect(contours[i]);
        if(rect.height < 3 || rect.height > 40 || rect.width < 3 || rect.width > 40)
            continue;
        Point p1(rect.x, rect.y);
        Point p2(rect.x+rect.width, rect.y+rect.height);
        Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
        rectangle(test, p1, p2, color);
    }

    //imwrite("../57_cnt.jpg", test);
    namedWindow("contours", WINDOW_AUTOSIZE);
    imshow("contours", test);
    waitKey(0);
}
