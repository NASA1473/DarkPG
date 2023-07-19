#ifndef FEATURE_EXACTION_LIME_H
#define FEATURE_EXACTION_LIME_H
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/base.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types_c.h>
#include <opencv2/core/matx.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/imgproc/imgproc_c.h>

#include <arm_neon.h>
#include <omp.h>
#include <fftw3.h>

#include <iostream>
#include <string>
#include <complex>


namespace lime_feature
{
class lime
{
public:
    int iterations;
    double alpha, rho, gamma, miu0;
    int strategy;
    cv::Mat L;
    int row, col;
    cv::Mat T_esti;
    cv::Mat Dh, DTD, Wv, Wh, W, Dvy, Dhx;
    

public:
    lime(int iters, float a, float r, float g, int strat, double mu);

    void loadMatrix(cv::Mat input);

    void matToSrc1(const cv::Mat& mat, float32_t *M);
    void matToSrc2(const cv::Mat& mat, float32_t *M);

    void FloatArrayTomat(cv::Mat& mat, const float32_t *M, int rows, int cols);
    
    void matrix_multiply_neon2(const float32_t* A, const float32_t* B, float32_t* C, int M, int N, int K);

    cv::Mat NeonMultiplication2(const cv::Mat& src1, const cv::Mat& src2);

    void load(std::string imgPath);

    cv::Mat fft2(cv::Mat input);
    cv::Mat fast_fft2(cv::Mat input);

    cv::Mat ifft2(cv::Mat input);
    cv::Mat fast_ifft2(cv::Mat input);

    cv::Mat fftshift(cv::Mat input);

    cv::Mat conj(const cv::Mat& complex_mat);

    void loadimage(cv::Mat _L);

    void Strategy(int kernel_size);

    cv::Mat multiplydtrans(cv::Mat G);
    cv::Mat multiplydtrans_withoutNeon(cv::Mat G);

    cv::Mat Complex_div(const cv::Mat& complex_mat1,const cv::Mat& complex_mat2);

    cv::Mat rescale_intensity(cv::Mat T,double in_min, double in_max, double out_min, double out_max);

    cv::Mat T_sub(cv::Mat G, cv::Mat Z, double miu);
    cv::Mat T_sub_withoutNeon(cv::Mat G, cv::Mat Z, double miu);

    cv::Mat multiplyd(cv::Mat T);
    cv::Mat multiplyd_withoutNeon(cv::Mat T);

    cv::Mat sign(cv::Mat arr);

    cv::Mat G_sub(cv::Mat T, cv::Mat Z, double miu, cv::Mat W);

    cv::Mat Z_sub(cv::Mat T, cv::Mat G, cv::Mat Z, double miu);

    double miu_sub(double miu);

    cv::Mat run();
    cv::Mat run_withoutNeon();

};

}

#endif //FEATURE_EXACTION_LIME_H
