#include "lime.h"
#include <vector>
#include <iostream>

namespace lime_feature
{   
    lime::lime(int iters, float a, float r, float g, int strat, double mu){
        iterations = iters;
        alpha = a;
        rho = r;
        gamma = g;
        strategy = strat;
        miu0 = mu;
    }

    void lime::loadMatrix(cv::Mat input) {
        cv::Mat img;
        input.convertTo(img, CV_64F, 1.0 / 255.0);
        loadimage(img);
    }

    void lime::FloatArrayTomat(cv::Mat& mat, const float32_t *M, int rows, int cols) 
    {  
        #pragma omp parallel for num_threads(8)
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                mat.at<float>(i, j) = M[j * rows + i];
            }
        }
    }

    void lime::matToSrc1(const cv::Mat& mat, float32_t *M) 
    {   
        int rows = mat.rows;
        int cols = mat.cols;

        #pragma omp parallel for num_threads(8)
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                M[i*cols + j] = mat.at<float>(i, j);
            }
        }
    }

    void lime::matToSrc2(const cv::Mat& mat, float32_t *M) 
    {   
        int rows = mat.rows;
        int cols = mat.cols;

        #pragma omp parallel for num_threads(8)
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                M[j*rows + i] = mat.at<float>(i, j);
            }
        }
    }

    void lime::matrix_multiply_neon2(const float32_t* A, const float32_t* B, float32_t* C, int M, int N, int K)
    {   //4,12,8
        int A_idx;
        int B_idx;

        #pragma omp parallel for num_threads(8)
        for (int j = 0; j < N; j++) {
            for (int i = 0; i < M; i++) {
                float32x4_t sum = vdupq_n_f32(0.0f);
                A_idx = i * K;
                B_idx = j * K;
                for (int k = 0; k < K; k += 4) {
                    float32x4_t a = vld1q_f32(A+A_idx+k);
                    float32x4_t b = vld1q_f32(B+B_idx+k);
                    sum = vmlaq_f32(sum, a, b);
                }
                C[j * M + i] = vaddvq_f32(sum);
            }
        }
    }

    cv::Mat lime::NeonMultiplication2(const cv::Mat& src1, const cv::Mat& src2)
    {
        assert(src1.cols == src2.rows);

        const int ksize = 4;
        const int dst_rows = src1.rows;
        const int dst_cols = src2.cols;
        const int kcols = src1.cols;

        int temp_kcols = kcols + (ksize - kcols % ksize) % ksize;

        cv::Mat src1_ext(dst_rows, temp_kcols, src1.type(), cv::Scalar(0));
        cv::Mat src2_ext(temp_kcols, dst_cols, src2.type(), cv::Scalar(0));

        src1.copyTo(src1_ext(cv::Rect(0, 0, src1.cols, src1.rows)));
        src2.copyTo(src2_ext(cv::Rect(0, 0, src2.cols, src2.rows)));

        std::vector<float32_t> A(dst_rows * temp_kcols);
        matToSrc1(src1_ext, A.data());
        std::vector<float32_t> B(temp_kcols * dst_cols);
        matToSrc2(src2_ext, B.data());
        std::vector<float32_t> C(dst_rows * dst_cols);

        matrix_multiply_neon2(A.data(), B.data(), C.data(), dst_rows, dst_cols, temp_kcols);

        cv::Mat result(dst_rows, dst_cols, CV_32F);
        FloatArrayTomat(result, C.data() ,dst_rows, dst_cols);

        return result;
    }



    void lime::load(std::string imgPath) {
        cv::Mat img = cv::imread(imgPath);
        img.convertTo(img, CV_32F, 1.0 / 255.0);
        // 处理图像
        loadimage(img);
    }

    cv::Mat lime::fft2(cv::Mat input){
        cv::Mat dst;
        cv::Mat planes[] = {cv::Mat_<float>(input), cv::Mat::zeros(input.size(), CV_32F)};
        cv::Mat complex;
        cv::merge(planes, 2, complex);
        cv::dft(complex, dst, cv::DFT_COMPLEX_OUTPUT);
        return dst;
    }

    cv::Mat lime::fast_fft2(cv::Mat input){
        int rows = input.rows;
        int cols = input.cols;

        // float *in = new float[rows * cols];
        fftwf_complex *in = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * rows * cols);
        fftwf_complex *out = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * rows * cols);
        
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                in[i * cols + j][0] = static_cast<float>(input.at<float>(i, j));
                in[i * cols + j][1] = 0;
            }
        }

        // fftwf_plan plan = fftwf_plan_dft_r2c_2d(rows, cols, in, out, FFTW_ESTIMATE);
        fftwf_plan plan = fftwf_plan_dft_2d(rows, cols, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
        fftwf_execute(plan);

        fftwf_destroy_plan(plan);
        delete[] in;

        cv::Mat output(rows, cols, CV_32FC2);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                float re = static_cast<float>(out[i * cols + j][0]);
                float im = static_cast<float>(out[i * cols + j][1]);
                output.at<cv::Vec2f>(i, j) = cv::Vec2f(re, im);
            }
        }

        fftwf_free(out);

        return output;
    }


    cv::Mat lime::ifft2(cv::Mat input) {
        cv::Mat dst;
        cv::idft(input, dst, cv::DFT_REAL_OUTPUT+cv::DFT_SCALE , 0);
        return dst;
    }

    cv::Mat lime::fast_ifft2(cv::Mat input){
        int rows = input.rows;
        int cols = input.cols;

        // 创建输入和输出数组
        fftwf_complex *in = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * rows * cols);
        fftwf_complex *out = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * rows * cols);

        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                in[i * cols + j][0] = input.at<cv::Vec2f>(i, j)[0];
                in[i * cols + j][1] = input.at<cv::Vec2f>(i, j)[1];
            }
        }

        fftwf_plan plan = fftwf_plan_dft_2d(rows, cols, in, out, FFTW_BACKWARD, FFTW_ESTIMATE);

        fftwf_execute(plan);

        fftwf_destroy_plan(plan);
        fftwf_free(in);

        cv::Mat output(rows, cols, CV_32FC1);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                output.at<float>(i, j) = static_cast<float>(out[i * cols + j][0]);
            }
        }

        // 释放输出数组所占用的内存
        fftwf_free(out);

        return output;
    }

    cv::Mat lime::fftshift(cv::Mat input){
        CV_Assert(input.type() == CV_32FC2 || input.type() == CV_64FC2);
        cv::Mat planes[2];
        cv::split(input, planes);
        int cx = input.cols / 2;
        int cy = input.rows / 2;
        cv::Mat q0_real(planes[0], cv::Rect(0, 0, cx, cy));
        cv::Mat q1_real(planes[0], cv::Rect(cx, 0, cx, cy));
        cv::Mat q2_real(planes[0], cv::Rect(0, cy, cx, cy));
        cv::Mat q3_real(planes[0], cv::Rect(cx, cy, cx, cy));
        cv::Mat tmp_real;
        q0_real.copyTo(tmp_real);
        q3_real.copyTo(q0_real);
        tmp_real.copyTo(q3_real);
        q1_real.copyTo(tmp_real);
        q2_real.copyTo(q1_real);
        tmp_real.copyTo(q2_real);

        cv::Mat q0_imag(planes[1], cv::Rect(0, 0, cx, cy));
        cv::Mat q1_imag(planes[1], cv::Rect(cx, 0, cx, cy));
        cv::Mat q2_imag(planes[1], cv::Rect(0, cy, cx, cy));
        cv::Mat q3_imag(planes[1], cv::Rect(cx, cy, cx, cy));
        cv::Mat tmp_imag;
        q0_imag.copyTo(tmp_imag);
        q3_imag.copyTo(q0_imag);
        tmp_imag.copyTo(q3_imag);
        q1_imag.copyTo(tmp_imag);
        q2_imag.copyTo(q1_imag);
        tmp_imag.copyTo(q2_imag);

        cv::merge(planes, 2, input);

        return input;
    }

    cv::Mat lime::conj(const cv::Mat& complex_mat){
        cv::Mat conj_mat(complex_mat.size(), CV_32FC2);
        //对复数矩阵进行遍历
        for (int i = 0; i < complex_mat.rows; i++) {
            for (int j = 0; j < complex_mat.cols; j++) {
                // 取出矩阵中的每个元素
                const cv::Vec2f& elem = complex_mat.at<cv::Vec2f>(i, j);
                // 将元素的虚部取负
                cv::Vec2f& conj_elem = conj_mat.at<cv::Vec2f>(i, j);
                conj_elem[0] = elem[0];
                conj_elem[1] = -elem[1];
            }
        }
        return conj_mat;
    }

    // cv::Mat lime::real(const cv::Mat& complex_mat) {
    //     CV_Assert(complex_mat.type() == CV_32FC2 || complex_mat.type() == CV_64FC2);
    //     std::vector<cv::Mat> channels;
    //     cv::split(complex_mat, channels);
    //     return channels[0];
    // }

    void lime::loadimage(cv::Mat L) {
        this->L = L;
        this->row = this->L.rows;
        this->col = this->L.cols;

        this->T_esti = cv::Mat::zeros(L.size(), CV_32FC1);
        cv::Mat L_channels[3];
        cv::split(L, L_channels);
        cv::Mat L_max;
        cv::max(L_channels[0], L_channels[1], L_max);
        cv::max(L_max, L_channels[2], L_max);
        L_max.convertTo(T_esti, CV_32F);

        //Dhx 450*450
        this->Dh = cv::Mat::zeros(col, col, CV_32FC1);
        for (int j = 0; j < col - 1; j++) {
            this->Dh.at<float>(j, j) = -1;
            this->Dh.at<float>(j, j + 1) = 1;
        }
        this->Dh.at<float>(col - 1, col - 1) = -1;
        this->Dh.at<float>(col - 1, 0) = 1;

        //Dvy 500*501
        this->Dvy = cv::Mat::zeros(row, row + 1, CV_32FC1);
        for (int i = 0; i <= row - 1; i++) {
            this->Dvy.at<float>(i, i) = -1;
            this->Dvy.at<float>(i, i + 1) = 1;
        }

        //Dhx 451*450
        this->Dhx = cv::Mat::zeros(col + 1, col, CV_32FC1);
        for (int j = 0; j <= col - 1; j++) {
            this->Dhx.at<float>(j, j) = -1;
            this->Dhx.at<float>(j + 1, j) = 1;
        }

        cv::Mat dx = cv::Mat::zeros(row, col, CV_32FC1);
        cv::Mat dy = cv::Mat::zeros(row, col, CV_32FC1);
        dx.at<float>(1, 0) = 1;
        dx.at<float>(1, 1) = -1;
        dy.at<float>(0, 1) = 1;
        dy.at<float>(1, 1) = -1;

        cv::Mat dxf = fftshift(fast_fft2(dx));
        cv::Mat dyf = fftshift(fast_fft2(dy));
        // printMat(dxf);

        cv::Mat test;
        cv::mulSpectrums(conj(dxf), dxf, test, 0);
        cv::Mat tmp;
        cv::mulSpectrums(conj(dyf), dyf, tmp, 0);
        cv::add(tmp, test, this->DTD);

        Strategy(5);
    }

    void lime::Strategy(int kernel_size) {
        if (strategy == 2) {
            int p = this->row * this->col;
            cv::Mat delTi = multiplyd(this->T_esti).t();
            cv::Mat dtvec = delTi.reshape(1, 2 * p);

            // separating parts used for W_x and W_y
            cv::Mat dtx = dtvec(cv::Range(0, p), cv::Range::all());
            cv::Mat dty = dtvec(cv::Range(p, 2 * p), cv::Range::all());

            // obtaining W_x and W_y
            cv::Mat w_gauss = cv::getGaussianKernel(kernel_size, 2.0, CV_32FC1);
            cv::Mat convl_x, convl_y; 
            // Get convl_x!
            cv::copyMakeBorder(dtx, dtx, 2, 2, 0, 0, cv::BORDER_CONSTANT, 0);
            dtx.at<double>(dtx.rows-1, 0) = 0.0;
            cv::filter2D(dtx, convl_x, -1, w_gauss, cv::Point(-1, -1), 0, cv::BORDER_CONSTANT);
            convl_x = convl_x.rowRange(2, convl_x.rows - 2);
            // Get convl_y!
            cv::copyMakeBorder(dty, dty, 2, 2, 0, 0, cv::BORDER_CONSTANT, 0);
            dty.at<double>(0, 0) = 0.0;
            dty.at<double>(1, 0) = 0.0;
            dty.at<double>(dty.rows-1, 0) = 0.0;
            cv::filter2D(dty, convl_y, -1, w_gauss, cv::Point(-1, -1), 0, cv::BORDER_CONSTANT);
            convl_y = convl_y.rowRange(2, convl_y.rows - 2);

            cv::Mat W_x, W_y;
            W_x = 1.0 / (cv::abs(convl_x) + 0.0001);
            W_y = 1.0 / (cv::abs(convl_y) + 0.0001);
            cv::Mat W_vec;
            cv::vconcat(W_x, W_y, W_vec);
            W_vec = W_vec.reshape(1, this->col);
            this->W = W_vec.t();
        } else {
            this->W = cv::Mat::ones(row * 2, col, CV_32FC1);
        }
    }


    cv::Mat lime::Complex_div(const cv::Mat& complex_mat1,const cv::Mat& complex_mat2){
        cv::Mat result = cv::Mat::zeros(complex_mat1.size(), CV_32FC2);
        cv::Mat tmp = cv::Mat::zeros(complex_mat1.size(), CV_32FC2);
        cv::mulSpectrums(complex_mat1, conj(complex_mat2), tmp, 0);
        for (int i = 0; i < tmp.rows; ++i) {
            for (int j = 0; j < tmp.cols; ++j) {
                cv::Vec2f num = tmp.at<cv::Vec2f>(i, j);
                cv::Vec2f den = complex_mat2.at<cv::Vec2f>(i, j);
                result.at<cv::Vec2f>(i, j) = num / (pow(den[0],2)+pow(den[1],2));
            }
        }
        return result;
    }

    cv::Mat lime::rescale_intensity(cv::Mat T,double in_min, double in_max, double out_min, double out_max)
    {
        cv::Mat result = (T - in_min) / (in_max - in_min);
        cv::multiply(result, (out_max - out_min), result);
        cv::add(result, out_min, result);
        return result;
    }

    cv::Mat lime::multiplydtrans(cv::Mat G){
        cv::Mat G_input = G.t();
        cv::Mat g = G_input.reshape(1, this->row * this->col * 2);
        cv::Mat Gx = g.rowRange(0, this->row * this->col).reshape(1, this->col);
        Gx = Gx.t();
        cv::Mat Gy = g.rowRange(this->row * this->col, this->row * this->col * 2).reshape(1, this->col);
        Gy = Gy.t();

        cv::Mat altGy = cv::Mat::zeros(this->row + 1, this->col, CV_32FC1);
        cv::Mat subaltGy = altGy(cv::Range(1, this->row+1), cv::Range(0, this->col));
        Gy.copyTo(subaltGy);
        cv::Mat subG = Gy(cv::Range(this->row-1, this->row), cv::Range(0, this->col-1));
        cv::Mat subaltGy2 = altGy(cv::Range(0, 1), cv::Range(1, this->col));
        subG.copyTo(subaltGy2);
        altGy.at<double>(0, 0) = Gy.at<double>(this->row-1, this->col-1);
        
        //Xh,altGy

        cv::Mat Dy = -this->Dvy;    //500*501
        cv::Mat delGy = NeonMultiplication2(Dy, altGy);
        cv::Mat delGx = NeonMultiplication2(Gx, this->Dh);
        cv::Mat delX = delGx + delGy;
        
        return delX;
    }

    cv::Mat lime::multiplydtrans_withoutNeon(cv::Mat G){
        cv::Mat G_input = G.t();
        cv::Mat g = G_input.reshape(1, this->row * this->col * 2);
        cv::Mat Gx = g.rowRange(0, this->row * this->col).reshape(1, this->col);
        Gx = Gx.t();
        cv::Mat Gy = g.rowRange(this->row * this->col, this->row * this->col * 2).reshape(1, this->col);
        Gy = Gy.t();

        cv::Mat altGy = cv::Mat::zeros(this->row + 1, this->col, CV_32FC1);
        cv::Mat subaltGy = altGy(cv::Range(1, this->row+1), cv::Range(0, this->col));
        Gy.copyTo(subaltGy);
        cv::Mat subG = Gy(cv::Range(this->row-1, this->row), cv::Range(0, this->col-1));
        cv::Mat subaltGy2 = altGy(cv::Range(0, 1), cv::Range(1, this->col));
        subG.copyTo(subaltGy2);
        altGy.at<double>(0, 0) = Gy.at<double>(this->row-1, this->col-1);
        
        //Xh,altGy

        cv::Mat Dy = -this->Dvy;    //500*501
        cv::Mat delGy = Dy * altGy;
        cv::Mat delGx = Gx * this->Dh;
        cv::Mat delX = delGx + delGy;
        
        return delX;
    }

    cv::Mat lime::multiplyd(cv::Mat T){
        cv::Mat altTy = cv::Mat::zeros(this->row + 1, this->col, CV_32FC1);
        cv::Mat subAltTy = altTy(cv::Range(0, this->row), cv::Range(0,  this->col));
        T.copyTo(subAltTy);
        cv::Mat subT = T(cv::Range(0, 1), cv::Range(1, this->col));
        cv::Mat subAltTy2 = altTy(cv::Range(this->row, this->row+1), cv::Range(0, this->col-1));
        subT.copyTo(subAltTy2);
        altTy.at<float>(this->row, this->col-1) = T.at<float>(0, 0);

        cv::Mat altTx = cv::Mat::zeros(this->row, this->col + 1, CV_32FC1);
        cv::Mat subAltTx = altTx(cv::Range(0, this->row), cv::Range(0, this->col));
        T.copyTo(subAltTx);
        cv::Mat subT0 = T(cv::Range(0, this->row),cv::Range(0, 1));
        cv::Mat subAltTx2 = altTx(cv::Range(0, this->row), cv::Range(this->col, this->col+1));
        subT0.copyTo(subAltTx2);

        cv::Mat temp1 = NeonMultiplication2(this->Dvy, altTy);
        cv::Mat temp2 = NeonMultiplication2(altTx, this->Dhx);

        // 将 delTx 和 delTy 转换成一维矩阵
        cv::Mat delTy = temp1.t();
        cv::Mat dty = delTy.reshape(1, this->row * this->col);
        cv::Mat delTx = temp2.t();
        cv::Mat dtx = delTx.reshape(1, this->row * this->col);
        // 将 dtx 和 dty 连接在一起成为新的一维矩阵 dt
        cv::Mat dt;
        cv::vconcat(dtx, dty, dt);
        // 将 dt 重新变成二维矩阵 delT
        cv::Mat temp3 = dt.reshape(1, this->col);
        cv::Mat delT = temp3.t();

        return delT;
    }

    cv::Mat lime::multiplyd_withoutNeon(cv::Mat T){
        cv::Mat altTy = cv::Mat::zeros(this->row + 1, this->col, CV_32FC1);
        cv::Mat subAltTy = altTy(cv::Range(0, this->row), cv::Range(0,  this->col));
        T.copyTo(subAltTy);
        cv::Mat subT = T(cv::Range(0, 1), cv::Range(1, this->col));
        cv::Mat subAltTy2 = altTy(cv::Range(this->row, this->row+1), cv::Range(0, this->col-1));
        subT.copyTo(subAltTy2);
        altTy.at<float>(this->row, this->col-1) = T.at<float>(0, 0);

        cv::Mat altTx = cv::Mat::zeros(this->row, this->col + 1, CV_32FC1);
        cv::Mat subAltTx = altTx(cv::Range(0, this->row), cv::Range(0, this->col));
        T.copyTo(subAltTx);
        cv::Mat subT0 = T(cv::Range(0, this->row),cv::Range(0, 1));
        cv::Mat subAltTx2 = altTx(cv::Range(0, this->row), cv::Range(this->col, this->col+1));
        subT0.copyTo(subAltTx2);

        cv::Mat temp1 = this->Dvy * altTy;
        cv::Mat temp2 = altTx * this->Dhx;

        // 将 delTx 和 delTy 转换成一维矩阵
        cv::Mat delTy = temp1.t();
        cv::Mat dty = delTy.reshape(1, this->row * this->col);
        cv::Mat delTx = temp2.t();
        cv::Mat dtx = delTx.reshape(1, this->row * this->col);
        // 将 dtx 和 dty 连接在一起成为新的一维矩阵 dt
        cv::Mat dt;
        cv::vconcat(dtx, dty, dt);
        // 将 dt 重新变成二维矩阵 delT
        cv::Mat temp3 = dt.reshape(1, this->col);
        cv::Mat delT = temp3.t();

        return delT;
    }

    cv::Mat lime::T_sub(cv::Mat G, cv::Mat Z, double miu) {
        // X=G-U;
        cv::Mat X = G - (Z / miu);
        cv::Mat delX = multiplydtrans(X);
        cv::Mat temp1 = 2 * this->T_esti + miu * delX;
        // cv::Mat temp1 = 2 * this->T_esti + miu * (this->Dv * Xv + Xh * this->Dh);   //Tnum=2*Ti+mu*delX;

        cv::Mat numerator = fftshift(fast_fft2(temp1));  
        //Td=2+mu*(dx_mod+dy_mod);
        cv::Mat denominator;
        cv::Scalar s(2);                     
        cv::Mat two(DTD.size(), CV_32FC2, s);
        cv::add(this->DTD * miu, two, denominator); 
        cv::Mat ratio = Complex_div(numerator, denominator);
        cv::Mat T = ifft2(fftshift(ratio));

        return T;
    }

    cv::Mat lime::T_sub_withoutNeon(cv::Mat G, cv::Mat Z, double miu) {
        // X=G-U;
        cv::Mat X = G - (Z / miu);
        cv::Mat delX = multiplydtrans_withoutNeon(X);
        cv::Mat temp1 = 2 * this->T_esti + miu * delX;
        // cv::Mat temp1 = 2 * this->T_esti + miu * (this->Dv * Xv + Xh * this->Dh);   //Tnum=2*Ti+mu*delX;

        cv::Mat numerator = fftshift(fast_fft2(temp1));  

        //Td=2+mu*(dx_mod+dy_mod);
        cv::Mat denominator;
        cv::Scalar s(2);                     
        cv::Mat two(DTD.size(), CV_32FC2, s);
        cv::add(this->DTD * miu, two, denominator); 
        cv::Mat ratio = Complex_div(numerator, denominator);
        cv::Mat T = ifft2(fftshift(ratio));

        return T;
    }
    

    cv::Mat lime::sign(cv::Mat input) {
        cv::Mat result = cv::Mat::zeros(input.size(), input.type());
        for (int row = 0; row < input.rows; ++row) {
            for (int col = 0; col < input.cols; ++col) {
                result.at<float>(row, col) = std::copysign(1.0, input.at<float>(row, col));
            }
        }
        return result;
    }

    //G=shrinkage(A,(delT+U)); A=alpha*W/mu;U=Z/mu;
    cv::Mat lime::G_sub(cv::Mat delT, cv::Mat Z, double miu, cv::Mat W) {
        //epsilon is A
        cv::Mat epsilon = this->alpha * W / miu; 
        //delT+U is temp:for temp3 is delT with U=Z/mu
        cv::Mat temp = delT + Z / miu;

        cv::Mat tempdst1 = cv::Mat::zeros(temp.size(), temp.type());
        tempdst1 = cv::abs(temp);
        tempdst1.convertTo(tempdst1, CV_32FC1);
        cv::Mat tempdst3 = tempdst1 - epsilon;
        cv::Mat tempdst4;
        //max(abs((delT+U))-A,Z) is tempdst4
        cv::Mat ling = cv::Mat::zeros(tempdst3.rows, tempdst3.cols, CV_32FC1);
        cv::max(tempdst3, ling, tempdst4);
        //sign((delT+U)) is tempdst5
        cv::Mat tempdst5;

        sign(temp).convertTo(tempdst5, CV_32FC1);

        //G=sign((delT+U)).*max(abs((delT+U))-epsilon,Z);
        return tempdst5.mul(tempdst4);
    }

    //Z=mu*(B+U)=Z+mu*(delT-G);  B=delT-G;U=Z/mu; 
    cv::Mat lime::Z_sub(cv::Mat delT, cv::Mat G, cv::Mat Z, double miu) {
        cv::Mat result = Z + miu * (delT - G); 
        return result;
    }

    double lime::miu_sub(double miu) {
        return miu * this->rho;
    }
    
    cv::Mat lime::run(){
        cv::Mat T, G, Z, delT;
        T = cv::Mat::zeros(row, col, CV_32FC1);
        delT = cv::Mat::zeros(row * 2, col, CV_32FC1);
        G = cv::Mat::zeros(row * 2, col, CV_32FC1);
        Z = cv::Mat::zeros(row * 2, col, CV_32FC1);
        double miu = this->miu0;

        for (int i = 0; i < iterations; i++) {
            T = T_sub(G, Z, miu);
            delT=multiplyd(T);
            G = G_sub(delT, Z, miu, this->W);
            Z = Z_sub(delT, G, Z, miu);
            miu = miu_sub(miu);
        }

        cv::Mat T_gamma;
        pow(T, gamma, T_gamma);
        cv::Mat T_3chan;
        cv::merge(std::vector<cv::Mat>{T_gamma, T_gamma, T_gamma}, T_3chan);
        this->L.convertTo(this->L, CV_32FC3);
        cv::Mat R = this->L.mul(1.0 / T_3chan);
        R.convertTo(R, CV_8U, 255.0);
        
        return R;
    }

    cv::Mat lime::run_withoutNeon(){
        cv::Mat T, G, Z, delT;
        T = cv::Mat::zeros(row, col, CV_32FC1);
        delT = cv::Mat::zeros(row * 2, col, CV_32FC1);
        G = cv::Mat::zeros(row * 2, col, CV_32FC1);
        Z = cv::Mat::zeros(row * 2, col, CV_32FC1);
        double miu = this->miu0;

        for (int i = 0; i < iterations; i++) {
            T = T_sub_withoutNeon(G, Z, miu);
            delT=multiplyd_withoutNeon(T);
            G = G_sub(delT, Z, miu, this->W);
            Z = Z_sub(delT, G, Z, miu);
            miu = miu_sub(miu);
        }

        cv::Mat T_gamma;
        pow(T, gamma, T_gamma);
        cv::Mat T_3chan;
        cv::merge(std::vector<cv::Mat>{T_gamma, T_gamma, T_gamma}, T_3chan);
        this->L.convertTo(this->L, CV_32FC3);
        cv::Mat R = this->L.mul(1.0 / T_3chan);
        R.convertTo(R, CV_8U, 255.0);
        
        return R;
    }
}