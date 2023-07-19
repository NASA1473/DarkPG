#include "yolo.h"
#include <vector>
#include <iostream>
#include <fstream>

namespace yolo_feature
{   
    //将class加载到vector中
    std::vector<std::string> yolo::load_class_list()
    {
        std::vector<std::string> class_list;
        std::ifstream ifs("../source/classes/class.txt");
        std::string line;
        while (getline(ifs, line))
        {
            //std::cout << "Class:---------------:::::::" << line << std::endl;
    
            class_list.push_back(line);
        }
        return class_list;
    }
 
    // 加载网络
    void yolo::load_net(cv::dnn::Net &net, bool is_cuda)
    {
        //使用readNet()函数加载YOLOV5S.ONNX文件
        cv::dnn::Net result = cv::dnn::readNet("../source/models/yolov5s.onnx");
        //std::cout<<"2"<<std::endl;
        //依据情况选定是否使用CUDA
        if (is_cuda)
        {
            std::cout << "Attempty to use CUDA\n";
            result.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
            result.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);
        }
        else
        {
            std::cout << "Running on CPU\n";
            result.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
            result.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        }
        net = result;
    }
    
    //定义框体颜色
    // const std::vector<cv::Scalar> colors = {cv::Scalar(255, 255, 0), cv::Scalar(0, 255, 0), cv::Scalar(0, 255, 255), cv::Scalar(255, 0, 0)};
 
 
    // 定义相关参数值与阈值
        // const float INPUT_WIDTH = 640.0;
        // const float INPUT_HEIGHT = 640.0;
        // const float SCORE_THRESHOLD = 0.2;
        // const float NMS_THRESHOLD = 0.4;
        // const float CONFIDENCE_THRESHOLD = 0.4;
    
    // // 定义输出结果的结构体类
    // struct Detection
    // {
    //     int class_id;
    //     float confidence;
    //     cv::Rect box;
    // };
    
    //将输入的图像进行预处理，返回一个格式化后的图像。
    cv::Mat yolo::format_yolov5(const cv::Mat &source) {
        int col = source.cols;
        int row = source.rows;
        int _max = MAX(col, row);
        cv::Mat result = cv::Mat::zeros(_max, _max, CV_8UC3);
        source.copyTo(result(cv::Rect(0, 0, col, row)));
        return result;
    }
 
 
    //YOLOV5网络的数据预处理以及前向推理（包括NMS处理）
    void yolo::detect(cv::Mat &image, cv::dnn::Net &net, std::vector<Detection> &output, const std::vector<std::string> &className) 
    {
        cv::Mat blob;
        auto input_image = yolo::format_yolov5(image);
        cv::dnn::blobFromImage(input_image, blob, 1./255., cv::Size(this->INPUT_WIDTH, this->INPUT_HEIGHT), cv::Scalar(), true, false);
        
        net.setInput(blob);  // blob-[3,640,640]
        std::vector<cv::Mat> outputs;
        // std::cout<< "2"<< std::endl;
        //网络计算到指定层（第二个参数指定的层），并返回该层的所有输出
        net.forward(outputs, net.getUnconnectedOutLayersNames()); // getUnconnectedOutLayersNames()返回具有未连接输出的层的名称,返回最终输出层
        // std::cout<< "3"<< std::endl;
        //计算x_factor和y_factor，用于后面还原  bounding box的位置和大小
        float x_factor = input_image.cols / this->INPUT_WIDTH;
        float y_factor = input_image.rows / this->INPUT_HEIGHT;
        // std::cout<< "4"<< std::endl;
        /*
        通过outputs[0]可以获得该输出层的结果，其中包含了该层所有的预测框的信息，包括预测框的位置、大小、置信度和类别概率。
        这些信息被保存在一个指向连续内存的地址中，可以通过.data来访问。
        outputs[0].data返回一个指向float类型的连续内存的指针，即该指针指向的是一个float类型的数组，其中包含了该层所有预测框的位置、大小、置信度和类别概率。
        因此，将该指针赋值给float* data后，就可以通过data来访问该数组中的每一个元素。
        同时，由于该数组是连续内存，因此可以通过指针的算术运算来访问该数组中的每一个元素，即使用data[i]来访问数组中第i个元素
        */
        float *data = (float *)outputs[0].data;
        //std::cout << "---------------------------------------:" << sizeof(&data) << std::endl;
        /*
        Yolov5s模型的输出大小为(1, 25200, 85)，其中：
        第一维是batch size，为1；
        第二维为每张输入图片生成的预测框数，即anchors数量 x (S1 x S1 + S2 x S2 + S3 x S3)，这里的S1, S2, S3分别为输出层的三个特征图的大小，取值为{80, 40, 20}，anchors数量为3，因此总的预测框数为25200；
        第三维为每个预测框的信息，包括4个坐标信息、1个置信度信息和80个类别得分信息，共85个信息。
        */
        const int dimensions = 85;
        const int rows = 25200;
        
        std::vector<int> class_ids;
        std::vector<float> confidences;
        std::vector<cv::Rect> boxes;
    
        // std::cout<< "4"<< std::endl;
        for (int i = 0; i < rows; ++i) {
            /*
            在C++中，指针使用[]操作符时，其作用与数组类似。
            当指针指向的是连续的内存区域时，可以使用[]操作符来访问该区域中的数据。
            例如，如果有一个指向float类型数据的指针p，我们可以通过p[i]来访问它所指向的内存中的第i个float类型的数据。
            这里的i是一个整数索引，指定了要访问的数据在内存中的偏移量。
            data[4]和data + 5可以分别访问指针所指向的内存中的第5个float类型的数据和从第6个float类型的数据开始的一段连续数据。
            */
            float confidence = data[4];
            if (confidence >= CONFIDENCE_THRESHOLD) {
    
                float * classes_scores = data + 5;
                /*
                cv::Mat的构造函数参数如下：
                第一个参数：矩阵的行数，这里设置为1，表示只有一行；
                第二个参数：矩阵的列数，这里设置为className.size()，表示有className.size()列；
                第三个参数：矩阵的数据类型，这里设置为CV_32FC1，表示元素类型为单通道32位浮点数；
                第四个参数：矩阵的数据指针，这里设置为classes_scores，表示矩阵的数据存储在classes_scores所指向的内存地址处，指向的是内存的首地址。
                */
                cv::Mat scores(1, className.size(), CV_32FC1, classes_scores);
                cv::Point class_id;
                double max_class_score;
    
                //获取最大类别分数以及其对应的索引
                minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
    
                //通过阈值进行筛选，将符合要求的类别、置信度以及框体进行保存
                if (max_class_score > SCORE_THRESHOLD) {
    
                    confidences.push_back(confidence);
    
                    class_ids.push_back(class_id.x);
    
                    float x = data[0];
                    float y = data[1];
                    float w = data[2];
                    float h = data[3];
                    int left = int((x - 0.5 * w) * x_factor);
                    int top = int((y - 0.5 * h) * y_factor);
                    int width = int(w * x_factor);
                    int height = int(h * y_factor);
                    boxes.push_back(cv::Rect(left, top, width, height));
                }
    
            }
            //一个边界框包含85个值————4个坐标信息、1个置信度信息和80个类别得分信息
            //data所指内存地址包含输出层所有预测框的位置、大小、置信度和类别概率，在yolov5s中共有25200个边界框，即data所指内存地址包含25200*85个值
            //在遍历一个边界框后，data指向需要向后移动85个位置，即 +85
            data += 85;
    
        }
    
        std::vector<int> nms_result;
        /*
        在目标检测任务中，一个目标可能会被多个边界框检测到，这些边界框可能会有不同的位置和大小，但表示同一个目标。
        非极大值抑制（Non-Maximum Suppression，NMS）是一种常用的方法，用于抑制这些重叠的边界框，只保留置信度最高的那个边界框，从而得到最终的目标检测结果。
        NMS的原理如下：
        首先，对所有的边界框按照其置信度进行排序，置信度最高的边界框排在最前面。
        从置信度最高的边界框开始，依次遍历其余边界框。
        对于当前遍历到的边界框，如果它与前面已经保留的边界框的重叠程度（通过计算IOU值）大于一定阈值（比如0.5），那么就将其抑制掉，不保留。
        继续遍历下一个边界框，重复上述过程，直到所有的边界框都被处理完毕。
        通过这样的处理，NMS可以抑制掉大量重叠的边界框，只保留最好的那个边界框，从而得到最终的目标检测结果。这种方法虽然简单，但是在实践中非常有效，已经被广泛应用于各种目标检测任务中。
        
        关于非极大值抑制，我在新的一篇进行了详细讲解，可在我们博客内容中搜索参考。
        */
        cv::dnn::NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, nms_result);
        
        //将经过NMS处理后的结果加载到const vector<Detection> output中
        for (int i = 0; i < nms_result.size(); i++) {
            int idx = nms_result[i];
            Detection result;
            result.class_id = class_ids[idx];
            result.confidence = confidences[idx];
            result.box = boxes[idx];
            output.push_back(result);
        }

    }
 
}

// int main(int argc, char **argv)
// {
//     // 加载class列表
//     std::vector<std::string> class_list = load_class_list();
 
//     //读取视频文件并判断是否成功打开
//     cv::Mat frame;
//     cv::Mat testframe;
//     cv::VideoCapture capture("../video/sample.mp4");
//     if (!capture.isOpened())
//     {
//         std::cerr << "Error opening video file\n";
//         return -1;
//     }

//     capture >> testframe;
//     int frame_width = testframe.cols;
//     int frame_height = testframe.rows;
//     cv::VideoWriter video("../output/outcpp.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 10, cv::Size(frame_width, frame_height));
 
//     //cv::Mat frame = cv::imread("misc/araras.jpg");
//     //判断是否使用CUDA,在选定使用CUDA前，请确保电脑支持GPU以及安装了CUDA、cudnn。
//     bool is_cuda = argc > 1 && strcmp(argv[1], "cuda") == 0;
 
//     //加载YOLOV5网络
//     cv::dnn::Net net;
//     load_net(net, is_cuda);
 
//     //创建高精度计时器
//     auto start = std::chrono::high_resolution_clock::now();
//     //纪录视频帧数
//     int frame_count = 0;
//     //计算FPS(每秒传输帧数)
//     float fps = -1;
//     int total_frames = 0;
 
//     while (true)
//     {
//         //读取视频帧, 关于关于视频处理应用也可参考我其他的opencv讲解内容
//         std::cout << "Start frame " << total_frames+1 << "\n";
//         capture.read(frame);
//         if (frame.empty())
//         {
//             std::cout << "End of stream\n";
//             break;
//         }
 
//         std::vector<Detection> output;
//         //YOLOV5S前向推理
//         detect(frame, net, output, class_list);
 
//         frame_count++;
//         total_frames++;
        
//         //检测的边界框总数
//         int detections = output.size();
 
//         for (int i = 0; i < detections; ++i)
//         {
 
//             auto detection = output[i];
 
//             auto box = detection.box;
//             auto classId = detection.class_id;
 
//             //通过取模运算为边界框选定颜色
//             const auto color = colors[classId % colors.size()];
 
//             //绘制边界框
//             cv::rectangle(frame, box, color, 3);
//             //绘制用于写类别的边框范围，一般就在边框的上面
//             cv::rectangle(frame, cv::Point(box.x, box.y - 20), cv::Point(box.x + box.width, box.y), color, cv::FILLED);
//             //在上面绘制的框界内写出类别
//             cv::putText(frame, class_list[classId].c_str(), cv::Point(box.x, box.y - 5), cv::FONT_HERSHEY_TRIPLEX, 0.8, cv::Scalar(0, 0, 0),1.5);
//         }
 
//         //根据帧数以及计时器结果计算FPS
//         if (frame_count >= 30)
//         {
 
//             auto end = std::chrono::high_resolution_clock::now();
//             fps = frame_count * 1000.0 / std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
 
//             frame_count = 0;
//             start = std::chrono::high_resolution_clock::now();
//         }
 
//         //如果FPS大于0，就在视频左上角写出来
//         if (fps > 0)
//         {
 
//             std::ostringstream fps_label;
//             fps_label << std::fixed << std::setprecision(2);
//             fps_label << "FPS: " << fps;
//             std::string fps_label_str = fps_label.str();
 
//             cv::putText(frame, fps_label_str.c_str(), cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
//         }
 
//         //cv::imshow("output", frame);
//         video.write(frame);

//         //std::string frame_path="../pics/frame"+std::to_string(frame_count)+".jpg";
//         //cv::imwrite(frame_path, frame);
 
//         //如果用户按下了键，那么capture.release()函数会释放视频捕获对象，跳出循环，关闭视频流。同时，程序输出一条消息提示用户程序已经结束。
//         if (cv::waitKey(1) != -1)
//         {
//             capture.release();
//             std::cout << "finished by user\n";
//             break;
//         }
//     }
 
//     //输出视频检测的总帧数
//     std::cout << "Total frames: " << total_frames << "\n";
//     video.release();
//     capture.release();
 
//     return 0;
// }