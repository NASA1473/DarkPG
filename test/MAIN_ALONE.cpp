#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <vector>
#include <lime.h>
#include <yolo.h>
#include <inference.h> 

void showProgress(double progress) {
    static int last_progress = -1; // 上一次进度
    int current_progress = static_cast<int>(progress * 100); // 当前进度

    if (current_progress != last_progress) { // 当进度变化时更新控制台信息
        std::cout << "\rVideo Write Progress: " << current_progress << "%" << std::flush;
        last_progress = current_progress;
    }

    if (current_progress == 100) { // 训练完成时输出结束信息
        std::cout << "\rVideo Write finished!" << std::endl;
    }
}

void video_demo(std::string videoName){
    lime_feature::lime* l;
    l = new lime_feature::lime(1, 0.15, 1.07, 0.6, 1, 0.8);
    bool runOnGPU = false;
    int frame_width = 640;
    int frame_height = 640;
    inference_feature::Inference inf("../source/models/yolov8n.onnx", cv::Size(frame_width, frame_height),
                                    "../source/classes/classes.txt", runOnGPU);

    std::string input_path = "../test/data/" + videoName;
    std::string output_path = "../test/result/result_" + videoName;

    cv::VideoCapture capture(input_path);
    if (!capture.isOpened()) {
        std::cout << "Failed to open video file!" << std::endl;
        return;
    }
    cv::Mat frame;
    int nframes = (int)capture.get(cv::CAP_PROP_FRAME_COUNT);
    int width = (int)capture.get(cv::CAP_PROP_FRAME_WIDTH);
    int height = (int)capture.get(cv::CAP_PROP_FRAME_HEIGHT);
	double fps = capture.get(cv::CAP_PROP_FPS);
    int rate = 0;
    cv::VideoWriter writer(output_path, capture.get(cv::CAP_PROP_FOURCC), fps, cv::Size(width, height), true);

    std::cout << "Video has " << nframes << " frames, with dimensions (" << width << ", " << height << ") and FPS " << fps << "." << std::endl;
    auto BP1 = std::chrono::high_resolution_clock::now();
    while(capture.read(frame)) {
        if (frame.empty()) { 
            break; 
        }
        double progress = static_cast<double>(rate) / (nframes-1);
        showProgress(progress);
        l->loadMatrix(frame);
        cv::Mat img_out = l->run_withoutNeon();
        std::vector<inference_feature::Detection> output = inf.runInference(img_out);
        int detections = output.size();
        for (int i = 0; i < detections; ++i)
        {
            inference_feature::Detection detection = output[i];

            cv::Rect box = detection.box;
            cv::Scalar color = detection.color;

            // Detection box
            cv::rectangle(img_out, box, color, 2);

            // Detection box text
            std::string classString = detection.className + ' ' + std::to_string(detection.confidence).substr(0, 4);
            cv::Size textSize = cv::getTextSize(classString, cv::FONT_HERSHEY_DUPLEX, 1, 2, 0);
            cv::Rect textBox(box.x, box.y - 40, textSize.width + 10, textSize.height + 20);

            cv::rectangle(img_out, textBox, color, cv::FILLED);
            cv::putText(img_out, classString, cv::Point(box.x + 5, box.y - 10), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 0), 2, 0);
        }
        std::string Info="Current number of items: "+std::to_string(detections);
        cv::putText(img_out, Info, cv::Point(10, 60), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 0), 2, 0);
        writer.write(img_out);
        rate++;
    }
    auto BP2 = std::chrono::high_resolution_clock::now();

    capture.release();
	writer.release();

    std::chrono::duration<double> runtime1 = BP2 - BP1;
    std::cout << std::endl;
    std::cout << "The Program run on Single core without Neon cost " << runtime1.count() << "s" << std::endl;
}

void video_demo_neon(std::string videoName){
    lime_feature::lime* l;
    l = new lime_feature::lime(1, 0.15, 1.07, 0.6, 1, 0.8);
    bool runOnGPU = false;
    int frame_width = 640;
    int frame_height = 640;
    inference_feature::Inference inf("../source/models/yolov8n.onnx", cv::Size(frame_width, frame_height),
                                    "../source/classes/classes.txt", runOnGPU);

    std::string input_path = "../test/data/" + videoName;
    std::string output_path = "../test/result/result_" + videoName;

    cv::VideoCapture capture(input_path);
    if (!capture.isOpened()) {
        std::cout << "Failed to open video file!" << std::endl;
        return;
    }
    cv::Mat frame;
    int nframes = (int)capture.get(cv::CAP_PROP_FRAME_COUNT);
    int width = (int)capture.get(cv::CAP_PROP_FRAME_WIDTH);
    int height = (int)capture.get(cv::CAP_PROP_FRAME_HEIGHT);
	double fps = capture.get(cv::CAP_PROP_FPS);
    int rate = 0;
    cv::VideoWriter writer(output_path, capture.get(cv::CAP_PROP_FOURCC), fps, cv::Size(width, height), true);

    std::cout << "Video has " << nframes << " frames, with dimensions (" << width << ", " << height << ") and FPS " << fps << "." << std::endl;
    auto BP1 = std::chrono::high_resolution_clock::now();
    while(capture.read(frame)) {
        if (frame.empty()) { 
            break; 
        } 
        double progress = static_cast<double>(rate) / (nframes-1);
        showProgress(progress);
        l->loadMatrix(frame);
        cv::Mat img_out = l->run();
        std::vector<inference_feature::Detection> output = inf.runInference(img_out);
        int detections = output.size();
        for (int i = 0; i < detections; ++i)
        {
            inference_feature::Detection detection = output[i];

            cv::Rect box = detection.box;
            cv::Scalar color = detection.color;

            // Detection box
            cv::rectangle(img_out, box, color, 2);

            // Detection box text
            std::string classString = detection.className + ' ' + std::to_string(detection.confidence).substr(0, 4);
            cv::Size textSize = cv::getTextSize(classString, cv::FONT_HERSHEY_DUPLEX, 1, 2, 0);
            cv::Rect textBox(box.x, box.y - 40, textSize.width + 10, textSize.height + 20);

            cv::rectangle(img_out, textBox, color, cv::FILLED);
            cv::putText(img_out, classString, cv::Point(box.x + 5, box.y - 10), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 0), 2, 0);
        }
        std::string Info="Current number of items: "+std::to_string(detections);
        cv::putText(img_out, Info, cv::Point(10, 60), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 0), 2, 0);
        writer.write(img_out);
        rate++;
    }
    auto BP2 = std::chrono::high_resolution_clock::now();
    
    capture.release();
	writer.release();

    std::chrono::duration<double> runtime2 = BP2 - BP1;
    std::cout << std::endl;
    std::cout << "The Program run on Single core with Neon cost " << runtime2.count() << "s" << std::endl;
}

int main(int argc, char **argv){ 
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " video_name [--neon]\n";
        return -1;
    }
    
    std::string videoName = argv[1];

    bool neonFlag = false;
    if (argc > 2 && std::string(argv[2]) == "--neon") {
        neonFlag = true;
    }

    if(neonFlag){
        video_demo_neon(videoName); 
    }
    else{
        video_demo(videoName); 
    }

    return 0; 
}
