#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <vector>
#include <lime.h>
#include <yolo.h>
#include <inference.h> 
#include <mpi.h>

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


int main(int argc, char **argv){ 
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); //获取当前进程的rank
    MPI_Comm_size(MPI_COMM_WORLD, &size); //获取进程数

    cv::VideoCapture capture("../test/data/YOLO_test.mp4");
    bool runOnGPU = false;
    
    if (!capture.isOpened()) { // 检查是否成功打开
        std::cout << "Failed to open video file!" << std::endl;
        return 0;
    }

    int nframes = (int)capture.get(cv::CAP_PROP_FRAME_COUNT);
    int width = (int)capture.get(cv::CAP_PROP_FRAME_WIDTH);
    int height = (int)capture.get(cv::CAP_PROP_FRAME_HEIGHT);
    double fps = capture.get(cv::CAP_PROP_FPS);
    const int buffer_size = width * height * 3; 
    double start, end;

    int frames_per_process = nframes / (size - 1);  //8
    int remaining_frames = nframes % (size - 1);    //4 

    int this_frames_per_process = 0;
    if (rank != 0){
        if (rank <= remaining_frames) {
            this_frames_per_process = frames_per_process + 1;
        }
        else{
            this_frames_per_process = frames_per_process;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD); 
    start = MPI_Wtime();

    if (rank == 0) { // 接收方
        cv::VideoWriter writer("../test/result/YOLO_test.mp4", capture.get(cv::CAP_PROP_FOURCC), fps, cv::Size(width, height), true);
        std::cout << "Video has " << nframes << " frames, with dimensions (" << width << ", " << height << ") and FPS " << fps << "." << std::endl;

        std::vector<cv::Mat> received_frames(nframes);
        std::vector<bool> is_received(nframes,false);

        int completed = 0;
        int pos = 0;
        int tag = 0;    // 消息标签
        int source = 0;
        MPI_Status status;
        MPI_Request request;

        while(completed < nframes){
            double progress = static_cast<double>(completed) / (nframes);
            showProgress(progress);

            int flag = 0;
            bool all_received = true;
            MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, &status); // 查询消息
            if(flag){
                int source = status.MPI_SOURCE;
                tag = status.MPI_TAG;

                cv::Mat new_frame(height, width, CV_8UC3);
                MPI_Irecv(new_frame.data, buffer_size, MPI_UNSIGNED_CHAR, source, tag, MPI_COMM_WORLD, &request);
                MPI_Wait(&request, &status);

                received_frames[tag] = new_frame;
                is_received[tag] = true;
                for(int i = pos;i <= tag;i++){
                    if(is_received[i] == false){
                        all_received = false;
                        break;
                    }
                }
                if(all_received == true){
                    for(int i = pos;i <= tag;i++){
                        writer.write(received_frames[i]);
                        // std::cout << "Process 0 received message, frame " << i << std::endl;
                    }
                    pos = tag + 1;
                }
                completed ++ ;
            }
        }

        for(int i = pos;i < nframes;i++){
            writer.write(received_frames[i]);
            //std::cout << "Process 0 received message, frame " << i << std::endl;
        }
        writer.release();
    }
    else { // 发送方
        lime_feature::lime* l;
        l = new lime_feature::lime(1, 0.15, 1.07, 0.6, 1, 0.8);
        int frame_width = 640;
        int frame_height = 640;
        inference_feature::Inference inf("../source/models/yolov8n.onnx", cv::Size(frame_width, frame_height),
                                    "../source/classes/classes.txt", runOnGPU);
        cv::Mat frame;
        for (int i = 0; i < this_frames_per_process; i++) {
            int ad_change = i * (size-1) + rank - 1;
            capture.set(cv::CAP_PROP_POS_FRAMES, ad_change);
            capture.read(frame);
            cv::Mat img_out;
            if (!frame.empty()) {
                l->loadMatrix(frame);
                img_out = l->run_withoutNeon();
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
            }
            MPI_Request request;
            int tag = ad_change;
            MPI_Isend(img_out.data, buffer_size, MPI_UNSIGNED_CHAR, 0, tag, MPI_COMM_WORLD, &request); // 非阻塞发送消息
            MPI_Wait(&request, MPI_STATUS_IGNORE); //等待发送完成
            //std::cout << "Process " << rank << " sent message, frame " << ad_change << std::endl; // 输出发送的消息和轮数
        }

        delete l;
        //delete b;

    }

    MPI_Barrier(MPI_COMM_WORLD);
    end = MPI_Wtime();

    capture.release();
    MPI_Finalize();

    if (rank == 0) { 
        std::cout << std::endl;
        printf("The Program run on Multi core without Neon cost %f s\n", end-start);
    }
 
    return 0; 
}

/*
int main(int argc, char **argv){ 
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); //获取当前进程的rank
    MPI_Comm_size(MPI_COMM_WORLD, &size); //获取进程数

    cv::VideoCapture capture("../test/data/test2.mp4");
    
    if (!capture.isOpened()) { // 检查是否成功打开
        std::cout << "Failed to open video file!" << std::endl;
        return 0;
    }

    int nframes = (int)capture.get(cv::CAP_PROP_FRAME_COUNT);
    int width = (int)capture.get(cv::CAP_PROP_FRAME_WIDTH);
    int height = (int)capture.get(cv::CAP_PROP_FRAME_HEIGHT);
    double fps = capture.get(cv::CAP_PROP_FPS);
    const int buffer_size = width * height * 3; 
    double start, end;

    int frames_per_process = nframes / (size - 1);  //8
    int remaining_frames = nframes % (size - 1);    //4 

    int this_frames_per_process = 0;
    if (rank != 0){
        if (rank <= remaining_frames) {
            this_frames_per_process = frames_per_process + 1;
        }
        else{
            this_frames_per_process = frames_per_process;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD); 
    start = MPI_Wtime();

    if (rank == 0) { // 接收方
        cv::VideoWriter writer("../test/result/test2.mp4", capture.get(cv::CAP_PROP_FOURCC), fps, cv::Size(width, height), true);
        std::cout << "Video has " << nframes << " frames, with dimensions (" << width << ", " << height << ") and FPS " << fps << "." << std::endl;

        std::vector<cv::Mat> received_frames(nframes);
        std::vector<bool> is_received(nframes,false);

        int completed = 0;
        int pos = 0;
        int tag = 0;    // 消息标签
        int source = 0;
        MPI_Status status;
        MPI_Request request;

        while(completed < nframes){
            int flag = 0;
            bool all_received = true;
            MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, &status); // 查询消息
            if(flag){
                int source = status.MPI_SOURCE;
                tag = status.MPI_TAG;

                cv::Mat new_frame(height, width, CV_8UC3);
                MPI_Irecv(new_frame.data, buffer_size, MPI_UNSIGNED_CHAR, source, tag, MPI_COMM_WORLD, &request);
                MPI_Wait(&request, &status);

                received_frames[tag] = new_frame;
                is_received[tag] = true;
                for(int i = pos;i <= tag;i++){
                    if(is_received[i] == false){
                        all_received = false;
                        break;
                    }
                }
                if(all_received == true){
                    for(int i = pos;i <= tag;i++){
                        writer.write(received_frames[i]);
                        std::cout << "Process 0 received message, frame " << i << std::endl;
                    }
                    pos = tag + 1;
                }
                completed ++ ;
            }
        }

        for(int i = pos;i < nframes;i++){
            writer.write(received_frames[i]);
            std::cout << "Process 0 received message, frame " << i << std::endl;
        }
        writer.release();
    }
    else { // 发送方
        lime_feature::lime* l;
        l = new lime_feature::lime(1, 0.15, 1.07, 0.6, 1, 0.8);
        yolo_feature::yolo* b;
        b = new yolo_feature::yolo();
        std::vector<std::string> class_list = b->load_class_list();
        cv::dnn::Net net;
        b->load_net(net, 0);
        cv::Mat frame;

        for (int i = 0; i < this_frames_per_process; i++) {
            int ad_change = i * (size-1) + rank - 1;
            capture.set(cv::CAP_PROP_POS_FRAMES, ad_change);
            capture.read(frame);
            cv::Mat img_out;
            if (!frame.empty()) {
                l->loadMatrix(frame);
                img_out = l->run();
                std::vector<yolo_feature::yolo::Detection> output;
                b->detect(img_out, net, output, class_list);
                int detections = output.size();
                for (int a = 0; a < detections; ++a)
                {
                    auto detection = output[a];
                    auto box = detection.box;
                    auto classId = detection.class_id;
                    //通过取模运算为边界框选定颜色
                    const auto color = b->colors[classId % b->colors.size()];
                    //绘制边界框
                    cv::rectangle(img_out, box, color, 3);
                    //绘制用于写类别的边框范围，一般就在边框的上面
                    cv::rectangle(img_out, cv::Point(box.x, box.y - 20), cv::Point(box.x + box.width, box.y), color, cv::FILLED);
                    //在上面绘制的框界内写出类别
                    cv::putText(img_out, class_list[classId].c_str(), cv::Point(box.x, box.y - 5), cv::FONT_HERSHEY_TRIPLEX, 0.8, cv::Scalar(0, 0, 0),1.5);
                }
            }
            MPI_Request request;
            int tag = ad_change;
            MPI_Isend(img_out.data, buffer_size, MPI_UNSIGNED_CHAR, 0, tag, MPI_COMM_WORLD, &request); // 非阻塞发送消息
            MPI_Wait(&request, MPI_STATUS_IGNORE); //等待发送完成
            std::cout << "Process " << rank << " sent message, frame " << ad_change << std::endl; // 输出发送的消息和轮数
            // round++; // 更新已发送的轮数
        }

        delete l;
        delete b;

    }


    MPI_Barrier(MPI_COMM_WORLD);
    end = MPI_Wtime();

    capture.release();
    MPI_Finalize();

    if (rank == 0) { 
        printf("Runtime = %f s\n", end-start);
    }
 
    return 0; 
}
*/