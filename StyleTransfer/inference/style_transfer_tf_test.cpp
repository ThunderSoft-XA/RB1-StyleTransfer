#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

#include "inference_tf.hpp"

using namespace inference;

cv::Mat vector2mat(std::vector<float> output, cv::Size2d size = { 512,512 }) 
{
	cv::Mat out_result(size.height, size.width, CV_32FC3, cv::Scalar(0));
    out_result.convertTo(out_result, CV_8UC3, 255.0); 
	out_result.data = (uchar *)output.data();
	return out_result;
}


int main(int argcm, char **argv)
{
    Settings predict_setting;
    Settings transformer_setting;

    // predict_setting.model_name = "./magenta_arbitrary-image-stylization-v1-256_fp16_prediction_1.tflite";
    predict_setting.model_name = "./magenta_arbitrary-image-stylization-v1-256_int8_prediction_1.tflite";
    predict_setting.input_bmp_name = "./style23.jpg";

    transformer_setting.model_name = "./magenta_arbitrary-image-stylization-v1-256_fp16_transfer_1.tflite";
    transformer_setting.input_bmp_name = "./belfry-2611573_1280.jpg";

    std::cout << "predict_setting.input_bmp_name : " << predict_setting.input_bmp_name
        << "\ntransformer_setting.input_bmp_name : " << transformer_setting.input_bmp_name << std::endl;

    cv::Mat input_mat = cv::imread(predict_setting.input_bmp_name);
    
    cv::Size2d mat_size(input_mat.cols,input_mat.rows);
    std::vector<uint8_t> in = mat2vector(input_mat,mat_size);
    tf::TFInference predict_inference;
    tf::TFInference transformer_inference;

    predict_inference.setSettings(&predict_setting);
    transformer_inference.setSettings(&transformer_setting);

    predict_inference.loadModel();
    predict_inference.setInferenceParam();

    cv::Mat predict_mat = cv::imread(predict_setting.input_bmp_name);
    // cv::resize(predict_mat,predict_mat,cv::Size(256,256));
    cv::imwrite("predict.png",predict_mat);
    tf::InputPair predict_input_pair(0,predict_mat);
    tf::InputPairVec predict_input_pair_vec;//(predict_inference.getInputsNum());
    predict_input_pair_vec.push_back(predict_input_pair);
    predict_inference.loadData(predict_input_pair_vec);
    std::vector<float> predict_result_vec;
    predict_inference.inferenceModel<float>(predict_result_vec);
    std::cout << "predict_result_vec size : " << predict_result_vec.size() << std::endl;

    transformer_inference.loadModel();
    transformer_inference.setInferenceParam();

    tf::InputPair transformer_input_pair_0(1,cv::Mat(predict_result_vec));
    cv::Mat transformer_input_mat = cv::imread(transformer_setting.input_bmp_name);
    // cv::resize(transformer_input_mat,transformer_input_mat,cv::Size(384,384));
    tf::InputPair transformer_input_pair_1(0,transformer_input_mat);
    tf::InputPairVec transformer_input_pair_vec;//(transformer_inference.getInputsNum());
    transformer_input_pair_vec.push_back(transformer_input_pair_0);
    transformer_input_pair_vec.push_back(transformer_input_pair_1);
    transformer_inference.loadData(transformer_input_pair_vec);
    std::vector<float> transformer_result_vec;
    transformer_inference.inferenceModel<float>(transformer_result_vec);

    std::cout << "transformer_result_vec size : " << transformer_result_vec.size() << std::endl;

    // 384, 384, 3
    // cv::Mat result_mat = vector2mat(transformer_result_vec,{384,384});
    cv::Mat result_mat(cv::Size(384,384),CV_8UC3);
    // result_mat.data = (uchar *)transformer_result_vec.data();
    // result_mat.convertTo(result_mat,CV_8UC3);
    // cv::cvtColor(result_mat, result_mat, cv::COLOR_RGB2BGR);
    int piexl_index = 0;
    for(auto piexl_value : transformer_result_vec) {
        result_mat.data[piexl_index] = piexl_value * 255;
        piexl_index++;
    }
    cv::imwrite("./result_mat.png",result_mat);

    cv::Mat out_result(input_mat.cols, input_mat.rows, CV_8UC3);
    out_result.data = in.data();
    cv::cvtColor(out_result, out_result, cv::COLOR_RGB2BGR);
    cv::imwrite("./result_mat1.png",out_result);

    return 0;

}