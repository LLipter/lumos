#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>

char* input_path = "../video/01.mp4";
AVFormatContext *pFormatCtx = NULL;
int video_stream_index = -1;
AVCodecParameters *pCodePara = NULL;
AVCodec *pCodec = NULL;
AVCodecContext *codec_ctx = NULL;


void cleanup(char* msg){
    if(pFormatCtx)
        avformat_free_context(pFormatCtx);
    if(codec_ctx)
        avcodec_free_context(codec_ctx);
    if(msg){
        perror(msg);
        exit(1);
    }
}

int main(int argc, char **argv) {
    //2、打开视频文件
    pFormatCtx = avformat_alloc_context();
    if ((avformat_open_input(&pFormatCtx, input_path, NULL, NULL)) < 0)
        cleanup("Cannot open input file");

    //3、获取视频信息
    if (avformat_find_stream_info(pFormatCtx, NULL) < 0)
        cleanup("Cannot find stream");

    //4、找到视频流的位置
    for (int i; i < pFormatCtx->nb_streams; i++) {
        if (pFormatCtx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO)
        {
            video_stream_index = i;
            break;
        }
    }
    if (video_stream_index == -1)
        cleanup("Cannot find stream index");

    //5、获取解码器
    pCodePara = pFormatCtx->streams[video_stream_index]->codecpar;
    int width = pCodePara->width;
    int height = pCodePara->height;
    pCodec = avcodec_find_decoder(pCodePara->codec_id);
    if (pCodec == NULL)
        cleanup("Cannot find decoder");

    //6、打开解码器
//    codec_ctx = avcodec_alloc_context3(pCodec);
//    avcodec_parameters_to_context(codec_ctx, pCodePara);
    if (avcodec_open2(pCodePara, pCodec, NULL) < 0)
        cleanup("Cannot open codec");





    cleanup(NULL);
    return 0;
}