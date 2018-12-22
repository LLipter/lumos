#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>

char* input_path = "../video/01.mp4";
AVFormatContext *pFormatCtx = NULL;
int video_stream_index = -1;

void cleanup(char* msg){
    if(pFormatCtx){
        avformat_free_context(pFormatCtx);
        avformat_close_input(&pFormatCtx);
    }
    perror(msg);
    exit(1);
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



    return 0;
}